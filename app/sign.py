"""符号限制的结构识别：校验、正交旋转搜索、等价类与去重。

约定（见 docs/data-contract.md）：
  B = P Q，P 为残差协方差 sigma 的固定下三角 Cholesky 因子（正对角），
  Q 为 n×n 正交矩阵；变量 i 对冲击 j 在第 h 期的响应为 [Phi_h P Q]_{i,j}。
  一条限制 (shock, variable, horizon, sign) 要求 sign * response >= margin。

Q 的取值域是完整的 O(n)，而不是只取 det=+1 的 SO(n)：只取 SO(n) 会系统性
漏掉“只翻转某一列”的方案，而这类方案在限制只覆盖部分冲击时恰恰可行。
搜索时把 O(n) 拆成 SO(n) 与 SO(n)·F 两个连通分支分别采样。

解释边界：
  * Haar 随机搜索给出的是“搜索到的可行方案”，不是识别集合全部；没搜到
    可行解不能解释为限制方案矛盾。矛盾判定只来自 certify 模块。
  * 限制画像相同的冲击，其标签互换给出同一识别方案；无限制冲击的列在
    其正交补子空间内只能识别到任意正交变换（连续等价类），因此其逐期
    响应没有可报告的“范围”，只能报告为不可识别。
"""
import hashlib
import json
import math
import random

from . import linalg as la


class RestrictionError(ValueError):
    """限制方案本身无效，或响应期数超出已计算范围。"""


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


# ------------------------------------------------------------------ 方案校验

def compile_restrictions(metadata, variables):
    """校验并编译限制方案，返回 (restrictions, margin, fingerprint, version)。

    restrictions 为 (shock, variable_index, horizon, sign±1) 元组列表，按
    三元组排序；同一三元组重复且符号相反即拒绝。
    """
    if not isinstance(metadata, dict):
        raise RestrictionError('metadata应为对象')
    raw = metadata.get('restrictions')
    if not isinstance(raw, list):
        raise RestrictionError('metadata.restrictions应为数组')
    margin = metadata.get('positive_margin', 0.0)
    if isinstance(margin, bool) or not isinstance(margin, (int, float)) or not math.isfinite(margin) or margin < 0:
        raise RestrictionError('positive_margin应为非负有限数')
    n = len(variables)
    seen = {}
    compiled = []
    for item in raw:
        if not isinstance(item, dict):
            raise RestrictionError('每条限制应为对象')
        shock = item.get('shock')
        variable = item.get('variable')
        horizon = item.get('horizon')
        sign = item.get('sign')
        if isinstance(shock, bool) or not isinstance(shock, int) or not 0 <= shock < n:
            raise RestrictionError('限制的shock索引无效')
        if variable not in variables:
            raise RestrictionError('限制引用了未知变量')
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 0:
            raise RestrictionError('限制的horizon应为非负整数')
        if sign not in ('+', '-'):
            raise RestrictionError('限制的sign应为+或-')
        key = (shock, variables.index(variable), horizon)
        value = 1 if sign == '+' else -1
        if key in seen and seen[key] != value:
            raise RestrictionError('同一冲击/变量/期数的符号限制冲突')
        if key in seen:
            continue  # 重复且同号的限制是同一约束，只保留一条
        seen[key] = value
        compiled.append((shock, variables.index(variable), horizon, value))
    compiled.sort()
    fingerprint = hashlib.sha256(_canonical({
        'variables': list(variables),
        'margin': margin,
        'restrictions': [[s, v, h, sg] for s, v, h, sg in compiled],
    }).encode('utf-8')).hexdigest()
    version = metadata.get('restriction_version', metadata.get('scheme_version'))
    if version is not None and (isinstance(version, bool) or not isinstance(version, (str, int))):
        raise RestrictionError('restriction_version/scheme_version应为字符串或整数')
    return compiled, float(margin), fingerprint, version


def restriction_profiles(restrictions, n):
    profiles = [[] for _ in range(n)]
    for shock, var, horizon, sign in restrictions:
        profiles[shock].append((var, horizon, sign))
    return [tuple(sorted(profile)) for profile in profiles]


def label_equivalence_groups(restrictions, n):
    """限制画像相同的冲击分组；同组冲击的标签不可区分。"""
    groups = {}
    for j, profile in enumerate(restriction_profiles(restrictions, n)):
        groups.setdefault(profile, []).append(j)
    return [group for group in groups.values() if len(group) > 1]


def unrestricted_shocks(restrictions, n):
    profiles = restriction_profiles(restrictions, n)
    return [j for j in range(n) if not profiles[j]]


def max_restriction_horizon(restrictions):
    return max((horizon for _, _, horizon, _ in restrictions), default=0)


# ------------------------------------------------------------------ 约化式映射

def var_ma(est, horizons):
    """Phi_0..Phi_H，Phi_0=I，Phi_h = sum_l A_l Phi_{h-l}。"""
    n, p = est.n, est.lags
    coefficients = []
    for lag in range(1, p + 1):
        coefficients.append([[est.beta[1 + n * (lag - 1) + j][i] for j in range(n)] for i in range(n)])
    phi = [la.eye(n)]
    for h in range(1, horizons + 1):
        cur = la.zeros(n, n)
        for lag in range(1, min(p, h) + 1):
            cur = la.add(cur, la.matmul(coefficients[lag - 1], phi[h - lag]))
        phi.append(cur)
    return phi


class Mapping:
    """固定约化式参数后的响应映射：给定 Q 产生 B 与各期响应。"""

    def __init__(self, est):
        self.est = est
        self.cholesky = la.cholesky(est.sigma)
        self.n = est.n
        self._phi = {}

    def response(self, q, horizons):
        phi = self._phi.get(horizons)
        if phi is None:
            phi = var_ma(self.est, horizons)
            self._phi[horizons] = phi
        b = la.matmul(self.cholesky, q)
        return b, [la.matmul(ph, b) for ph in phi]


def response_value(theta, restriction):
    shock, var, horizon, _ = restriction
    if horizon >= len(theta):
        raise RestrictionError('响应期数超出已计算范围')
    return theta[horizon][var][shock]


def check_restrictions(theta, restrictions, margin):
    rows = []
    accepted = True
    for restriction in restrictions:
        signed = restriction[3] * response_value(theta, restriction)
        ok = signed >= margin
        accepted = accepted and ok
        rows.append({'shock': restriction[0], 'variable_index': restriction[1],
                     'horizon': restriction[2], 'sign': '+' if restriction[3] > 0 else '-',
                     'signed_response': signed, 'margin': margin, 'satisfied': ok})
    return accepted, rows


def worst_violation(theta, restrictions, margin):
    """最大违反量；<=0 表示全部满足。"""
    return max((margin - restriction[3] * response_value(theta, restriction)
                for restriction in restrictions), default=-0.0)


# ------------------------------------------------------------------ 正交采样

class Reflection:
    """det=-1 的固定反射 diag(-1, 1, ..., 1)，用于覆盖 O(n) 的 det=-1 分支。"""

    @staticmethod
    def matrix(n):
        m = la.eye(n)
        m[0][0] = -1.0
        return m


def haar_so(n, rng):
    """SO(n) 上的 Haar 样本：正态矩阵 QR，R 对角归正。"""
    a = [[rng.gauss(0.0, 1.0) for _ in range(n)] for _ in range(n)]
    q, r = _qr(a)
    for j in range(n):
        if r[j][j] < 0.0:
            for i in range(n):
                q[i][j] = -q[i][j]
    return q


def haar_o(n, rng):
    """O(n) 上的 Haar 样本：两个连通分支等概率。"""
    q = haar_so(n, rng)
    if rng.random() < 0.5:
        f = Reflection.matrix(n)
        q = la.matmul(q, f)
    return q


def _qr(a):
    n = len(a)
    q = la.zeros(n, n)
    r = la.zeros(n, n)
    for j in range(n):
        v = [a[i][j] for i in range(n)]
        for k in range(j):
            dot = sum(q[i][k] * a[i][j] for i in range(n))
            r[k][j] = dot
            v = [v[i] - dot * q[i][k] for i in range(n)]
        nv = math.sqrt(sum(x * x for x in v))
        if nv < 1e-300:
            raise RestrictionError('随机正交矩阵退化，请增加随机次数')
        r[j][j] = nv
        for i in range(n):
            q[i][j] = v[i] / nv
    return q, r


# ------------------------------------------------------------------ 等价与去重

def canonicalize(q, theta, restrictions, n, digits=8):
    """把无限制列的整体符号固定为最大绝对值元素为正，返回 (q, theta, flips)。

    有限制的列不翻转（翻转会破坏限制）。返回的 q 不做舍入，保留正交性；
    舍入只发生在签名哈希中。该规范化只用于去重与表示，不改变可行集合。
    """
    profiles = restriction_profiles(restrictions, n)
    flips = [1] * n
    for j in range(n):
        if profiles[j]:
            continue
        column = [q[i][j] for i in range(n)]
        pivot = max(range(n), key=lambda i: abs(column[i]))
        if column[pivot] < 0.0:
            flips[j] = -1
    qc = [[q[i][j] * flips[j] for j in range(n)] for i in range(n)]
    tc = [[[theta[h][i][j] * flips[j] for j in range(n)] for i in range(n)] for h in range(len(theta))]
    return qc, tc, flips


def canonical_signature(q_canonical, restrictions, n, digits=8):
    """规范正交矩阵的哈希；对限制画像相同的冲击标签置换不变。"""
    columns = [tuple(round(q_canonical[i][j], digits) for i in range(n)) for j in range(n)]
    order = list(range(n))
    for group in label_equivalence_groups(restrictions, n):
        ordered = sorted(group, key=lambda j: columns[j])
        for pos, j in zip(sorted(group), ordered):
            order[pos] = j
    payload = [columns[order[k]] for k in range(n)]
    return hashlib.sha256(_canonical(payload).encode('utf-8')).hexdigest(), order


# ------------------------------------------------------------------ 随机搜索

def search(est, restrictions, horizons, target, max_attempts, seed, margin=None):
    """Haar O(n) 随机搜索，返回可行的样本方案与搜索统计。

    只有在同一个 Q 下同时满足全部限制的方案才会被接受；不同冲击的列
    不能分别挑选。accepted 中保留规范代表元与原始 Q 的符号记录。
    """
    if margin is None:
        raise RestrictionError('搜索需要明确的符号裕度')
    if target < 0 or max_attempts < 0:
        raise RestrictionError('搜索次数不能为负')
    mapping = Mapping(est)
    n = est.n
    rng = random.Random(seed)
    accepted = []
    seen = set()
    duplicate_count = 0
    best_violation = float('inf')
    attempts = 0
    while attempts < max_attempts and len(accepted) < target:
        attempts += 1
        try:
            q = haar_o(n, rng)
        except RestrictionError:
            continue
        _, theta = mapping.response(q, horizons)
        ok, _ = check_restrictions(theta, restrictions, margin)
        if not ok:
            best_violation = min(best_violation, worst_violation(theta, restrictions, margin))
            continue
        qc, tc, flips = canonicalize(q, theta, restrictions, n)
        signature, order = canonical_signature(qc, restrictions, n)
        if signature in seen:
            duplicate_count += 1
            continue
        seen.add(signature)
        accepted.append({'q': qc, 'theta': tc, 'flips': flips,
                         'signature': signature, 'label_order': order})
    return {
        'accepted': accepted,
        'attempts': attempts,
        'accepted_count': len(accepted),
        'duplicate_count': duplicate_count,
        'best_remaining_violation': None if best_violation == float('inf') else best_violation,
        'margin': margin,
        'seed': seed,
        'status': 'found_feasible_sample' if accepted else 'no_feasible_sample_found',
        'note': '随机搜索只给出可行样本方案；未找到不等于限制方案不可行，'
                '不可行判定须由 certify 的证书给出。',
    }


# ------------------------------------------------------------------ 结果汇总

def summarize_responses(variables, restrictions, accepted, horizons):
    """按冲击/变量/期数汇总“搜索到的范围”，并标注识别状态。

    * 有约束且不与其它冲击同画像：报告搜索到的范围（searched_range）。
    * 限制画像相同的冲击组：标签不可区分，报告组内并集范围，
      标注 label_not_distinguishable。
    * 无限制冲击：在其正交补子空间内可任意旋转，报逐期范围没有意义，
      只报告 not_identified。
    """
    if not accepted:
        return []
    n = len(variables)
    groups = label_equivalence_groups(restrictions, n)
    group_of = {}
    for group in groups:
        for j in group:
            group_of[j] = group
    unrestricted = set(unrestricted_shocks(restrictions, n))
    profile_count = {}
    for j, profile in enumerate(restriction_profiles(restrictions, n)):
        if not profile:
            continue
        profile_count.setdefault(profile, []).append(j)

    rows = []
    for j in range(n):
        if j in unrestricted:
            rows.append({'shock': j, 'status': 'not_identified',
                         'reason': '该冲击无符号限制，其列只在正交补内识别到任意正交变换，'
                                   '逐期响应没有可报告的集合范围'})
            continue
        members = group_of.get(j, [j])
        for i in range(n):
            for h in range(horizons + 1):
                values = [a['theta'][h][i][m] for a in accepted for m in members]
                rows.append({
                    'shock': j, 'variable': variables[i], 'horizon': h,
                    'status': 'searched_range' if len(members) == 1 else 'label_not_distinguishable',
                    'label_group': list(members) if len(members) > 1 else None,
                    'found_min': min(values), 'found_max': max(values),
                    'found_median': _median(values), 'draw_count': len(values),
                    'kind': 'searched_range',
                    'note': '仅反映随机搜索命中的方案，不是认证的识别集合边界',
                })
    return rows


def _median(values):
    s = sorted(values)
    m = len(s)
    if m == 0:
        return None
    if m % 2:
        return s[m // 2]
    return 0.5 * (s[m // 2 - 1] + s[m // 2])


def scheme_summary(restrictions, n, margin, fingerprint, version):
    profiles = restriction_profiles(restrictions, n)
    unrestricted = unrestricted_shocks(restrictions, n)
    groups = label_equivalence_groups(restrictions, n)
    notes = []
    if unrestricted:
        notes.append('无限制冲击的响应与标签不可由符号限制识别')
    if groups:
        notes.append('同画像冲击的标签互换给出同一识别集合，不能当作独立证据')
    return {
        'fingerprint': fingerprint,
        'version': version,
        'positive_margin': margin,
        'restriction_count': len(restrictions),
        'per_shock_restrictions': [len(profile) for profile in profiles],
        'unrestricted_shocks': unrestricted,
        'label_equivalence_groups': groups,
        'max_horizon': max_restriction_horizon(restrictions),
        'sign_and_label_equivalence': {
            'sign_flips': '有限制列的符号由限制固定；无限制列在去重时固定为最大绝对值元素为正，'
                          '因此同一方案不会因整列变号被重复计数',
            'label_swaps': '限制画像相同的冲击组内标签互换为同一方案，已按组报告',
            'unrestricted_rotation': '无限制冲击只识别到正交补内的任意正交变换，'
                                     '其逐期响应不构成可报告集合',
        },
        'notes': notes,
    }
