"""给定约化式参数下的识别集合认证。

维度与做法：
  n=1: Q ∈ {+1, -1}，两个孤立点，穷举即精确。
  n=2: det=+1 分支 Q(a)=[[cos a,-sin a],[sin a,cos a]]，
       det=-1 分支 Q(a)=[[bs*cos a, -sin a],[bs*sin a, cos a]]（bs=-1），a∈[0,2π)。
       每条限制对角度是 a sin t + b cos t，可行集是圆弧的精确并集，边界由
       反余弦解析给出；内=外=精确边界。
  n=3: det=+1 分支为 ZYZ 欧拉角 Q=Rz(α)Ry(β)Rz(γ)，
       α,γ∈[0,2π)，β∈[0,π]；det=-1 分支为 SO(3)·diag(-1,1,1)。
       均匀网格完整覆盖，再对未决盒做区间二分精化。

n=3 证书如何复核：
  1. 参数域先被一张均匀网格完整覆盖；每个网格盒再用二分递归精化。
  2. 每条符号限制都是“响应”的线性泛函，响应又是 Q 的连续函数；用区间
     算术在盒上给出该泛函的上、下界。
  3. 所有限制下界都 ≥ margin：整盒可行，计入内近似，并检查盒心见证点。
  4. 至少一条限制上界 < margin：整盒不可行，排除。
  5. 其余为未决盒，只能进外近似。
  因此  内近似响应范围 ⊆ 真实识别集合响应范围 ⊆ 外近似响应范围。
  未决测度为 0 时内外一致，才可称认证边界。网格完整覆盖保证外近似的
  包含关系；只做随机抽样没有这个性质。

若所有盒都不可行且没有未决盒，则给出“限制方案不可行”的证书：参数域被
有限个盒覆盖，且每个盒上至少一条限制严格不可满足。随机搜索未命中不构成
不可行证据。
"""
import math

from . import linalg as la
from .sign import Mapping, check_restrictions, var_ma

TWO_PI = 2.0 * math.pi
HALF_PI = 0.5 * math.pi
SUPPORTED_DIMENSIONS = (1, 2, 3)


class CertificateError(ValueError):
    """无法给出证书，例如维度不受支持或响应期数不足。"""


def _down(x):
    for _ in range(6):
        x = math.nextafter(x, -math.inf)
    return x


def _up(x):
    for _ in range(6):
        x = math.nextafter(x, math.inf)
    return x


class Iv:
    """闭区间；乘加向两侧外扩 6 个 ULP。三角函数界按解析极值给出。"""
    __slots__ = ('lo', 'hi')

    def __init__(self, lo, hi):
        if lo > hi:
            lo, hi = hi, lo
        self.lo = lo
        self.hi = hi

    def __add__(self, other):
        o = _as_iv(other)
        return Iv(_down(self.lo + o.lo), _up(self.hi + o.hi))

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        o = _as_iv(other)
        return Iv(_down(self.lo - o.hi), _up(self.hi - o.lo))

    def __rsub__(self, other):
        return _as_iv(other).__sub__(self)

    def __neg__(self):
        return Iv(-self.hi, -self.lo)

    def __mul__(self, other):
        o = _as_iv(other)
        if not (math.isfinite(self.lo) and math.isfinite(self.hi)
                and math.isfinite(o.lo) and math.isfinite(o.hi)):
            return Iv(-math.inf, math.inf)
        products = (self.lo * o.lo, self.lo * o.hi, self.hi * o.lo, self.hi * o.hi)
        return Iv(_down(min(products)), _up(max(products)))

    def __rmul__(self, other):
        return self.__mul__(other)

    def width(self):
        return self.hi - self.lo

    def __repr__(self):
        return 'Iv(%r, %r)' % (self.lo, self.hi)


def _as_iv(value):
    return value if isinstance(value, Iv) else Iv(float(value), float(value))


def _contains_angle(lo, hi, target, period=TWO_PI):
    return math.ceil((lo - target) / period) <= math.floor((hi - target) / period)


def sin_iv(angle):
    lo, hi = angle.lo, angle.hi
    if not (math.isfinite(lo) and math.isfinite(hi)) or hi - lo >= TWO_PI:
        return Iv(-1.0, 1.0)
    low = min(math.sin(lo), math.sin(hi))
    high = max(math.sin(lo), math.sin(hi))
    if _contains_angle(lo, hi, HALF_PI):
        high = 1.0
    if _contains_angle(lo, hi, -HALF_PI):
        low = -1.0
    return Iv(max(-1.0, _down(low)), min(1.0, _up(high)))


def cos_iv(angle):
    lo, hi = angle.lo, angle.hi
    if not (math.isfinite(lo) and math.isfinite(hi)) or hi - lo >= TWO_PI:
        return Iv(-1.0, 1.0)
    low = min(math.cos(lo), math.cos(hi))
    high = max(math.cos(lo), math.cos(hi))
    if _contains_angle(lo, hi, 0.0):
        high = 1.0
    if _contains_angle(lo, hi, math.pi):
        low = -1.0
    return Iv(max(-1.0, _down(low)), min(1.0, _up(high)))


# --------------------------------------------------------- 参数化正交矩阵

def branch_specs(n):
    """[(det, ranges)]；ranges 为每个角度的参数区间。"""
    if n == 1:
        return [(1, []), (-1, [])]
    if n == 2:
        return [(1, [(0.0, TWO_PI)]), (-1, [(0.0, TWO_PI)])]
    if n == 3:
        return [(1, [(0.0, TWO_PI), (0.0, math.pi), (0.0, TWO_PI)]),
                (-1, [(0.0, TWO_PI), (0.0, math.pi), (0.0, TWO_PI)])]
    raise CertificateError('证书仅支持 1 至 3 个变量；更高维不做边界声明')


def q_from_angles(branch, n, angles):
    if n == 1:
        return [[1.0 if branch > 0 else -1.0]]
    if n == 2:
        c, s = math.cos(angles[0]), math.sin(angles[0])
        q = [[c, -s], [s, c]]
        if branch < 0:
            for i in range(2):
                q[i][0] = -q[i][0]
        return q
    alpha, beta, gamma = angles
    ca, sa = math.cos(alpha), math.sin(alpha)
    cb, sb = math.cos(beta), math.sin(beta)
    cc, sc = math.cos(gamma), math.sin(gamma)
    q = [
        [ca * cb * cc - sa * sc, -ca * cb * sc - sa * cc, ca * sb],
        [sa * cb * cc + ca * sc, -sa * cb * sc + ca * cc, sa * sb],
        [-sb * cc, sb * sc, cb],
    ]
    if branch < 0:
        for i in range(3):
            q[i][0] = -q[i][0]
    return q


def q_interval(branch, n, box):
    """box 为每个角度的 (lo,hi)，返回 Iv 矩阵（不含 det 符号）。"""
    if n == 1:
        return [[Iv(1.0, 1.0)]]
    if n == 2:
        c, s = cos_iv(Iv(*box[0])), sin_iv(Iv(*box[0]))
        return [[c, -s], [s, c]]
    a, b, g = (Iv(*box[0]), Iv(*box[1]), Iv(*box[2]))
    ca, sa = cos_iv(a), sin_iv(a)
    cb, sb = cos_iv(b), sin_iv(b)
    cc, sc = cos_iv(g), sin_iv(g)
    return [
        [ca * cb * cc - sa * sc, -(ca * cb * sc) - sa * cc, ca * sb],
        [sa * cb * cc + ca * sc, -(sa * cb * sc) + ca * cc, sa * sb],
        [-(sb * cc), sb * sc, cb],
    ]


def apply_branch_sign(q, branch):
    if branch < 0:
        for i in range(len(q)):
            q[i][0] = -q[i][0]
    return q


def haar_mass(branch, n, box):
    """参数盒在 O(n) Haar 概率下的质量；det=±1 两个分支各占 1/2。

    n=3 的 ZYZ 体积元为 sinβ dα dβ dγ / (8π²)，对 β 积分为
    (cos β_lo - cos β_hi)/2。
    """
    if n == 1:
        return 0.5
    if n == 2:
        return 0.5 * (box[0][1] - box[0][0]) / TWO_PI
    wa = box[0][1] - box[0][0]
    wg = box[2][1] - box[2][0]
    beta_lo, beta_hi = box[1]
    return 0.5 * (wa / TWO_PI) * (wg / TWO_PI) * (math.cos(beta_lo) - math.cos(beta_hi)) / 2.0


# -------------------------------------------------- 响应与限制的区间传播

def _theta_entries(phi_p, q_iv, entries):
    """entries 为 (h, i, j)；返回相同顺序的 Iv 值。

    theta[h][i][j] = sum_k PhiP[h][i][k] Q[k][j]。
    """
    out = []
    for h, i, j in entries:
        row = phi_p[h][i]
        total = Iv(0.0, 0.0)
        for k in range(len(row)):
            coefficient = row[k]
            if coefficient == 0.0:
                continue
            total = total + coefficient * q_iv[k][j]
        out.append(total)
    return out


def _restriction_entries(restrictions):
    return [(horizon, var, shock) for shock, var, horizon, _ in restrictions]


def classify_box(phi_p, branch, n, box, restrictions, margin):
    """返回 ('feasible'|'infeasible'|'undecided', signed_bounds)。"""
    q_iv = apply_branch_sign(q_interval(branch, n, box), branch)
    entries = _restriction_entries(restrictions)
    values = _theta_entries(phi_p, q_iv, entries)
    bounds = [(sign * value) for value, (_, _, _, sign) in zip(values, restrictions)]
    if all(value.lo >= margin for value in bounds):
        return 'feasible', bounds
    if any(value.hi < margin for value in bounds):
        return 'infeasible', bounds
    return 'undecided', bounds


# ---------------------------------------------------------------- 网格

def _uniform_grid(n, counts):
    """按每维等分计数构造网格盒。返回 [(branch, box)]，完整覆盖参数域。"""
    specs = branch_specs(n)
    boxes = []
    for branch, ranges in specs:
        if not ranges:
            boxes.append((branch, tuple()))
            continue
        divisions = []
        for (lo, hi), count in zip(ranges, counts):
            step = (hi - lo) / count
            divisions.append([(lo + k * step, lo + (k + 1) * step) for k in range(count)])
        if len(divisions) == 1:
            combos = [(d,) for d in divisions[0]]
        elif len(divisions) == 2:
            combos = [(a, b) for a in divisions[0] for b in divisions[1]]
        else:
            combos = [(a, b, c) for a in divisions[0] for b in divisions[1] for c in divisions[2]]
        for combo in combos:
            boxes.append((branch, tuple(combo)))
    return boxes


def default_grid(n):
    if n == 1:
        return []
    if n == 2:
        return (256,)
    return (24, 12, 24)


def certify(est, restrictions, margin, horizons, *, grid=None, refine_budget=20000,
            min_width=1e-4, max_boxes=None):
    """对符号限制识别集合做均匀覆盖 + 二分精化的区间证书。

    grid: 每维网格数；None 用 default_grid。refine_budget 是未决盒再细分的
    最大次数；max_boxes 兼容旧参数名，等价于 refine_budget。
    返回证书字典；response_bounds 给出内、外区间。
    """
    n = est.n
    if n not in SUPPORTED_DIMENSIONS:
        raise CertificateError('证书仅支持 1 至 3 个变量')
    if n in (1, 2):
        # 一、二维下限制对单一角度是 a sin t + b cos t，可行集是圆弧的精确
        # 并集；直接解析求解，给出精确边界而非近似。
        return _certify_low_dimension(est, restrictions, margin, horizons)
    if max_boxes is not None:
        refine_budget = max_boxes
    deepest = max((restriction[2] for restriction in restrictions), default=0)
    if deepest > horizons:
        raise CertificateError('证书期数超出响应计算范围')
    mapping = Mapping(est)
    phi = var_ma(est, horizons)
    phi_p = [la.matmul(ph, mapping.cholesky) for ph in phi]

    counts = tuple(grid) if grid is not None else default_grid(n)
    expected = {1: 0, 2: 1, 3: 3}[n]
    if len(counts) != expected:
        raise CertificateError('网格维度与变量数不符')
    if any(isinstance(c, bool) or not isinstance(c, int) or c < 1 for c in counts):
        raise CertificateError('网格每维等分数应为正整数')
    seeds = _uniform_grid(n, counts)

    feasible, infeasible, undecided = [], [], []
    processed = refined = 0
    stack = []
    for branch, box in seeds:
        processed += 1
        state, bounds = classify_box(phi_p, branch, n, box, restrictions, margin)
        if state == 'feasible':
            feasible.append(_finalize_feasible(mapping, branch, n, box, restrictions, margin, horizons))
        elif state == 'infeasible':
            infeasible.append({'branch': branch, 'box': box})
        else:
            stack.append({'branch': branch, 'box': box})

    while stack and refined < refine_budget:
        item = stack.pop()
        refined += 1
        branch, box = item['branch'], item['box']
        widest = max(range(len(box)),
                     key=lambda k: (box[k][1] - box[k][0]) / (math.pi if (n == 3 and k == 1) else TWO_PI))
        lo, hi = box[widest]
        if hi - lo <= min_width:
            state, _ = classify_box(phi_p, branch, n, box, restrictions, margin)
            if state == 'feasible':
                feasible.append(_finalize_feasible(mapping, branch, n, box, restrictions, margin, horizons))
            elif state == 'infeasible':
                infeasible.append({'branch': branch, 'box': box})
            else:
                undecided.append({'branch': branch, 'box': box})
            continue
        mid = 0.5 * (lo + hi)
        for sub in ((lo, mid), (mid, hi)):
            child = list(box)
            child[widest] = sub
            child_box = tuple(child)
            state, _ = classify_box(phi_p, branch, n, child_box, restrictions, margin)
            if state == 'feasible':
                feasible.append(_finalize_feasible(mapping, branch, n, child_box, restrictions, margin, horizons))
            elif state == 'infeasible':
                infeasible.append({'branch': branch, 'box': child_box})
            else:
                stack.append({'branch': branch, 'box': child_box})
    # 预算耗尽后仍未决的盒子全部进外近似。
    while stack:
        undecided.append(stack.pop())
        processed += 1

    mass_feasible = sum(item['haar_mass'] for item in feasible)
    mass_undecided = sum(haar_mass(item['branch'], n, item['box']) for item in undecided)
    mass_infeasible = max(0.0, 1.0 - mass_feasible - mass_undecided)
    if feasible and not undecided:
        status = 'certified_feasible_complete'
    elif feasible:
        status = 'certified_feasible_partial'
    elif not undecided:
        status = 'certified_infeasible'
    else:
        status = 'inconclusive'

    cert = {
        'method': 'interval_uniform_cover_plus_bisection',
        'dimension': n,
        'supported_dimensions': list(SUPPORTED_DIMENSIONS),
        'branches': [{'det': branch, 'angle_ranges': [[lo, hi] for lo, hi in ranges]}
                     for branch, ranges in branch_specs(n)],
        'grid': None if n == 1 else list(counts),
        'grid_box_count': len(seeds),
        'refine_budget': refine_budget,
        'refinements_used': refined,
        'min_width_radians': min_width,
        'box_accounting': {
            'seed_boxes': len(seeds),
            'feasible_boxes': len(feasible),
            'infeasible_boxes': len(infeasible),
            'undecided_boxes': len(undecided),
            'processed': processed,
            'seed_coverage_complete': True,
            'residual_undecided_mass': mass_undecided,
        },
        'haar_mass': {'feasible_inner': mass_feasible, 'undecided': mass_undecided,
                      'infeasible_excluded': mass_infeasible},
        'status': status,
        'arithmetic': 'double 精度区间运算；乘加向两侧外扩 6 个 ULP，三角界取解析极值。'
                      '这是可复核的数值证书，不是形式化证明。',
        'coverage_note': '均匀网格完整覆盖参数域；内近似为证书可行盒之并，外近似为可行盒∪未决盒。'
                         '未决测度为 0 时内外一致，才可称认证边界；未决盒仅能进外近似。',
        'restrictions': _restriction_rows(restrictions, margin, feasible),
        'response_bounds': _response_bounds(mapping, phi_p, n, horizons, feasible, undecided, restrictions),
        'feasible_boxes': [_pack_feasible(item) for item in feasible[:64]] +
                          ([{'truncated': True, 'omitted': len(feasible) - 64}] if len(feasible) > 64 else []),
        'undecided_boxes': [_pack_undecided(item, n) for item in undecided[:64]] +
                           ([{'truncated': True, 'omitted': len(undecided) - 64}] if len(undecided) > 64 else []),
        'representative_structures': _representatives(feasible),
    }
    if status == 'certified_infeasible':
        cert['infeasibility_witness'] = _infeasibility_witness(phi_p, n, infeasible, restrictions, margin)
    return cert


def _certify_low_dimension(est, restrictions, margin, horizons):
    """n=1,2 的解析证书：可行集是有限条圆弧的并，内=外=精确边界。"""
    mapping = Mapping(est)
    phi = var_ma(est, horizons)
    phi_p = [la.matmul(ph, mapping.cholesky) for ph in phi]
    if est.n == 1:
        return _certify_n1(mapping, phi_p, restrictions, margin, horizons)
    return _certify_n2(mapping, phi_p, restrictions, margin, horizons)


def _certify_n1(mapping, phi_p, restrictions, margin, horizons):
    """n=1: Q∈{+1,-1}，两点穷举即精确。"""
    feasible, infeasible = [], []
    for branch in (1, -1):
        q = [[float(branch)]]
        _, theta = mapping.response(q, horizons)
        ok, details = check_restrictions(theta, restrictions, margin)
        box = tuple()
        if ok:
            feasible.append({'branch': branch, 'box': box, 'angles': [], 'q': q,
                             'witness': details, 'haar_mass': 0.5,
                             'exact_arcs': [[0.0, TWO_PI]]})
        else:
            infeasible.append({'branch': branch, 'box': box})
    return _assemble_certificate(mapping, phi_p, 1, horizons, feasible, infeasible, [],
                                 restrictions, margin, method='analytic_exhaustive',
                                 arithmetic='n=1 时 O(1) 只有两个点，穷举即精确，无区间近似。',
                                 coverage_note='两个 det 分支各为单点，可行/不可行判定精确。')


def _certify_n2(mapping, phi_p, restrictions, margin, horizons):
    """n=2: 每条限制化为 a sin t + b cos t >= margin，可行集为圆弧并集。

    单条限制若在整圆上不可满足（下界高于振幅），该分支直接不可行；振幅
    恰好等于下界时可行集退化为零测度点，单独枚举候选点。
    """
    feasible, infeasible = [], []
    for branch in (1, -1):
        arcs = [(0.0, TWO_PI)]
        degenerate_points = []
        impossible = False
        for restriction in restrictions:
            a, b = _n2_coefficients(phi_p, restriction, branch)
            arc = _arc_for_inequality(restriction[3] * a, restriction[3] * b, margin)
            if arc is None:
                impossible = True
                break
            start, end = arc
            width = end - start
            if width >= TWO_PI - 1e-14:
                continue  # 整圆满足，对交无约束
            if width <= 1e-14:
                # 该限制只在单点（及浮点邻域）满足，把该点记为候选，
                # 并用整圆与一个极窄弧求交，保留可能的零测度解。
                degenerate_points.append((start + end) / 2.0)
                arcs = _intersect_arcs(arcs, [(start, end)])
                continue
            arcs = _intersect_arcs(arcs, [(start, end)])
            if not arcs:
                break
        if impossible:
            infeasible.append({'branch': branch, 'box': ((0.0, TWO_PI),)})
            continue

        scale = max(1.0, abs(margin))
        boundary_tol = 1e-12 * scale
        candidates = sorted({round(point % TWO_PI, 12) for point in degenerate_points})
        accepted_points = []
        for angle in candidates:
            signed = [_signed_n2(phi_p, restriction, branch, angle) for restriction in restrictions]
            if any(value < margin - boundary_tol for value in signed):
                continue
            accepted_points.append((angle, signed))

        if not arcs:
            for angle, signed in accepted_points:
                q = q_from_angles(branch, 2, [angle])
                _, theta = mapping.response(q, horizons)
                ok, details = check_restrictions(theta, restrictions, margin)
                if not ok:
                    details = _tolerance_details(restrictions, signed, margin, theta, boundary_tol)
                feasible.append({'branch': branch, 'box': ((0.0, TWO_PI),),
                                 'angles': [angle], 'q': q, 'witness': details,
                                 'haar_mass': 0.0, 'exact_arcs': [],
                                 'boundary_only': not ok,
                                 'boundary_tolerance': boundary_tol if not ok else 0.0})
            if not accepted_points:
                infeasible.append({'branch': branch, 'box': ((0.0, TWO_PI),)})
            continue

        box = ((0.0, TWO_PI),)
        for start, end in arcs:
            angle = 0.5 * (start + end) % TWO_PI
            q = q_from_angles(branch, 2, [angle])
            _, theta = mapping.response(q, horizons)
            ok, details = check_restrictions(theta, restrictions, margin)
            if not ok:
                raise CertificateError('解析弧上的见证点未通过限制检查，需复核实现')
            feasible.append({'branch': branch, 'box': box, 'angles': [angle], 'q': q,
                             'witness': details,
                             'haar_mass': 0.5 * ((end - start) % TWO_PI) / TWO_PI,
                             'exact_arcs': [[start, end]],
                             'boundary_only': False, 'boundary_tolerance': 0.0})
    return _assemble_certificate(mapping, phi_p, 2, horizons, feasible, infeasible, [],
                                 restrictions, margin, method='analytic_arc_union',
                                 arithmetic='n=2 时限制为 a sin t + b cos t，'
                                            '可行集是圆弧的精确并集，边界由反余弦解析给出。'
                                            '退化为单点时按闭集端点单独验证。',
                                 coverage_note='两条 det 分支各覆盖整圆；可行弧并集即精确识别集合。')


def _signed_n2(phi_p, restriction, branch, angle):
    a, b = _n2_coefficients(phi_p, restriction, branch)
    return restriction[3] * (a * math.sin(angle) + b * math.cos(angle))


def _tolerance_details(restrictions, signed, margin, theta, tolerance):
    """边界容差内的见证记录：明确标出是浮点端点而非严格可行。"""
    rows = []
    for restriction, value in zip(restrictions, signed):
        rows.append({'shock': restriction[0], 'variable_index': restriction[1],
                     'horizon': restriction[2], 'sign': '+' if restriction[3] > 0 else '-',
                     'signed_response': value, 'margin': margin,
                     'satisfied': value >= margin,
                     'within_boundary_tolerance': True,
                     'boundary_tolerance': tolerance})
    return rows


def _n2_coefficients(phi_p, restriction, branch):
    """响应 = a sin t + b cos t。Q=[[bs*c,-s],[bs*s,c]]，bs 为分支符号。"""
    shock, var, horizon, _ = restriction
    p0 = phi_p[horizon][var][0]
    p1 = phi_p[horizon][var][1]
    if shock == 0:
        return branch * p1, branch * p0
    return -p0, p1


def _arc_for_inequality(a, b, lower):
    """{t: a sin t + b cos t >= lower} 写成圆心角与宽度；不可能时返回 None。"""
    r = math.hypot(a, b)
    if r == 0.0:
        return (0.0, TWO_PI) if lower <= 0.0 else None
    if lower > r * (1.0 + 1e-15):
        return None
    if lower <= -r * (1.0 + 1e-15):
        return (0.0, TWO_PI)
    ratio = max(-1.0, min(1.0, lower / r))
    half = math.acos(ratio)
    center = math.atan2(a, b)
    return (center - half, center + half)


def _merge_intervals(intervals, period=TWO_PI):
    """合并环面上的区间；输入 (start, end) 可为任意实数端点。"""
    if not intervals:
        return []
    pieces = []
    for start, end in intervals:
        width = end - start
        if width <= 0.0:
            continue
        width = min(width, period)
        base = start % period
        if base + width <= period:
            pieces.append((base, base + width))
        else:
            pieces.append((base, period))
            pieces.append((0.0, base + width - period))
    pieces.sort()
    merged = []
    for start, end in pieces:
        if merged and start <= merged[-1][1] + 1e-13:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    if len(merged) > 1 and merged[0][0] <= 1e-13 and merged[-1][1] >= period - 1e-13:
        merged[0] = (merged[-1][0] - period, merged[0][1])
        merged.pop()
    return merged


def _intersect_arcs(first, second, period=TWO_PI):
    merged_first = _merge_intervals(first, period)
    merged_second = _merge_intervals(second, period)
    out = []
    for s1, e1 in merged_first:
        for s2, e2 in merged_second:
            for shift in (-period, 0.0, period):
                lo = max(s1, s2 + shift)
                hi = min(e1, e2 + shift)
                if hi > lo + 1e-15:
                    out.append((lo, hi))
    return _merge_intervals(out, period)


def _sinusoid_extreme(a, b, start, end):
    """f(t)=a sin t + b cos t 在闭区间上的最小、最大值（解析）。"""
    r = math.hypot(a, b)
    phi0 = math.atan2(b, a)  # f(t) = r sin(t + phi0)
    candidates = [math.sin(start + phi0), math.sin(end + phi0)]
    # 极值点 t = pi/2 - phi0 + k*pi
    base = math.pi / 2.0 - phi0
    kmin = math.ceil((start - base) / math.pi)
    kmax = math.floor((end - base) / math.pi)
    for k in range(kmin, kmax + 1):
        candidates.append(math.sin(base + k * math.pi + phi0))
    return r * min(candidates), r * max(candidates)


def _finalize_feasible(mapping, branch, n, box, restrictions, margin, horizons):
    angles = [0.5 * (lo + hi) for lo, hi in box]
    q = q_from_angles(branch, n, angles)
    _, theta = mapping.response(q, horizons)
    ok, details = check_restrictions(theta, restrictions, margin)
    if not ok:
        for corner in _corners(box):
            q = q_from_angles(branch, n, corner)
            _, theta = mapping.response(q, horizons)
            ok, details = check_restrictions(theta, restrictions, margin)
            if ok:
                angles = corner
                break
        else:
            raise CertificateError('见证点未通过限制检查：区间算术与精确计算不一致')
    return {'branch': branch, 'box': box, 'angles': angles, 'q': q,
            'witness': details, 'haar_mass': haar_mass(branch, n, box)}


def _assemble_certificate(mapping, phi_p, n, horizons, feasible, infeasible, undecided,
                          restrictions, margin, *, method, arithmetic, coverage_note):
    """统一组装证书字典；n=1,2 时 arcs 已精确，内=外。"""
    if n == 1:
        mass_feasible = sum(item['haar_mass'] for item in feasible)
    else:
        mass_feasible = sum(item['haar_mass'] for item in feasible)
    mass_undecided = 0.0
    if feasible:
        status = 'certified_feasible_complete'
    elif infeasible:
        status = 'certified_infeasible'
    else:
        status = 'inconclusive'
    bounds = _analytic_bounds(mapping, phi_p, n, horizons, feasible, restrictions)
    cert = {
        'method': method,
        'dimension': n,
        'supported_dimensions': list(SUPPORTED_DIMENSIONS),
        'branches': [{'det': branch, 'angle_ranges': [[lo, hi] for lo, hi in ranges]}
                     for branch, ranges in branch_specs(n)],
        'box_accounting': {'seed_boxes': len(branch_specs(n)),
                           'feasible_boxes': len(feasible),
                           'infeasible_boxes': len(infeasible),
                           'undecided_boxes': 0,
                           'processed': len(feasible) + len(infeasible),
                           'seed_coverage_complete': True,
                           'residual_undecided_mass': 0.0},
        'haar_mass': {'feasible_inner': mass_feasible, 'undecided': mass_undecided,
                      'infeasible_excluded': max(0.0, 1.0 - mass_feasible)},
        'status': status,
        'arithmetic': arithmetic,
        'coverage_note': coverage_note,
        'restrictions': _restriction_rows(restrictions, margin, feasible),
        'response_bounds': bounds,
        'feasible_boxes': [_pack_analytic(item) for item in feasible],
        'undecided_boxes': [],
        'representative_structures': _representatives(feasible),
    }
    if status == 'certified_infeasible':
        cert['infeasibility_witness'] = {
            'explanation': '每个 det 分支的可行弧集合为空；分支覆盖整个 O(n)。',
            'margin': margin,
            'branches_checked': [1, -1],
            'strictly_below_margin': True,
            'box_count': len(infeasible),
        }
    return cert


def _pack_analytic(item):
    return {'branch': item['branch'], 'angle_arcs': item.get('exact_arcs'),
            'witness_angles': item['angles'], 'witness_q': item['q'],
            'witness_signed_responses': item['witness'], 'haar_mass': item['haar_mass']}


def _analytic_bounds(mapping, phi_p, n, horizons, feasible, restrictions):
    """精确响应边界：在每段可行弧上解析求正弦极值。"""
    profiles = {}
    for shock, var, horizon, sign_ in restrictions:
        profiles.setdefault(shock, []).append((var, horizon, sign_))
    unrestricted = [j for j in range(n) if j not in profiles]
    rows = []
    for j in range(n):
        if j in unrestricted:
            for i in range(n):
                for h in range(horizons + 1):
                    rows.append({'shock': j, 'variable_index': i, 'horizon': h,
                                 'identified_by_certificate': False, 'kind': 'not_identified',
                                 'inner_lo': None, 'inner_hi': None,
                                 'outer_lo': None, 'outer_hi': None,
                                 'note': '该冲击无符号限制，列在正交补内可任意正交变换，'
                                         '不构成可报告的逐期集合区间'})
            continue
        for i in range(n):
            for h in range(horizons + 1):
                lows, highs, boundary = [], [], False
                for item in feasible:
                    if item.get('boundary_only'):
                        boundary = True
                    for start, end in item['exact_arcs']:
                        if n == 1:
                            value = phi_p[h][i][0] * item['q'][0][0]
                            lows.append(value)
                            highs.append(value)
                            continue
                        a, b = _n2_coefficients(phi_p, (j, i, h, 1), item['branch'])
                        low, high = _sinusoid_extreme(a, b, start, end)
                        lows.append(low)
                        highs.append(high)
                    if not item['exact_arcs'] and item.get('angles'):
                        # 零测度可行点：直接在该点取值。
                        a, b = _n2_coefficients(phi_p, (j, i, h, 1), item['branch'])
                        angle = item['angles'][0]
                        value = a * math.sin(angle) + b * math.cos(angle)
                        lows.append(value)
                        highs.append(value)
                if not lows:
                    continue
                rows.append({'shock': j, 'variable_index': i, 'horizon': h,
                             'identified_by_certificate': True,
                             'inner_lo': min(lows), 'inner_hi': max(highs),
                             'outer_lo': min(lows), 'outer_hi': max(highs),
                             'outer_lo_is_infinite': False, 'outer_hi_is_infinite': False,
                             'gap_lo': 0.0, 'gap_hi': 0.0,
                             'kind': 'certified_exact_bounds',
                             'measure_zero_feasible_set': boundary,
                             'note': ('解析弧上的精确极值；内近似与外近似重合'
                                      + ('；可行集为零测度点，极值取自该点' if boundary else ''))})
    return rows


def _corners(box):
    """盒角点，按到中心的距离排序；角点可能落在开区间的另一侧。"""
    if not box:
        return [()]
    out = []
    for bits in range(2 ** len(box)):
        out.append(tuple((lo if (bits >> k) & 1 else hi) for k, (lo, hi) in enumerate(box)))
    return out


def _infeasibility_witness(phi_p, n, infeasible, restrictions, margin):
    """每盒上“最接近满足”的限制的上界；全部严格低于 margin 才是不可行证书。"""
    overall = None
    per_box = []
    for item in infeasible:
        q_iv = apply_branch_sign(q_interval(item['branch'], n, item['box']), item['branch'])
        values = _theta_entries(phi_p, q_iv, _restriction_entries(restrictions))
        uppers = [(sign * value).hi for value, (_, _, _, sign) in zip(values, restrictions)]
        tightest = max(uppers)
        overall = tightest if overall is None else max(overall, tightest)
        per_box.append({'branch': item['branch'],
                        'angle_ranges': [[lo, hi] for lo, hi in item['box']],
                        'tightest_upper_bound': tightest})
    return {
        'explanation': '参数域被有限个盒完整覆盖；每个盒的区间上界证明至少一条限制严格不可满足。',
        'max_signed_upper_bound_over_all_boxes': overall,
        'margin': margin,
        'strictly_below_margin': overall is not None and overall < margin,
        'box_count': len(infeasible),
        'boxes': per_box[:64],
    }


def _restriction_rows(restrictions, margin, feasible):
    rows = []
    for index, restriction in enumerate(restrictions):
        shock, var, horizon, sign = restriction
        values = [item['witness'][index]['signed_response'] for item in feasible]
        rows.append({'shock': shock, 'variable_index': var, 'horizon': horizon,
                     'sign': '+' if sign > 0 else '-', 'margin': margin,
                     'witness_signed_response_min': min(values) if values else None,
                     'witness_signed_response_max': max(values) if values else None})
    return rows


def _pack_feasible(item):
    return {'branch': item['branch'], 'angle_ranges': [[lo, hi] for lo, hi in item['box']],
            'witness_angles': item['angles'], 'witness_q': item['q'],
            'witness_signed_responses': item['witness'], 'haar_mass': item['haar_mass']}


def _pack_undecided(item, n):
    return {'branch': item['branch'], 'angle_ranges': [[lo, hi] for lo, hi in item['box']],
            'haar_mass': haar_mass(item['branch'], n, item['box'])}


def _representatives(feasible, limit=8):
    """按 (分支, 角度) 排序的见证结构；用于历史分解的单一可行结构。"""
    items = sorted(feasible, key=lambda x: (x['branch'], x['angles']))
    return [{'branch': item['branch'], 'angles': item['angles'], 'q': item['q'],
             'signed_responses': item['witness']} for item in items[:limit]]


def _response_bounds(mapping, phi_p, n, horizons, feasible, undecided, restrictions):
    """内、外响应区间；无限制冲击标注为不可识别。"""
    profiles = {}
    for shock, var, horizon, sign in restrictions:
        profiles.setdefault(shock, []).append((var, horizon, sign))
    unrestricted = [j for j in range(n) if j not in profiles]

    def accumulate(items):
        store = [[[None for _ in range(n)] for _ in range(n)] for _ in range(horizons + 1)]
        for item in items:
            if 'box' not in item:
                continue
            q_iv = apply_branch_sign(q_interval(item['branch'], n, item['box']), item['branch'])
            entries = [(h, i, j) for h in range(horizons + 1) for i in range(n) for j in range(n)]
            values = _theta_entries(phi_p, q_iv, entries)
            for (h, i, j), value in zip(entries, values):
                cell = store[h][i][j]
                if cell is None:
                    store[h][i][j] = [value.lo, value.hi]
                else:
                    cell[0] = min(cell[0], value.lo)
                    cell[1] = max(cell[1], value.hi)
        return store

    inner = accumulate(feasible)
    outer = accumulate(feasible + undecided)

    rows = []
    for j in range(n):
        if j in unrestricted:
            for i in range(n):
                for h in range(horizons + 1):
                    rows.append({'shock': j, 'variable_index': i, 'horizon': h,
                                 'identified_by_certificate': False, 'kind': 'not_identified',
                                 'inner_lo': None, 'inner_hi': None,
                                 'outer_lo': None, 'outer_hi': None,
                                 'note': '该冲击无符号限制，列在正交补内可任意正交变换，'
                                         '不构成可报告的逐期集合区间'})
            continue
        for i in range(n):
            for h in range(horizons + 1):
                out = outer[h][i][j]
                if out is None:
                    continue
                inn = inner[h][i][j]
                gap_lo = inn[0] - out[0] if inn is not None else None
                gap_hi = out[1] - inn[1] if inn is not None else None
                rows.append({
                    'shock': j, 'variable_index': i, 'horizon': h,
                    'identified_by_certificate': True,
                    'inner_lo': inn[0] if inn is not None else None,
                    'inner_hi': inn[1] if inn is not None else None,
                    'outer_lo': out[0], 'outer_hi': out[1],
                    'outer_lo_is_infinite': not math.isfinite(out[0]),
                    'outer_hi_is_infinite': not math.isfinite(out[1]),
                    'gap_lo': gap_lo, 'gap_hi': gap_hi,
                    'kind': 'certified_sandwich' if (gap_lo or gap_hi) else 'certified_exact_bounds',
                    'note': '内近似⊆真实识别集合⊆外近似',
                })
    return rows
