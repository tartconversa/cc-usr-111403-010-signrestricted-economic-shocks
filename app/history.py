"""历史分解：在同一可行结构下拆分初始条件、确定项与各结构冲击。

对保留样本的每一期 t（t=0 为保留样本第一期）：
    y_t = 初始条件_t + 确定项_t + sum_j 冲击j贡献_t
初始条件_t = top_n(A^{t+1} S_{-1})，S_{-1} = [y_{-1}; y_{-2}; ...; y_{-p}]；
确定项_t = top_n(sum_{k=0}^{t} A^k d)，d = [c; 0]；
冲击j贡献_t = sum_{s=0}^{t} (Phi_{t-s} B e_j) eps_{s,j}，eps_s = B^{-1} u_s。

B = P Q 必须来自与当期分析同一个可行 Q，不能把不同旋转的各列拼进来。
重构误差为代数恒等式的数值检验，逐期逐变量输出。
"""
from . import linalg as la
from .sign import Mapping, var_ma


def decompose(est, q, variables):
    n, p, t_obs = est.n, est.lags, est.t_obs
    mapping = Mapping(est)
    b, _ = mapping.response(q, 0)
    phi = var_ma(est, max(t_obs - 1, 0))
    try:
        b_inv = la.inverse(b)
    except la.LinAlgError as exc:
        raise ValueError('结构矩阵不可逆，历史分解无定义') from exc

    initial_state = []
    first_x = est.x[0]
    for lag in range(1, p + 1):
        initial_state.extend(first_x[1 + (lag - 1) * n: 1 + lag * n])
    if len(initial_state) != n * p:
        raise ValueError('初始状态维度与滞后阶数不符')

    deterministic_seed = list(est.intercept) + [0.0] * (n * (p - 1))
    initial_power = [float(v) for v in initial_state]
    deterministic_power = list(deterministic_seed)
    deterministic_total = [0.0] * (n * p)

    structural_shocks, reduced_residuals = [], []
    initial_rows, det_rows, residual_rows = [], [], []
    shock_rows = [[] for _ in range(n)]
    reconstructed, actual, errors, identity_gap = [], [], [], []

    for t in range(t_obs):
        initial_power = la.matvec(est.companion, initial_power)
        deterministic_total = [a + b for a, b in zip(deterministic_total, deterministic_power)]
        deterministic_power = la.matvec(est.companion, deterministic_power)
        init_now = initial_power[:n]
        det_now = deterministic_total[:n]

        eps = la.matvec(b_inv, est.residuals[t])
        structural_shocks.append(eps)
        reduced_residuals.append(list(est.residuals[t]))

        per_shock = [[0.0] * n for _ in range(n)]
        for j in range(n):
            for s in range(t + 1):
                weight = structural_shocks[s][j]
                if weight == 0.0:
                    continue
                lagged = phi[t - s]
                for i in range(n):
                    response = sum(lagged[i][k] * b[k][j] for k in range(n))
                    per_shock[j][i] += response * weight

        total_shock = [sum(per_shock[j][i] for j in range(n)) for i in range(n)]
        recon = [init_now[i] + det_now[i] + total_shock[i] for i in range(n)]
        initial_rows.append(init_now)
        det_rows.append(det_now)
        for j in range(n):
            shock_rows[j].append(per_shock[j])
        # 当期残差项显式保留：B eps_t = u_t，是冲击贡献中 h=0 的部分。
        residual_rows.append([sum(b[i][j] * eps[j] for j in range(n)) for i in range(n)])
        reconstructed.append(recon)
        actual.append(list(est.y[t]))
        residual_row = [actual[-1][i] - recon[i] for i in range(n)]
        errors.append(residual_row)
        # 结构冲击重构：sum_j B[:,j] eps_sj 应等于 u_s。
        rebuilt_u = [sum(b[i][j] * eps[j] for j in range(n)) for i in range(n)]
        identity_gap.append([rebuilt_u[i] - reduced_residuals[-1][i] for i in range(n)])

    max_error = max((abs(v) for row in errors for v in row), default=0.0)
    max_identity_gap = max((abs(v) for row in identity_gap for v in row), default=0.0)

    return {
        'months': list(est.months),
        'variables': list(variables),
        'structure': {'q': q, 'b': b, 'sigma': est.sigma},
        'initial_state': initial_state,
        'initial_condition': initial_rows,
        'deterministic': det_rows,
        'residual_term': residual_rows,
        'shocks': [{'shock': j, 'contributions': shock_rows[j]} for j in range(n)],
        'actual': actual,
        'reconstructed': reconstructed,
        'reconstruction_error': errors,
        'structural_shocks': structural_shocks,
        'reduced_residuals': reduced_residuals,
        'reduced_residual_identity_gap': identity_gap,
        'checks': {
            'max_abs_reconstruction_error': max_error,
            'max_abs_reduced_residual_identity_gap': max_identity_gap,
            'max_abs_residual_term_vs_reduced': max(
                (abs(residual_rows[t][i] - reduced_residuals[t][i])
                 for t in range(t_obs) for i in range(n)), default=0.0),
            'initial_condition_included': True,
            'residual_term_included': True,
            'residual_representation': '当期残差以结构冲击 B*eps_t=u_t 计入 h=0 的冲击贡献；'
                                       'residual_term 字段单独列出该部分，reconstruction 不重复相加',
            'same_structure_for_all_shocks': True,
        },
    }
