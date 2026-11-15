"""纯标准库线性代数，供约化式与结构分析使用。

矩阵以行优先嵌套列表表示。特征值例程用于稳定性与协方差诊断，
均为 double 精度，不宣称符号或边界的机器精度证明。
"""
import math


class LinAlgError(ValueError):
    """线性代数失败：奇异矩阵、非正定协方差或特征值未收敛。"""


def zeros(n, m):
    return [[0.0] * m for _ in range(n)]


def eye(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def transpose(a):
    return [list(row) for row in zip(*a)]


def matmul(a, b):
    n, k, m = len(a), len(b), len(b[0])
    if len(a[0]) != k:
        raise LinAlgError('矩阵维度不匹配')
    out = zeros(n, m)
    for i in range(n):
        ai = a[i]
        oi = out[i]
        for t in range(k):
            x = ai[t]
            if x == 0.0:
                continue
            bt = b[t]
            for j in range(m):
                oi[j] += x * bt[j]
    return out


def matvec(a, x):
    return [sum(aij * xj for aij, xj in zip(row, x)) for row in a]


def scale(a, s):
    return [[v * s for v in row] for row in a]


def add(a, b):
    return [[x + y for x, y in zip(ra, rb)] for ra, rb in zip(a, b)]


def sub(a, b):
    return [[x - y for x, y in zip(ra, rb)] for ra, rb in zip(a, b)]


def norm(a):
    return math.sqrt(sum(v * v for row in a for v in row))


def cholesky(a):
    """下三角 Cholesky，要求对称正定。"""
    n = len(a)
    low = zeros(n, n)
    for i in range(n):
        for j in range(i + 1):
            s = a[i][j] - sum(low[i][k] * low[j][k] for k in range(j))
            if i == j:
                if not s > 0.0 or not math.isfinite(s):
                    raise LinAlgError('协方差矩阵非正定')
                low[i][j] = math.sqrt(s)
            else:
                low[i][j] = s / low[j][j]
    return low


def inverse(a):
    """Gauss-Jordan 带部分主元求逆。"""
    n = len(a)
    aug = [list(a[i]) + [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            raise LinAlgError('矩阵奇异')
        aug[col], aug[pivot] = aug[pivot], aug[col]
        p = aug[col][col]
        aug[col] = [v / p for v in aug[col]]
        for r in range(n):
            if r == col:
                continue
            f = aug[r][col]
            if f:
                aug[r] = [x - f * y for x, y in zip(aug[r], aug[col])]
    return [row[n:] for row in aug]


def _jacobi(a, max_iter=100):
    """对称矩阵 Jacobi 特征分解，返回 (特征值, 特征向量列)。"""
    n = len(a)
    m = [list(row) for row in a]
    v = eye(n)
    for _ in range(max_iter):
        diag_scale = sum(m[i][i] ** 2 for i in range(n))
        off = sum(m[i][j] ** 2 for i in range(n) for j in range(n) if i != j)
        if off <= 1e-26 * max(diag_scale, 1e-300) or off == 0.0:
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                if abs(m[p][q]) < 1e-300:
                    continue
                theta = (m[q][q] - m[p][p]) / (2.0 * m[p][q])
                t = math.copysign(1.0, theta) / (abs(theta) + math.sqrt(theta * theta + 1.0))
                c = 1.0 / math.sqrt(t * t + 1.0)
                s = t * c
                for k in range(n):
                    mkp, mkq = m[k][p], m[k][q]
                    m[k][p] = c * mkp - s * mkq
                    m[k][q] = s * mkp + c * mkq
                for k in range(n):
                    mpk, mqk = m[p][k], m[q][k]
                    m[p][k] = c * mpk - s * mqk
                    m[q][k] = s * mpk + c * mqk
                for k in range(n):
                    vkp, vkq = v[k][p], v[k][q]
                    v[k][p] = c * vkp - s * vkq
                    v[k][q] = s * vkp + c * vkq
    else:
        raise LinAlgError('特征值未收敛')
    return [m[i][i] for i in range(n)], v


def sym_eig(a):
    vals, vecs = _jacobi(a)
    order = sorted(range(len(vals)), key=lambda i: vals[i], reverse=True)
    return [vals[i] for i in order], [[vecs[r][i] for i in order] for r in range(len(a))]


def charpoly(a):
    """Faddeev-LeVerrier：返回单调递减到常数项的特征多项式系数 [1, c1, ..., cn]。"""
    n = len(a)
    m = eye(n)
    coeffs = [1.0]
    for k in range(1, n + 1):
        am = matmul(a, m)
        c = -sum(am[i][i] for i in range(n)) / k
        coeffs.append(c)
        m = [[am[i][j] + (c if i == j else 0.0) for j in range(n)] for i in range(n)]
    return coeffs


def poly_roots(coeffs, max_iter=500):
    """Durand-Kerner 求复根，返回复根列表。coeffs 从最高次到常数项。"""
    lead = coeffs[0]
    if lead == 0.0:
        raise LinAlgError('多项式首项为零')
    c = [x / lead for x in coeffs]
    while len(c) > 1 and abs(c[0]) < 1e-300:
        c.pop(0)
    deg = len(c) - 1
    if deg <= 0:
        return []
    radius = 1.0 + max(abs(x) for x in c[1:])
    roots = [complex(radius) * complex(math.cos(2 * math.pi * i / deg), math.sin(2 * math.pi * i / deg))
             for i in range(deg)]
    for _ in range(max_iter):
        shift = 0.0
        for i in range(deg):
            z = roots[i]
            p = c[0]
            for k in range(1, deg + 1):
                p = p * z + c[k]
            d = complex(1.0, 0.0)
            for j in range(deg):
                if j != i:
                    d *= (z - roots[j])
            if abs(d) < 1e-300:
                d = 1e-300
            delta = p / d
            roots[i] = z - delta
            shift = max(shift, abs(delta))
        if shift < 1e-14 * max(1.0, radius):
            break
    else:
        raise LinAlgError('特征值未收敛')
    return roots


def max_abs_eig(a):
    """最大模特征值，用于伴随矩阵稳定性。"""
    if not a:
        return 0.0
    return max(abs(r) for r in poly_roots(charpoly(a)))
