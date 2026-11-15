"""小样本下 n=1 与 n=2 的精确穷举参照，用于检验搜索与证书。

n=1: Q=±1，穷举两点。
n=2: Q 由单角 a 参数化，两分支 a∈[0,2π)。在给定角度分辨率下网格穷举，
     用很小的网格与证书结果对照，验证“证书外包络确实包住穷举结果”。
"""
import math


def exhaustive_n1(est, restrictions, margin, horizons):
    from app.sign import Mapping, check_restrictions
    mapping = Mapping(est)
    feasible, all_points = [], []
    for branch in (1, -1):
        q = [[1.0 if branch > 0 else -1.0]]
        _, theta = mapping.response(q, horizons)
        ok, details = check_restrictions(theta, restrictions, margin)
        all_points.append({'branch': branch, 'q': q, 'theta': theta, 'accepted': ok})
        if ok:
            feasible.append({'branch': branch, 'q': q, 'angles': [], 'theta': theta,
                             'witness': details})
    return {'feasible': feasible, 'points': all_points}


def exhaustive_n2(est, restrictions, margin, horizons, steps=4096):
    from app.sign import Mapping, check_restrictions
    mapping = Mapping(est)
    feasible = []
    for branch in (1, -1):
        for k in range(steps):
            angle = 2.0 * math.pi * k / steps
            c, s = math.cos(angle), math.sin(angle)
            q = [[c, -s], [s, c]]
            if branch < 0:
                for i in range(2):
                    q[i][0] = -q[i][0]
            _, theta = mapping.response(q, horizons)
            ok, details = check_restrictions(theta, restrictions, margin)
            if ok:
                feasible.append({'branch': branch, 'angle': angle, 'q': q, 'theta': theta,
                                 'witness': details})
    return {'feasible': feasible, 'steps': steps}


def response_envelope(points, n, horizons, key='theta'):
    """穷举点的响应外包络：内近似应被它包含，外近似应包含它。"""
    envelope = [[[None, None] for _ in range(n)] for _ in range(horizons + 1)]
    for point in points:
        theta = point[key]
        for h in range(horizons + 1):
            for i in range(n):
                for j in range(n):
                    value = theta[h][i][j]
                    cell = envelope[h][i][j]
                    if cell[0] is None:
                        envelope[h][i][j] = [value, value]
                    else:
                        cell[0] = min(cell[0], value)
                        cell[1] = max(cell[1], value)
    return envelope
