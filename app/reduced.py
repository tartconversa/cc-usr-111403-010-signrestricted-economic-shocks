"""约化式 VAR 估计、稳定性与残差协方差诊断。

估计使用与 app.baseline 相同的滞后窗口定义，但额外要求保留样本的
月份严格连续，否则结构递归与历史分解没有定义。仅做水平值估计，
不对变量做差分或标准化——契约没有规定变换，程序不擅自加入。
"""
import hashlib
import json
import math
import random
from dataclasses import dataclass

from . import linalg as la
from .baseline import analyze


class EstimationError(ValueError):
    """约化式资料不足以进行稳定估计。"""


@dataclass(frozen=True)
class ReducedVAR:
    variables: list
    lags: int
    months: list
    y: list
    x: list
    beta: list
    intercept: list
    residuals: list
    sigma: list
    sigma_df: int
    t_obs: int
    k_regressors: int
    companion: list
    max_eig: float
    regressor_gram_eigenvalues: list
    condition_number: float
    diagnostics: dict
    warnings: list
    sample_digest: str

    @property
    def n(self):
        return len(self.variables)


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def build_sample(snapshot):
    """返回 (variables, lags, months, Y, X, pre_sample_y, pre_sample_month)。

    前 p 期观测从首行 X 中恢复，用于递归自举与历史分解初始条件。
    """
    metadata = snapshot.metadata
    variables = metadata.get('variables')
    lags = metadata.get('lags')
    if not isinstance(variables, list) or not variables or any(not isinstance(v, str) or not v for v in variables):
        raise EstimationError('metadata.variables应为非空变量键数组')
    if len(set(variables)) != len(variables):
        raise EstimationError('metadata.variables含重复变量')
    if not isinstance(lags, int) or isinstance(lags, bool) or lags < 1:
        raise EstimationError('metadata.lags应为正整数')
    rows = analyze(snapshot)
    if not rows:
        raise EstimationError('没有满足完整性要求的滞后窗口')
    months = [r['month'] for r in rows]
    for m in months:
        if not isinstance(m, int) or isinstance(m, bool):
            raise EstimationError('结构分析要求月份为整数')
    if any(months[i + 1] != months[i] + 1 for i in range(len(months) - 1)):
        raise EstimationError('保留样本月份不连续，无法定义VAR递归')
    n = len(variables)
    if len(rows[0]['X']) != 1 + lags * n:
        raise EstimationError('滞后窗口维度与variables不符')
    y = [r['Y'] for r in rows]
    x = [r['X'] for r in rows]
    first_x = rows[0]['X']
    pre = []
    for lag in range(1, lags + 1):
        pre.append([first_x[1 + (lag - 1) * n + j] for j in range(n)])
    digest = hashlib.sha256(_canonical({'lags': lags, 'variables': variables, 'months': months,
                                        'y': y, 'x': x, 'pre': pre}).encode('utf-8')).hexdigest()
    return variables, lags, months, y, x, pre, digest


def _ols(y, x):
    t, k = len(x), len(x[0])
    n = len(y[0])
    if t <= k:
        raise EstimationError('观测数不大于回归元数，无法估计残差协方差')
    xtx = la.zeros(k, k)
    xty = la.zeros(k, n)
    for row, yrow in zip(x, y):
        for i in range(k):
            ri = row[i]
            for j in range(i, k):
                xtx[i][j] += ri * row[j]
            for j in range(n):
                xty[i][j] += ri * yrow[j]
    for i in range(k):
        for j in range(i):
            xtx[i][j] = xtx[j][i]
    diag = [math.sqrt(xtx[i][i]) for i in range(k)]
    if any(d <= 0.0 or not math.isfinite(d) for d in diag):
        raise EstimationError('设计矩阵列退化')
    scaled = [[xtx[i][j] / (diag[i] * diag[j]) for j in range(k)] for i in range(k)]
    vals, _ = la.sym_eig(scaled)
    cond = vals[0] / vals[-1] if vals[-1] > 0 else float('inf')
    if not math.isfinite(cond) or vals[-1] <= 1e-12 * vals[0]:
        raise EstimationError('设计矩阵近似共线或秩亏')
    inv = la.inverse(scaled)
    # 回归元已按 D=diag(diag) 缩放：beta = D^{-1} inv(D) D^{-1} X'y。
    beta = [[sum(inv[i][q] * xty[q][j] / diag[q] for q in range(k)) / diag[i] for j in range(n)]
            for i in range(k)]
    fitted = [[sum(row[i] * beta[i][j] for i in range(k)) for j in range(n)] for row in x]
    resid = [[yrow[j] - frow[j] for j in range(n)] for yrow, frow in zip(y, fitted)]
    return beta, resid, cond


def _ljung_box(resid, max_lag):
    t, n = len(resid), len(resid[0])
    out = []
    h = max(1, min(max_lag, t // 4))
    for j in range(n):
        col = [r[j] for r in resid]
        denom = sum(v * v for v in col)
        if denom <= 0.0:
            out.append({'variable_index': j, 'statistic': None, 'df': h, 'p_value': None,
                        'note': '残差方差为零，无法诊断'})
            continue
        q = 0.0
        for lag in range(1, h + 1):
            num = sum(col[i] * col[i - lag] for i in range(lag, t))
            rho = num / denom
            q += rho * rho / (t - lag)
        q *= t * (t + 2)
        p = _chi2_sf(q, h)
        out.append({'variable_index': j, 'lag_order': h, 'statistic': q, 'df': h, 'p_value': p})
    return out


def _jarque_bera(resid):
    t, n = len(resid), len(resid[0])
    out = []
    for j in range(n):
        col = [r[j] for r in resid]
        mean = sum(col) / t
        m2 = sum((v - mean) ** 2 for v in col) / t
        if m2 <= 0.0:
            out.append({'variable_index': j, 'statistic': None, 'p_value': None,
                        'note': '残差方差为零，无法诊断'})
            continue
        m3 = sum((v - mean) ** 3 for v in col) / t
        m4 = sum((v - mean) ** 4 for v in col) / t
        skew = m3 / m2 ** 1.5
        kurt = m4 / m2 ** 2
        stat = t / 6.0 * (skew ** 2 + (kurt - 3.0) ** 2 / 4.0)
        out.append({'variable_index': j, 'statistic': stat, 'df': 2,
                    'p_value': math.exp(-stat / 2.0), 'skewness': skew, 'kurtosis': kurt})
    return out


def _chi2_sf(x, df):
    """Wilson-Hilferty 正态近似；不宣称精确卡方尾概率。"""
    if df <= 0:
        return None
    if x <= 0.0:
        return 1.0
    z = ((x / df) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * df))) / math.sqrt(2.0 / (9.0 * df))
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def _companion(beta, n, p):
    np = n * p
    comp = la.zeros(np, np)
    for lag in range(1, p + 1):
        for i in range(n):
            for j in range(n):
                comp[i][n * (lag - 1) + j] = beta[1 + n * (lag - 1) + j][i]
    for lag in range(2, p + 1):
        for i in range(n):
            comp[n * (lag - 1) + i][n * (lag - 2) + i] = 1.0
    return comp


def estimate_from_series(variables, lags, months, y, x, pre, sample_digest):
    n = len(variables)
    t, k = len(x), len(x[0])
    if t <= k:
        raise EstimationError('观测数不大于回归元数，无法估计残差协方差')
    beta, resid, cond = _ols(y, x)
    sigma = la.zeros(n, n)
    df = t - k
    for urow in resid:
        for i in range(n):
            for j in range(i, n):
                sigma[i][j] += urow[i] * urow[j]
    for i in range(n):
        for j in range(i, n):
            sigma[i][j] /= df
            sigma[j][i] = sigma[i][j]
    try:
        la.cholesky(sigma)
        sigma_pd = True
    except la.LinAlgError:
        sigma_pd = False
    if not sigma_pd:
        raise EstimationError('残差协方差非正定，结构分解无定义')
    vals, _ = la.sym_eig(sigma)
    if vals[-1] <= 0.0:
        raise EstimationError('残差协方差特征值为非正')
    sigma_cond = vals[0] / vals[-1]
    companion = _companion(beta, n, lags)
    max_eig = la.max_abs_eig(companion)
    warnings = []
    if max_eig >= 1.0:
        warnings.append('伴随矩阵最大模特征值不小于1：VAR不稳定')
    elif max_eig > 0.97:
        warnings.append('伴随矩阵最大模特征值接近1：存在近单位根')
    if cond > 1e3:
        warnings.append('设计矩阵条件数偏高')
    joint_lb = _ljung_box(resid, 12)
    jb = _jarque_bera(resid)
    if any(r.get('p_value') is not None and r['p_value'] < 0.05 for r in joint_lb):
        warnings.append('残差存在显著自相关（Ljung-Box，正态近似）')
    if any(r.get('p_value') is not None and r['p_value'] < 0.05 for r in jb):
        warnings.append('残差正态性诊断拒绝（Jarque-Bera）')
    diagnostic_lag = max(1, min(12, t // 4))
    diagnostics = {
        'residual_covariance': sigma,
        'covariance_positive_definite': True,
        'covariance_eigenvalues': vals,
        'covariance_condition_number': sigma_cond,
        'residual_std': [math.sqrt(sigma[i][i]) for i in range(n)],
        'ljung_box': joint_lb,
        'ljung_box_lag_order': diagnostic_lag,
        'jarque_bera': jb,
        'note': 'Ljung-Box与Jarque-Bera尾概率用正态近似，仅作诊断提示',
    }
    return ReducedVAR(variables=list(variables), lags=lags, months=list(months), y=[list(r) for r in y],
                      x=[list(r) for r in x], beta=beta, intercept=[beta[0][j] for j in range(n)],
                      residuals=resid, sigma=sigma, sigma_df=df, t_obs=t, k_regressors=k,
                      companion=companion, max_eig=max_eig, regressor_gram_eigenvalues=vals,
                      condition_number=cond, diagnostics=diagnostics, warnings=warnings,
                      sample_digest=sample_digest)


def estimate(snapshot):
    variables, lags, months, y, x, pre, digest = build_sample(snapshot)
    return estimate_from_series(variables, lags, months, y, x, pre, digest)


def bootstrap(est, draws, seed):
    """递归残差自举：从真实前p期观测出发，按残差向量有放回重抽。

    每次自举重新估计约化式，用于参数不确定性传播；结构集合不确定性
    在每次自举的约化式条件下另行搜索，不可与参数抽样混为一谈。
    """
    if draws < 0:
        raise ValueError('自举次数不能为负')
    t, n, p = est.t_obs, est.n, est.lags
    if not est.residuals:
        raise EstimationError('没有残差可供自举')
    rng = random.Random(seed)
    out = []
    for draw in range(draws):
        ystar = [list(r) for r in est.y]
        xstar = [list(r) for r in est.x]
        for ti in range(p, t):
            u = est.residuals[rng.randrange(t)]
            yrow = []
            for i in range(n):
                value = est.beta[0][i] + u[i]
                for lag in range(1, p + 1):
                    for j in range(n):
                        value += est.beta[1 + n * (lag - 1) + j][i] * ystar[ti - lag][j]
                yrow.append(value)
            ystar[ti] = yrow
            xrow = [1.0]
            for lag in range(1, p + 1):
                xrow.extend(ystar[ti - lag])
            xstar[ti] = xrow
        digest = hashlib.sha256((est.sample_digest + ':bootstrap:' + str(draw)).encode('utf-8')).hexdigest()
        try:
            out.append(estimate_from_series(est.variables, p, est.months, ystar, xstar, [], digest))
        except EstimationError as exc:
            out.append(exc)
    return out
