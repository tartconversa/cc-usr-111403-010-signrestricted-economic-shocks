"""结构分析的编排入口：约化式诊断、识别搜索、证书、历史分解与方案对照。

设计边界（与 docs/structural-method.md 对应）：
  * 一次结构分解只使用一个正交旋转矩阵 Q，全部冲击、变量与期数限制在
    同一 Q 下同时检查；不会为不同冲击分别挑 Q。
  * “搜索到的范围”（search）与“认证集合边界”（certificate）分字段输出；
    搜索未命中只记录为 no_feasible_sample_found，不升级为不可行判断。
  * 参数不确定性（自举约化式）与给定参数下的集合不确定性（证书内/外近似）
    分开传播：不把各次抽样的响应混在一起取百分位当作唯一冲击的置信区间。
  * 历史分解使用同一可行结构，保留初始条件与结构残差，并输出重构误差。
  * 方案对照按限制方案版本输出逐项变化；标签不可比时明确拒绝比较。
"""
import copy
import hashlib
import math

from . import certify as certify_mod
from . import history as history_mod
from . import reduced
from . import sign
from .catalog import load
from .service import SCHEMAS

DEFAULT_HORIZONS = 8
DEFAULT_SEARCH_ATTEMPTS = 20000
DEFAULT_SEARCH_TARGET = 200
MAX_HORIZONS = 60

DEFAULT_OPTIONS = {
    'horizons': None,
    'search_attempts': DEFAULT_SEARCH_ATTEMPTS,
    'search_target': DEFAULT_SEARCH_TARGET,
    'seed': 0,
    'certify': True,
    'certify_grid': None,
    'certify_refine_budget': 12000,
    'certify_min_width': 1e-4,
    'bootstrap_draws': 0,
    'bootstrap_seed': 0,
    'bootstrap_certify': False,
    'include_history': True,
    'history_structure': 'first_certified',
}


class StructuralError(ValueError):
    """结构分析配置或资料不满足要求。"""


def _validate_options(options):
    merged = dict(DEFAULT_OPTIONS)
    if options is not None:
        if not isinstance(options, dict):
            raise StructuralError('options应为对象')
        unknown = set(options) - set(DEFAULT_OPTIONS)
        if unknown:
            raise StructuralError('未知配置项: ' + ', '.join(sorted(unknown)))
        merged.update(options)
    integer_fields = ('search_attempts', 'search_target', 'seed', 'certify_refine_budget',
                      'bootstrap_draws', 'bootstrap_seed')
    for name in integer_fields:
        value = merged[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise StructuralError('%s应为非负整数' % name)
    if merged['search_attempts'] == 0 or merged['search_target'] == 0:
        raise StructuralError('搜索次数与目标数应为正整数')
    if (merged['horizons'] is not None
            and (isinstance(merged['horizons'], bool) or not isinstance(merged['horizons'], int)
                 or merged['horizons'] < 0 or merged['horizons'] > MAX_HORIZONS)):
        raise StructuralError('horizons应为0至%d的整数' % MAX_HORIZONS)
    if merged['certify_grid'] is not None:
        grid = merged['certify_grid']
        if not isinstance(grid, (list, tuple)) or any(
                isinstance(c, bool) or not isinstance(c, int) or c < 1 for c in grid):
            raise StructuralError('certify_grid应为正整数数组')
        merged['certify_grid'] = tuple(grid)
    if not isinstance(merged['certify_min_width'], (int, float)) or isinstance(merged['certify_min_width'], bool) \
            or not math.isfinite(merged['certify_min_width']) or merged['certify_min_width'] <= 0:
        raise StructuralError('certify_min_width应为正的有限数')
    for name in ('certify', 'bootstrap_certify', 'include_history'):
        if not isinstance(merged[name], bool):
            raise StructuralError('%s应为布尔值' % name)
    if merged['history_structure'] not in ('first_certified', 'first_searched'):
        raise StructuralError('history_structure取值无效')
    return merged


def _shock_labels(metadata, n):
    labels = metadata.get('shock_labels')
    if labels is None:
        return ['shock%d' % j for j in range(n)]
    if not isinstance(labels, list) or len(labels) != n or any(
            not isinstance(label, str) or not label for label in labels):
        raise StructuralError('metadata.shock_labels应为长度等于变量数的非空字符串数组')
    if len(set(labels)) != n:
        raise StructuralError('metadata.shock_labels不得重复')
    return list(labels)


def _pick_history_structure(cert, search_result):
    if cert and cert.get('representative_structures'):
        rep = cert['representative_structures'][0]
        return rep['q'], {'source': 'certificate_feasible_box', 'branch': rep['branch'],
                          'angles': rep['angles']}
    if search_result and search_result.get('accepted'):
        item = search_result['accepted'][0]
        return item['q'], {'source': 'searched_rotation', 'branch': None,
                           'flips': item.get('flips')}
    return None, None


def _certify_structure(est, restrictions, margin, horizons, options):
    try:
        return certify_mod.certify(
            est, restrictions, margin, horizons,
            grid=options['certify_grid'],
            refine_budget=options['certify_refine_budget'],
            min_width=options['certify_min_width'],
        )
    except certify_mod.CertificateError as exc:
        return {'status': 'unsupported', 'reason': str(exc), 'response_bounds': []}


def _bootstrap_parameter(est, options, restrictions, margin, horizons):
    """参数不确定性：逐次自举重新估计约化式；集合按每次的约化式分别处理。

    返回每次抽样的约化式标量诊断，以及（可选）每次的证书状态与内/外区间。
    不把不同抽样、不同旋转的响应混在一起取百分位。
    """
    draws = reduced.bootstrap(est, options['bootstrap_draws'], options['bootstrap_seed'])
    samples = []
    failed = 0
    for draw in draws:
        if isinstance(draw, Exception):
            failed += 1
            samples.append({'status': 'estimation_failed', 'reason': str(draw)})
            continue
        record = {
            'status': 'ok',
            'sample_digest': draw.sample_digest,
            'max_eig': draw.max_eig,
            'stable': draw.max_eig < 1.0,
            'coefficients': draw.beta,
            'residual_covariance': draw.sigma,
        }
        if options['bootstrap_certify']:
            cert = _certify_structure(draw, restrictions, margin, horizons, options)
            record['certificate_status'] = cert.get('status')
            record['set_bounds'] = [
                {'shock': row['shock'], 'variable_index': row['variable_index'], 'horizon': row['horizon'],
                 'inner_lo': row.get('inner_lo'), 'inner_hi': row.get('inner_hi'),
                 'outer_lo': row.get('outer_lo'), 'outer_hi': row.get('outer_hi')}
                for row in cert.get('response_bounds', [])
                if row.get('identified_by_certificate')
            ]
            record['undecided_mass'] = cert.get('haar_mass', {}).get('undecided')
        samples.append(record)
    summary = _summarize_parameter_samples(samples)
    return {'draws': options['bootstrap_draws'], 'failed_draws': failed,
            'samples': samples, 'summary': summary,
            'interpretation': '每个抽样各有一个识别集合；summary只描述集合端点的抽样分布，'
                              '不是某一冲击响应的点识别置信区间。'}


def _quantile(sorted_values, q):
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (len(sorted_values) - 1) * q
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(sorted_values) - 1)
    if lo == hi:
        return sorted_values[lo]
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def _summarize_parameter_samples(samples):
    ok = [s for s in samples if s['status'] == 'ok']
    summary = {
        'ok_draws': len(ok),
        'stable_fraction': (sum(1 for s in ok if s['stable']) / len(ok)) if ok else None,
        'max_eig_values': sorted(s['max_eig'] for s in ok),
        'max_eig_quantiles': {},
        'structural_set_endpoint_quantiles': [],
    }
    if ok:
        eig = sorted(s['max_eig'] for s in ok)
        summary['max_eig_quantiles'] = {
            'q05': _quantile(eig, 0.05), 'q50': _quantile(eig, 0.5), 'q95': _quantile(eig, 0.95),
        }
    with_sets = [s for s in ok if s.get('set_bounds')]
    if with_sets:
        keys = {}
        for sample in with_sets:
            for row in sample['set_bounds']:
                keys.setdefault((row['shock'], row['variable_index'], row['horizon']), []).append(row)
        for (shock, var, horizon), rows in sorted(keys.items()):
            inner_lo = sorted(r['inner_lo'] for r in rows if r['inner_lo'] is not None)
            inner_hi = sorted(r['inner_hi'] for r in rows if r['inner_hi'] is not None)
            outer_lo = sorted(r['outer_lo'] for r in rows if r['outer_lo'] is not None)
            outer_hi = sorted(r['outer_hi'] for r in rows if r['outer_hi'] is not None)
            summary['structural_set_endpoint_quantiles'].append({
                'shock': shock, 'variable_index': var, 'horizon': horizon,
                'draws_with_certificate': len(rows),
                'inner_lo': {'q05': _quantile(inner_lo, 0.05), 'q50': _quantile(inner_lo, 0.5),
                             'q95': _quantile(inner_lo, 0.95)},
                'inner_hi': {'q05': _quantile(inner_hi, 0.05), 'q50': _quantile(inner_hi, 0.5),
                             'q95': _quantile(inner_hi, 0.95)},
                'outer_lo': {'q05': _quantile(outer_lo, 0.05), 'q50': _quantile(outer_lo, 0.5),
                             'q95': _quantile(outer_lo, 0.95)},
                'outer_hi': {'q05': _quantile(outer_hi, 0.05), 'q50': _quantile(outer_hi, 0.5),
                             'q95': _quantile(outer_hi, 0.95)},
                'note': '集合端点的抽样分位数；不同抽样各有自己的识别集合，不可当作点识别冲击的置信区间',
            })
    return summary


def _json_safe(value):
    """把结果整理成 allow_nan=False 可编码的结构，拒绝非有限数。"""
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise StructuralError('结果含非有限数，拒绝输出')
        return value
    return value


def analyze(payload, options=None):
    """对单份资料与单一限制方案执行结构分析。"""
    opts = _validate_options(options)
    snapshot = load(payload, SCHEMAS)
    metadata = snapshot.metadata
    try:
        est = reduced.estimate(snapshot)
    except reduced.EstimationError as exc:
        raise StructuralError('约化式估计失败: ' + str(exc)) from exc
    restrictions, margin, fingerprint, version = sign.compile_restrictions(metadata, est.variables)
    labels = _shock_labels(metadata, est.n)
    horizons = opts['horizons']
    if horizons is None:
        horizons = max(DEFAULT_HORIZONS, sign.max_restriction_horizon(restrictions))
    if horizons < sign.max_restriction_horizon(restrictions):
        raise StructuralError('horizons小于限制所需的最大期数')

    search_result = sign.search(est, restrictions, horizons, opts['search_target'],
                                opts['search_attempts'], opts['seed'], margin=margin)
    certificate = None
    if opts['certify']:
        certificate = _certify_structure(est, restrictions, margin, horizons, opts)

    structure_q, structure_info = _pick_history_structure(certificate, search_result)
    history_result = None
    history_error = None
    if opts['include_history'] and structure_q is not None:
        try:
            history_result = history_mod.decompose(est, structure_q, est.variables)
            history_result['structure_source'] = structure_info
        except (ValueError, ArithmeticError) as exc:
            history_error = str(exc)

    # 搜索范围与认证边界分开命名，避免把二者当同一证据。
    searched = {
        'status': search_result['status'],
        'attempts': search_result['attempts'],
        'accepted_count': search_result['accepted_count'],
        'duplicate_count': search_result['duplicate_count'],
        'best_remaining_violation': search_result['best_remaining_violation'],
        'margin': margin,
        'seed': opts['seed'],
        'ranges': sign.summarize_responses(est.variables, restrictions, search_result['accepted'], horizons),
        'note': search_result['note'],
    }
    certification = None
    if certificate is not None:
        certification = {
            'status': certificate.get('status'),
            'method': certificate.get('method'),
            'dimension': certificate.get('dimension'),
            'supported_dimensions': certificate.get('supported_dimensions'),
            'haar_mass': certificate.get('haar_mass'),
            'box_accounting': certificate.get('box_accounting'),
            'grid': certificate.get('grid'),
            'refine_budget': certificate.get('refine_budget'),
            'refinements_used': certificate.get('refinements_used'),
            'arithmetic': certificate.get('arithmetic'),
            'coverage_note': certificate.get('coverage_note'),
            'ranges': certificate.get('response_bounds'),
            'infeasibility_witness': certificate.get('infeasibility_witness'),
            'reason': certificate.get('reason'),
            'note': ('未决盒不为零时只能声明内/外近似，不能声明精确边界'
                     if certificate.get('status') == 'certified_feasible_partial'
                     else '证书覆盖整个参数域；随机搜索不参与此判定'),
        }

    diagnostics = {
        't_obs': est.t_obs,
        'k_regressors': est.k_regressors,
        'n_variables': est.n,
        'lags': est.lags,
        'months': est.months,
        'sample_digest': est.sample_digest,
        'intercept': est.intercept,
        'coefficients': est.beta,
        'residual_covariance': est.sigma,
        'residual_df': est.sigma_df,
        'companion_max_abs_eigenvalue': est.max_eig,
        'stable': est.max_eig < 1.0,
        'regressor_condition_number': est.condition_number,
        'residual_covariance_positive_definite': True,
        'residual_covariance_eigenvalues': est.diagnostics['covariance_eigenvalues'],
        'residual_covariance_condition_number': est.diagnostics['covariance_condition_number'],
        'ljung_box': est.diagnostics['ljung_box'],
        'jarque_bera': est.diagnostics['jarque_bera'],
        'warnings': est.warnings,
        'note': 'Ljung-Box/Jarque-Bera尾概率使用正态近似；稳定性看伴随矩阵最大模特征值',
    }

    parameter_uncertainty = None
    if opts['bootstrap_draws']:
        parameter_uncertainty = _bootstrap_parameter(est, opts, restrictions, margin, horizons)

    scheme = sign.scheme_summary(restrictions, est.n, margin, fingerprint, version)
    scheme['shock_labels'] = labels

    result = {
        'schema_version': 1,
        'input_digest': snapshot.digest,
        'data_version': metadata.get('data_version'),
        'horizons': horizons,
        'variables': est.variables,
        'shock_labels': labels,
        'reduced_form': diagnostics,
        'restriction_scheme': scheme,
        'searched_feasible_ranges': searched,
        'certified_set': certification,
        'set_uncertainty': {
            'kind': 'identification_set_given_parameters',
            'note': '给定点估计参数下，符号限制识别集合的内/外近似；'
                    '与参数抽样不确定性分开报告',
        },
        'parameter_uncertainty': parameter_uncertainty,
        'history_decomposition': history_result,
        'history_error': history_error,
        'cross_checks': _cross_checks(est, restrictions, margin, horizons, search_result,
                                      certificate, history_result),
    }
    return _json_safe(result)


def _cross_checks(est, restrictions, margin, horizons, search_result, certificate, history_result):
    checks = []
    if certificate and certificate.get('status') == 'unsupported':
        checks.append('维度超出证书支持范围；仅报告搜索到的范围，不声明识别集合边界')
    if certificate:
        if certificate.get('status') == 'certified_infeasible':
            if search_result['status'] == 'found_feasible_sample':
                checks.append('搜索与证书矛盾：证书称不可行但搜索命中，需复核实现')
            else:
                checks.append('证书判定不可行，搜索未命中与之一致；不可行结论只采信证书')
        elif certificate.get('status') == 'certified_feasible_complete':
            if search_result['status'] != 'found_feasible_sample':
                checks.append('证书称可行但搜索未命中，需增加搜索次数或复核随机种子')
        elif certificate.get('status') == 'certified_feasible_partial':
            checks.append('证书只给出部分内近似；未决盒已计入外近似，不能宣称边界精确')
    if restrictions and est.max_eig >= 1.0:
        checks.append('约化式不稳定，结构响应与历史分解仅作数值展示')
    if history_result:
        if history_result['checks']['max_abs_reconstruction_error'] > 1e-8:
            checks.append('历史分解重构误差超过容差，需检查实现或数值条件')
    if not checks:
        checks.append('未发现跨模块矛盾')
    return checks


# ------------------------------------------------------------------ 方案对照

def _scheme_from_entry(metadata, entry, variables):
    scheme_metadata = copy.deepcopy(metadata)
    if 'restrictions' in entry:
        scheme_metadata['restrictions'] = copy.deepcopy(entry['restrictions'])
    if 'positive_margin' in entry:
        scheme_metadata['positive_margin'] = entry['positive_margin']
    if 'restriction_version' in entry:
        scheme_metadata['restriction_version'] = entry['restriction_version']
    if 'shock_labels' in entry:
        scheme_metadata['shock_labels'] = copy.deepcopy(entry['shock_labels'])
    restrictions, margin, fingerprint, version = sign.compile_restrictions(scheme_metadata, variables)
    labels = _shock_labels(scheme_metadata, len(variables))
    return scheme_metadata, restrictions, margin, fingerprint, version, labels


def compare_schemes(payload, entries, options=None):
    """按限制方案版本对照。entries 为 [{'label', 'restrictions', 'positive_margin',
    'restriction_version', 'shock_labels'}]。

    只有当变量顺序与冲击标签完全一致时，才逐冲击比较；否则拒绝把不同标签
    的数值并排。
    """
    if not isinstance(entries, list) or not entries:
        raise StructuralError('entries应为非空数组')
    snapshot = load(payload, SCHEMAS)
    metadata = snapshot.metadata
    try:
        est = reduced.estimate(snapshot)
    except reduced.EstimationError as exc:
        raise StructuralError('约化式估计失败: ' + str(exc)) from exc
    options = _validate_options(options)
    base_labels = _shock_labels(metadata, est.n)

    compiled = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise StructuralError('每个方案应为对象')
        scheme_metadata, restrictions, margin, fingerprint, version, labels = _scheme_from_entry(
            metadata, entry, est.variables)
        compiled.append({'label': entry.get('label', version if version is not None else fingerprint[:12]),
                         'restrictions': restrictions, 'margin': margin,
                         'fingerprint': fingerprint, 'version': version,
                         'shock_labels': labels, 'metadata': scheme_metadata})
    names = [item['label'] for item in compiled]
    if len(set(names)) != len(names):
        raise StructuralError('方案label不得重复')

    comparable = all(item['shock_labels'] == compiled[0]['shock_labels'] for item in compiled)
    if not comparable:
        raise StructuralError('方案间的冲击标签不一致，不能按冲击并列比较；'
                              '请提供一致的shock_labels或分别报告')

    horizons = options['horizons']
    if horizons is None:
        horizons = max([DEFAULT_HORIZONS] + [sign.max_restriction_horizon(item['restrictions'])
                                            for item in compiled])
    outputs = []
    for item in compiled:
        search_result = sign.search(est, item['restrictions'], horizons, options['search_target'],
                                    options['search_attempts'], options['seed'], margin=item['margin'])
        certificate = None
        if options['certify']:
            certificate = _certify_structure(est, item['restrictions'], item['margin'], horizons, options)
        outputs.append({
            'label': item['label'],
            'version': item['version'],
            'fingerprint': item['fingerprint'],
            'positive_margin': item['margin'],
            'restrictions': [{'shock': s, 'variable': est.variables[v], 'horizon': h,
                              'sign': '+' if sg > 0 else '-'}
                             for s, v, h, sg in item['restrictions']],
            'shock_labels': item['shock_labels'],
            'search_status': search_result['status'],
            'search_accepted_count': search_result['accepted_count'],
            'searched_ranges': sign.summarize_responses(est.variables, item['restrictions'],
                                                        search_result['accepted'], horizons),
            'certificate_status': certificate.get('status') if certificate else None,
            'certificate_haar_mass': certificate.get('haar_mass') if certificate else None,
            'certified_ranges': certificate.get('response_bounds') if certificate else None,
            'infeasibility_witness': certificate.get('infeasibility_witness') if certificate else None,
        })

    comparison = []
    keys = set()
    for output in outputs:
        for row in output['searched_ranges']:
            if row.get('kind') == 'searched_range' and 'variable' in row:
                keys.add((row['shock'], row['variable'], row['horizon']))
    for shock, variable, horizon in sorted(keys):
        entry = {'shock': shock, 'shock_label': base_labels[shock],
                 'variable': variable, 'horizon': horizon, 'by_scheme': []}
        variable_index = est.variables.index(variable)
        for output in outputs:
            search_row = next((r for r in output['searched_ranges']
                               if r.get('kind') == 'searched_range'
                               and r.get('shock') == shock and r.get('variable') == variable
                               and r.get('horizon') == horizon), None)
            cert_row = None
            if output['certified_ranges'] is not None:
                cert_row = next((r for r in output['certified_ranges']
                                 if r['shock'] == shock and r['variable_index'] == variable_index
                                 and r['horizon'] == horizon), None)
            entry['by_scheme'].append({
                'label': output['label'],
                'searched_found_min': search_row['found_min'] if search_row else None,
                'searched_found_max': search_row['found_max'] if search_row else None,
                'searched_status': search_row['status'] if search_row else None,
                'certified_inner_lo': cert_row['inner_lo'] if cert_row else None,
                'certified_inner_hi': cert_row['inner_hi'] if cert_row else None,
                'certified_outer_lo': cert_row['outer_lo'] if cert_row else None,
                'certified_outer_hi': cert_row['outer_hi'] if cert_row else None,
            })
        comparison.append(entry)

    return _json_safe({
        'schema_version': 1,
        'input_digest': snapshot.digest,
        'data_version': metadata.get('data_version'),
        'horizons': horizons,
        'variables': est.variables,
        'shock_labels': base_labels,
        'comparable_by_shock_label': True,
        'schemes': outputs,
        'comparison': comparison,
        'note': '每个方案各自搜索、各自认证；不同方案的结果不能拼成同一识别集合。'
                '方案版本与指纹用于复核具体限制内容。',
    })


def main(argv=None):
    import json
    import sys
    from pathlib import Path
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        raise SystemExit('请指定资料JSON路径，可选第二参数为配置JSON路径')
    payload = json.loads(Path(args[0]).read_text(encoding='utf-8'))
    options = None
    entries = None
    if len(args) > 1:
        options = json.loads(Path(args[1]).read_text(encoding='utf-8'))
        if not isinstance(options, dict):
            raise SystemExit('配置应为JSON对象')
        if 'compare_entries' in options:
            entries = options.pop('compare_entries')
    if entries is not None:
        result = compare_schemes(payload, entries, options)
    else:
        result = analyze(payload, options)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
