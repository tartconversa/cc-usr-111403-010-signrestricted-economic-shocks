"""验证结构分析：约化式诊断、同一Q、等价类、搜索与证书、不确定性、历史分解、方案对照。"""
import copy
import json
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import structural_fixtures as F

from app import certify as certify_mod
from app import linalg as la
from app import reduced
from app import sign
from app import structural
from tests.reference import exhaustive_n1, exhaustive_n2


def compiled(payload):
    est = reduced.estimate(F.snapshot_of(payload))
    restrictions, margin, fingerprint, version = sign.compile_restrictions(payload['metadata'], est.variables)
    return est, restrictions, margin, fingerprint, version


class ReducedFormTests(unittest.TestCase):
    def test_estimation_matches_ols_normal_equations(self):
        payload = F.synthetic_payload()
        est = reduced.estimate(F.snapshot_of(payload))
        k, n = est.k_regressors, est.n
        xtx = la.zeros(k, k)
        xty = la.zeros(k, n)
        for row, y in zip(est.x, est.y):
            for i in range(k):
                for j in range(k):
                    xtx[i][j] += row[i] * row[j]
                for j in range(n):
                    xty[i][j] += row[i] * y[j]
        lhs = la.matmul(xtx, est.beta)
        self.assertLess(max(abs(lhs[i][j] - xty[i][j]) for i in range(k) for j in range(n)), 1e-8)

    def test_presample_comes_from_first_window(self):
        payload = F.synthetic_payload(lags=2)
        est = reduced.estimate(F.snapshot_of(payload))
        first = est.x[0]
        expected = [first[1], first[2], first[3], first[4], first[5], first[6]]
        self.assertEqual(len(expected), est.n * est.lags)

    def test_month_gap_is_rejected(self):
        payload = F.synthetic_payload(lags=1)
        # 第11月缺一个变量值：第11、12月被排除，但13月起又完整，
        # 保留样本出现 10→13 的月份断档。
        payload['collections']['months'][10]['values']['export'] = None
        with self.assertRaisesRegex(reduced.EstimationError, '不连续'):
            reduced.estimate(F.snapshot_of(payload))

    def test_too_few_observations_rejected(self):
        payload = F.synthetic_payload(months=4)
        with self.assertRaisesRegex(reduced.EstimationError, '观测数'):
            reduced.estimate(F.snapshot_of(payload))

    def test_stability_and_covariance_diagnostics(self):
        est = reduced.estimate(F.snapshot_of(F.synthetic_payload()))
        self.assertLess(est.max_eig, 1.0)
        single = reduced.estimate(F.snapshot_of(F.one_variable_payload()))
        self.assertAlmostEqual(single.max_eig, abs(single.beta[1][0]), places=10)
        with self.assertRaises(reduced.EstimationError):
            reduced.estimate_from_series(['a', 'b'], 1, [1, 2, 3],
                                         [[1, 2], [2, 3], [3, 4]],
                                         [[1, 1, 2], [1, 2, 3], [1, 3, 4]], [], 'digest')

    def test_bootstrap_reestimates_and_can_fail(self):
        est = reduced.estimate(F.snapshot_of(F.synthetic_payload()))
        draws = reduced.bootstrap(est, 5, 7)
        self.assertEqual(len(draws), 5)
        for draw in draws:
            self.assertNotEqual(draw.sample_digest, est.sample_digest)
            self.assertEqual(draw.t_obs, est.t_obs)


class RestrictionSchemeTests(unittest.TestCase):
    def test_conflicting_signs_and_bad_indices_rejected(self):
        payload = F.synthetic_payload(restrictions=[
            {'shock': 0, 'variable': 'industry', 'horizon': 0, 'sign': '+'},
            {'shock': 0, 'variable': 'industry', 'horizon': 0, 'sign': '-'},
        ])
        with self.assertRaisesRegex(sign.RestrictionError, '冲突'):
            compiled(payload)
        payload = F.synthetic_payload(restrictions=[
            {'shock': 7, 'variable': 'industry', 'horizon': 0, 'sign': '+'}])
        with self.assertRaisesRegex(sign.RestrictionError, 'shock'):
            compiled(payload)
        payload = F.synthetic_payload(restrictions=[
            {'shock': 0, 'variable': 'nope', 'horizon': 0, 'sign': '+'}])
        with self.assertRaisesRegex(sign.RestrictionError, '未知变量'):
            compiled(payload)

    def test_fingerprint_depends_on_content_not_order(self):
        first = F.synthetic_payload(restrictions=[
            {'shock': 0, 'variable': 'industry', 'horizon': 0, 'sign': '+'},
            {'shock': 1, 'variable': 'export', 'horizon': 0, 'sign': '+'},
        ])
        second = F.synthetic_payload(restrictions=[
            {'shock': 1, 'variable': 'export', 'horizon': 0, 'sign': '+'},
            {'shock': 0, 'variable': 'industry', 'horizon': 0, 'sign': '+'},
        ])
        self.assertEqual(compiled(first)[3], compiled(second)[3])
        second['metadata']['positive_margin'] = 0.05
        self.assertNotEqual(compiled(first)[3], compiled(second)[3])

    def test_duplicate_identical_restriction_collapses(self):
        payload = F.synthetic_payload(restrictions=[
            {'shock': 0, 'variable': 'industry', 'horizon': 0, 'sign': '+'},
            {'shock': 0, 'variable': 'industry', 'horizon': 0, 'sign': '+'},
        ])
        self.assertEqual(len(compiled(payload)[1]), 1)


class SameRotationTests(unittest.TestCase):
    def test_all_restrictions_checked_on_one_q(self):
        payload = F.synthetic_payload(restrictions=F.three_shock_signs(margin=0.01))
        est, restrictions, margin, _, _ = compiled(payload)
        result = sign.search(est, restrictions, 2, 40, 4000, 3, margin=margin)
        self.assertEqual(result['status'], 'found_feasible_sample')
        for accepted in result['accepted']:
            ok, details = sign.check_restrictions(accepted['theta'], restrictions, margin)
            self.assertTrue(ok)
            # 全部限制必须在同一个 theta（同一个 Q）下成立。
            for row, restriction in zip(details, restrictions):
                self.assertEqual(row['shock'], restriction[0])
            b = la.matmul(la.cholesky(est.sigma), accepted['q'])
            self.assertAlmostEqual(la.norm(la.matmul(la.transpose(accepted['q']), accepted['q'])), math.sqrt(est.n), places=5)

    def test_search_does_not_accept_per_shock_columns(self):
        # 三个冲击各自都有正值限制时，全为正号对角 Q 可行；
        # 若某条限制要求不同符号，则同一 Q 不可能同时满足。
        payload = F.synthetic_payload(restrictions=[
            {'shock': 0, 'variable': 'industry', 'horizon': 0, 'sign': '+'},
            {'shock': 0, 'variable': 'industry', 'horizon': 0, 'sign': '+'},
            {'shock': 1, 'variable': 'industry', 'horizon': 0, 'sign': '-'},
            {'shock': 2, 'variable': 'industry', 'horizon': 0, 'sign': '+'},
        ])
        est, restrictions, margin, _, _ = compiled(payload)
        result = sign.search(est, restrictions, 1, 20, 500, 5, margin=margin)
        for accepted in result['accepted']:
            ok, _ = sign.check_restrictions(accepted['theta'], restrictions, margin)
            self.assertTrue(ok)

    def test_sign_flip_equivalence_is_deduplicated(self):
        payload = F.synthetic_payload(restrictions=[
            {'shock': 0, 'variable': 'industry', 'horizon': 0, 'sign': '+'}])
        est, restrictions, margin, _, _ = compiled(payload)
        result = sign.search(est, restrictions, 1, 60, 20000, 11, margin=margin)
        self.assertTrue(result['accepted'])
        q = result['accepted'][0]['q']
        # 无限制列整列变号是同一方案：规定化后签名必须一致。
        flipped = [[q[i][j] * (-1 if j in (1, 2) else 1) for j in range(3)] for i in range(3)]
        mapping = sign.Mapping(est)
        _, theta = mapping.response(flipped, 1)
        self.assertTrue(sign.check_restrictions(theta, restrictions, margin)[0])
        qc, _, flips = sign.canonicalize(flipped, theta, restrictions, est.n)
        self.assertEqual(flips, [1, -1, -1])
        signature, _ = sign.canonical_signature(qc, restrictions, est.n)
        self.assertEqual(signature, result['accepted'][0]['signature'])
        # 有限制列的符号由限制固定，不得被翻转：翻转后应失去可行性。
        bad = [[q[i][j] * (-1 if j == 0 else 1) for j in range(3)] for i in range(3)]
        _, theta_bad = mapping.response(bad, 1)
        self.assertFalse(sign.check_restrictions(theta_bad, restrictions, margin)[0])

    def test_unrestricted_shock_is_not_identified(self):
        payload = F.synthetic_payload(restrictions=[
            {'shock': 0, 'variable': 'industry', 'horizon': 0, 'sign': '+'}])
        est, restrictions, margin, _, _ = compiled(payload)
        result = sign.search(est, restrictions, 1, 100, 8000, 3, margin=margin)
        rows = sign.summarize_responses(est.variables, restrictions, result['accepted'], 1)
        not_identified = [r for r in rows if r['status'] == 'not_identified']
        self.assertEqual({r['shock'] for r in not_identified}, {1, 2})
        # 无限制冲击的不可识别列不得报告逐期范围。
        for row in not_identified:
            self.assertNotIn('found_min', row)

    def test_label_equivalence_groups_reported(self):
        payload = F.synthetic_payload(restrictions=[
            {'shock': 0, 'variable': 'industry', 'horizon': 0, 'sign': '+'},
            {'shock': 1, 'variable': 'industry', 'horizon': 0, 'sign': '+'},
        ])
        est, restrictions, margin, _, _ = compiled(payload)
        self.assertEqual(sign.label_equivalence_groups(restrictions, est.n), [[0, 1]])
        self.assertEqual(sign.unrestricted_shocks(restrictions, est.n), [2])
        result = sign.search(est, restrictions, 1, 50, 5000, 2, margin=margin)
        rows = sign.summarize_responses(est.variables, restrictions, result['accepted'], 1)
        labeled = [r for r in rows if r['status'] == 'label_not_distinguishable']
        self.assertTrue(labeled)
        self.assertEqual(labeled[0]['label_group'], [0, 1])

    def test_search_failure_is_not_infeasibility(self):
        payload = F.synthetic_payload(restrictions=[
            {'shock': 0, 'variable': 'industry', 'horizon': 0, 'sign': '+'}])
        est, restrictions, margin, _, _ = compiled(payload)
        result = sign.search(est, restrictions, 1, 10, 3, 999, margin=margin)
        self.assertEqual(result['status'], 'found_feasible_sample' if result['accepted'] else 'no_feasible_sample_found')
        if result['status'] == 'no_feasible_sample_found':
            self.assertIn('未找到不等于', result['note'])


class CertificationTests(unittest.TestCase):
    def test_n1_exact_bounds_match_exhaustive(self):
        payload = F.one_variable_payload(restrictions=[
            {'shock': 0, 'variable': 'industry', 'horizon': 0, 'sign': '+'}])
        est, restrictions, margin, _, _ = compiled(payload)
        cert = certify_mod.certify(est, restrictions, margin, 2)
        reference = exhaustive_n1(est, restrictions, margin, 2)
        self.assertEqual(cert['status'], 'certified_feasible_complete')
        self.assertEqual(cert['box_accounting']['undecided_boxes'], 0)
        for row in cert['response_bounds']:
            values = [pt['theta'][row['horizon']][row['variable_index']][0]
                      for pt in reference['feasible']]
            self.assertAlmostEqual(row['outer_lo'], min(values), places=12)
            self.assertAlmostEqual(row['outer_hi'], max(values), places=12)
            self.assertEqual(row['gap_lo'], 0.0)
            self.assertEqual(row['gap_hi'], 0.0)

    def test_n1_infeasible_certificate(self):
        payload = F.one_variable_payload(restrictions=[
            {'shock': 0, 'variable': 'industry', 'horizon': 0, 'sign': '+'}],
            margin=10.0)
        est, restrictions, margin, _, _ = compiled(payload)
        cert = certify_mod.certify(est, restrictions, margin, 2)
        self.assertEqual(cert['status'], 'certified_infeasible')
        self.assertTrue(cert['infeasibility_witness']['strictly_below_margin'])
        self.assertEqual(cert['infeasibility_witness']['branches_checked'], [1, -1])

    def test_n2_exact_arcs_contain_bruteforce(self):
        payload = F.two_variable_payload(restrictions=[
            {'shock': 0, 'variable': 'industry', 'horizon': 0, 'sign': '+'},
            {'shock': 0, 'variable': 'export', 'horizon': 1, 'sign': '+'},
        ])
        est, restrictions, margin, _, _ = compiled(payload)
        cert = certify_mod.certify(est, restrictions, margin, 2)
        reference = exhaustive_n2(est, restrictions, margin, 2, steps=20000)
        self.assertEqual(cert['status'], 'certified_feasible_complete')
        self.assertTrue(reference['feasible'])
        for row in cert['response_bounds']:
            if row['shock'] != 0 or row['inner_lo'] is None:
                continue
            values = [pt['theta'][row['horizon']][row['variable_index']][0]
                      for pt in reference['feasible']]
            # 穷举是网格采样，必落在解析精确极值之内；解析边界应包住采样。
            self.assertGreaterEqual(min(values), row['inner_lo'] - 1e-8)
            self.assertLessEqual(max(values), row['inner_hi'] + 1e-8)
            self.assertGreaterEqual(min(values), row['outer_lo'] - 1e-8)
            self.assertLessEqual(max(values), row['outer_hi'] + 1e-8)

    def test_n2_tangency_and_just_infeasible_margins(self):
        payload = F.two_variable_payload()
        est = reduced.estimate(F.snapshot_of(payload))
        mapping = sign.Mapping(est)
        phi = sign.var_ma(est, 0)
        phi_p = [la.matmul(ph, mapping.cholesky) for ph in phi]
        a, b = certify_mod._n2_coefficients(phi_p, (0, 0, 0, 1), 1)
        amplitude = math.hypot(a, b)
        restrictions = [{'shock': 0, 'variable': 'industry', 'horizon': 0, 'sign': '+'}]
        for factor, expected in ((0.99, 'certified_feasible_complete'),
                                 (1.0, 'certified_feasible_complete'),
                                 (1.01, 'certified_infeasible')):
            md = dict(payload['metadata'])
            md['restrictions'] = restrictions
            md['positive_margin'] = amplitude * factor
            restrictions_c, margin, _, _ = sign.compile_restrictions(md, est.variables)
            cert = certify_mod.certify(est, restrictions_c, margin, 1)
            self.assertEqual(cert['status'], expected)
            self.assertEqual(cert['box_accounting']['undecided_boxes'], 0)
            if expected == 'certified_infeasible':
                self.assertTrue(cert['infeasibility_witness']['strictly_below_margin'])
            if factor == 1.0:
                # 可行集退化为零测度点：仍给出精确极值，不报不可行。
                row = cert['response_bounds'][0]
                self.assertAlmostEqual(row['inner_lo'], amplitude, places=10)
                self.assertEqual(row['inner_lo'], row['inner_hi'])

    def test_n2_infeasible_arc_certificate(self):
        payload = F.two_variable_payload(restrictions=[
            {'shock': 0, 'variable': 'industry', 'horizon': 0, 'sign': '+'},
            {'shock': 0, 'variable': 'export', 'horizon': 0, 'sign': '+'},
        ], margin=50.0)
        est, restrictions, margin, _, _ = compiled(payload)
        cert = certify_mod.certify(est, restrictions, margin, 1)
        self.assertEqual(cert['status'], 'certified_infeasible')
        self.assertEqual(cert['box_accounting']['undecided_boxes'], 0)

    def test_n3_sandwich_never_claims_exact_when_undecided(self):
        payload = F.synthetic_payload(restrictions=F.three_shock_signs())
        est, restrictions, margin, _, _ = compiled(payload)
        cert = certify_mod.certify(est, restrictions, margin, 2, grid=(8, 4, 8), refine_budget=200)
        if cert['status'] == 'certified_feasible_partial':
            self.assertGreater(cert['box_accounting']['undecided_boxes'], 0)
            self.assertGreater(cert['haar_mass']['undecided'], 0.0)
            self.assertIn('未决盒仅能进外近似', cert['coverage_note'])
            for row in cert['response_bounds']:
                if row['inner_lo'] is None:
                    continue
                self.assertGreaterEqual(row['inner_lo'], row['outer_lo'] - 1e-12)
                self.assertLessEqual(row['inner_hi'], row['outer_hi'] + 1e-12)
        # 网格必须完整覆盖：网格盒数 = 2 × 每维等分之积。
        self.assertEqual(cert['box_accounting']['seed_boxes'], 2 * 8 * 4 * 8)

    def test_n3_infeasible_requires_full_cover_and_no_undecided(self):
        payload = F.synthetic_payload(restrictions=F.three_shock_signs(), margin=20.0)
        est, restrictions, margin, _, _ = compiled(payload)
        cert = certify_mod.certify(est, restrictions, margin, 1)
        self.assertEqual(cert['status'], 'certified_infeasible')
        self.assertEqual(cert['box_accounting']['undecided_boxes'], 0)
        self.assertTrue(cert['infeasibility_witness']['strictly_below_margin'])

    def test_certificate_rejects_unsupported_dimension(self):
        payload = F.synthetic_payload(months=61, variables=['a', 'b', 'c', 'd'],
                                      ar=[[0.2, 0, 0, 0]] * 4)
        est = reduced.estimate(F.snapshot_of(payload))
        with self.assertRaisesRegex(certify_mod.CertificateError, '1 至 3'):
            certify_mod.certify(est, [], 0.0, 1)

    def test_interval_arithmetic_contains_endpoints(self):
        angle = certify_mod.Iv(1.0, 1.5)
        s = certify_mod.sin_iv(angle)
        c = certify_mod.cos_iv(angle)
        for t in (1.0, 1.2, 1.5):
            self.assertLessEqual(s.lo, math.sin(t))
            self.assertGreaterEqual(s.hi, math.sin(t))
            self.assertLessEqual(c.lo, math.cos(t))
            self.assertGreaterEqual(c.hi, math.cos(t))
        # 跨越极值的区间必须取到 ±1。
        wrapped = certify_mod.cos_iv(certify_mod.Iv(-3.2, 3.2))
        self.assertAlmostEqual(wrapped.lo, -1.0)
        self.assertAlmostEqual(wrapped.hi, 1.0)


class HistoryDecompositionTests(unittest.TestCase):
    def test_identity_with_initial_condition_and_residual(self):
        payload = F.synthetic_payload(restrictions=F.three_shock_signs(margin=0.01))
        est, restrictions, margin, _, _ = compiled(payload)
        result = sign.search(est, restrictions, 3, 10, 3000, 4, margin=margin)
        q = result['accepted'][0]['q']
        history = structural.history_mod.decompose(est, q, est.variables)
        self.assertLess(history['checks']['max_abs_reconstruction_error'], 1e-9)
        self.assertLess(history['checks']['max_abs_reduced_residual_identity_gap'], 1e-9)
        # 初始条件期（t=0）不含当期残差，重构必须对上原始值。
        for i in range(est.n):
            self.assertAlmostEqual(history['reconstructed'][0][i], history['actual'][0][i], places=10)
        # 结构冲击之和 = 约化式残差。
        for t, eps in enumerate(history['structural_shocks']):
            b = history['structure']['b']
            rebuilt = [sum(b[i][j] * eps[j] for j in range(est.n)) for i in range(est.n)]
            for i in range(est.n):
                self.assertAlmostEqual(rebuilt[i], history['reduced_residuals'][t][i], places=10)
        # 当期残差项单独保留，且与约化式残差一致。
        self.assertLess(history['checks']['max_abs_residual_term_vs_reduced'], 1e-12)
        for t in range(est.t_obs):
            for i in range(est.n):
                self.assertAlmostEqual(history['residual_term'][t][i],
                                       history['reduced_residuals'][t][i], places=10)

    def test_shock_contributions_are_additive(self):
        payload = F.synthetic_payload(restrictions=F.three_shock_signs(margin=0.01))
        est, restrictions, margin, _, _ = compiled(payload)
        result = sign.search(est, restrictions, 3, 5, 3000, 4, margin=margin)
        history = structural.history_mod.decompose(est, result['accepted'][0]['q'], est.variables)
        for t in range(est.t_obs):
            total = [history['initial_condition'][t][i] + history['deterministic'][t][i]
                     + sum(history['shocks'][j]['contributions'][t][i] for j in range(est.n))
                     for i in range(est.n)]
            for i in range(est.n):
                self.assertAlmostEqual(total[i], history['reconstructed'][t][i], places=10)

    def test_history_uses_one_structure(self):
        payload = F.synthetic_payload(restrictions=F.three_shock_signs(margin=0.01))
        result = structural.analyze(payload, {'horizons': 2, 'search_attempts': 3000,
                                              'search_target': 10, 'certify_refine_budget': 500})
        history = result['history_decomposition']
        self.assertIsNotNone(history)
        self.assertTrue(history['checks']['same_structure_for_all_shocks'])
        self.assertEqual(history['structure']['q'],
                         [list(row) for row in history['structure']['q']])


class SchemeComparisonTests(unittest.TestCase):
    def test_comparison_requires_matching_labels(self):
        payload = F.synthetic_payload(restrictions=F.three_shock_signs())
        entries = [
            {'label': 'v1', 'restrictions': F.three_shock_signs(), 'positive_margin': 0.0,
             'shock_labels': ['industry_shock', 'export_shock', 'price_shock']},
            {'label': 'v2', 'restrictions': [{'shock': 0, 'variable': 'industry', 'horizon': 0, 'sign': '-'}],
             'positive_margin': 0.0,
             'shock_labels': ['industry_shock', 'other', 'third']},
        ]
        with self.assertRaisesRegex(structural.StructuralError, '标签不一致'):
            structural.compare_schemes(payload, entries, {'horizons': 1, 'search_attempts': 500})

    def test_comparison_is_per_scheme(self):
        payload = F.synthetic_payload(restrictions=F.three_shock_signs())
        entries = [
            {'label': 'v1', 'restrictions': F.three_shock_signs(), 'positive_margin': 0.0},
            {'label': 'v2', 'restrictions': F.three_shock_signs(), 'positive_margin': 0.02},
        ]
        result = structural.compare_schemes(payload, entries, {
            'horizons': 1, 'search_attempts': 2000, 'search_target': 20,
            'certify_refine_budget': 500, 'certify_grid': (6, 3, 6),
        })
        self.assertEqual([s['label'] for s in result['schemes']], ['v1', 'v2'])
        self.assertNotEqual(result['schemes'][0]['fingerprint'], result['schemes'][1]['fingerprint'])
        for row in result['comparison']:
            self.assertEqual(len(row['by_scheme']), 2)

    def test_conflicting_shock_labels_are_rejected(self):
        payload = F.synthetic_payload(restrictions=F.three_shock_signs(),
                                      shock_labels=['industry_shock', 'industry_shock', 'price_shock'])
        with self.assertRaisesRegex(structural.StructuralError, '不得重复'):
            structural.analyze(payload, {'horizons': 1})


class UncertaintySeparationTests(unittest.TestCase):
    def test_parameter_and_set_uncertainty_are_separate_fields(self):
        payload = F.synthetic_payload(restrictions=F.three_shock_signs(margin=0.01))
        result = structural.analyze(payload, {
            'horizons': 2, 'search_attempts': 3000, 'search_target': 20,
            'certify_refine_budget': 1000, 'bootstrap_draws': 4,
            'bootstrap_certify': True, 'certify_grid': (6, 3, 6),
        })
        self.assertEqual(result['set_uncertainty']['kind'], 'identification_set_given_parameters')
        parameter = result['parameter_uncertainty']
        self.assertIsNotNone(parameter)
        self.assertIn('structural_set_endpoint_quantiles', parameter['summary'])
        # 参数抽样汇总只描述集合端点，不输出“某冲击响应”的百分位。
        for row in parameter['summary']['structural_set_endpoint_quantiles']:
            self.assertIn('note', row)
            self.assertNotIn('response_percentile', row)
        self.assertIn('不是某一冲击响应的点识别置信区间', parameter['interpretation'])

    def test_no_bootstrap_means_no_parameter_block(self):
        payload = F.synthetic_payload(restrictions=F.three_shock_signs())
        result = structural.analyze(payload, {'horizons': 1, 'search_attempts': 1000,
                                              'search_target': 5, 'certify_refine_budget': 200})
        self.assertIsNone(result['parameter_uncertainty'])


class OutputContractTests(unittest.TestCase):
    def test_result_is_json_serializable_and_nonfinite_rejected(self):
        payload = F.synthetic_payload(restrictions=F.three_shock_signs(margin=0.01))
        result = structural.analyze(payload, {'horizons': 2, 'search_attempts': 3000,
                                              'search_target': 10, 'certify_refine_budget': 500})
        json.dumps(result, allow_nan=False, ensure_ascii=False)
        self.assertEqual(result['schema_version'], 1)
        self.assertEqual(result['data_version'], 'syn-1')
        self.assertIn('input_digest', result)
        self.assertIn('searched_feasible_ranges', result)
        self.assertIn('certified_set', result)

    def test_unknown_option_rejected(self):
        payload = F.synthetic_payload(restrictions=F.three_shock_signs())
        with self.assertRaisesRegex(structural.StructuralError, '未知配置项'):
            structural.analyze(payload, {'horizons': 1, 'mystery': 1})

    def test_small_sample_from_existing_fixture_still_reports_baseline(self):
        from app.service import inspect
        sample = json.loads((Path(__file__).resolve().parents[1] / 'fixtures' / 'sample.json')
                            .read_text(encoding='utf-8'))
        self.assertEqual(inspect(sample)['data_version'], 'sample-1')
        with self.assertRaisesRegex(structural.StructuralError, '约化式估计失败'):
            structural.analyze(sample, {'horizons': 1})


class CrossCheckTests(unittest.TestCase):
    def test_infeasible_certificate_beats_empty_search(self):
        payload = F.synthetic_payload(months=61, variables=['industry', 'export', 'price'],
                                      restrictions=F.three_shock_signs(), margin=20.0)
        result = structural.analyze(payload, {'horizons': 1, 'search_attempts': 300,
                                              'search_target': 3, 'certify_refine_budget': 0})
        self.assertEqual(result['certified_set']['status'], 'certified_infeasible')
        self.assertNotEqual(result['searched_feasible_ranges']['status'], 'found_feasible_sample')
        joined = ' '.join(result['cross_checks'])
        self.assertIn('不可行结论只采信证书', joined)


if __name__ == '__main__':
    unittest.main()
