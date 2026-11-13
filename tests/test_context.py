"""验证业务摘要以及读取端的修订、隔离和错误边界。"""
import copy
import json
from pathlib import Path
import unittest
from app.catalog import load
from app.service import inspect, SCHEMAS

SAMPLE = json.loads((Path(__file__).resolve().parents[1] / 'fixtures' / 'sample.json').read_text(encoding='utf-8'))
EXPECTED = [{'month': 2, 'Y': [2, 3, 4], 'X': [1, 1, 2, 3]}, {'month': 3, 'Y': [3, 4, 5], 'X': [1, 2, 3, 4]}]

class InitialContextTests(unittest.TestCase):
    def test_business_summary_and_provenance(self):
        result = inspect(SAMPLE)
        self.assertEqual(result['summary'], EXPECTED)
        self.assertEqual(len(result['input_digest']), 64)
        self.assertEqual(result['data_version'], 'sample-1')

    def test_order_and_exact_retry_do_not_change_snapshot(self):
        data = copy.deepcopy(SAMPLE)
        for rows in data['collections'].values():
            rows.reverse()
            if rows: rows.append(copy.deepcopy(rows[0]))
        self.assertEqual(inspect(data), inspect(SAMPLE))

    def test_tombstone_and_restore(self):
        data = copy.deepcopy(SAMPLE)
        name = next(iter(SCHEMAS))
        original = copy.deepcopy(data['collections'][name][0])
        data['collections'][name].append({'id': original['id'], 'revision': 2, 'deleted': True})
        snap = load(data, SCHEMAS)
        self.assertNotIn(original['id'], [r['id'] for r in snap.rows(name)])
        restored = {**original, 'revision': 3}
        data['collections'][name].append(restored)
        self.assertIn(restored, load(data, SCHEMAS).rows(name))

    def test_conflict_is_not_hidden_by_later_revision(self):
        data = copy.deepcopy(SAMPLE)
        rows = next(iter(data['collections'].values()))
        old = copy.deepcopy(rows[0])
        rows.extend([{**old, 'unknown': '冲突内容'}, {**old, 'revision': 9}])
        with self.assertRaisesRegex(ValueError, '冲突'): load(data, SCHEMAS)

    def test_result_mutation_does_not_modify_snapshot(self):
        snap = load(SAMPLE, SCHEMAS)
        name = next(iter(SCHEMAS))
        copy_rows = snap.rows(name)
        copy_rows[0]['id'] = 'changed'
        meta = snap.metadata
        meta['data_version'] = 'changed'
        self.assertNotEqual(snap.rows(name)[0]['id'], 'changed')
        self.assertEqual(snap.metadata['data_version'], 'sample-1')

    def test_missing_field_and_nonfinite_rejected(self):
        data = copy.deepcopy(SAMPLE)
        name = next(iter(SCHEMAS))
        del data['collections'][name][0][SCHEMAS[name][0]]
        with self.assertRaisesRegex(ValueError, '业务字段'): load(data, SCHEMAS)
        data = copy.deepcopy(SAMPLE)
        data['metadata']['bad'] = float('nan')
        with self.assertRaisesRegex(ValueError, '非有限'): load(data, SCHEMAS)

if __name__ == '__main__': unittest.main()
