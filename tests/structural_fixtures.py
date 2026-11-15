"""结构 VAR 的测试资料构造与共用断言。"""
import json
import math
import random
from pathlib import Path

from app.catalog import load
from app.service import SCHEMAS

SCHEMAS_UNDER_TEST = SCHEMAS


def synthetic_payload(months=81, lags=1, variables=None, ar=None, seed=11,
                      restrictions=None, margin=0.0, shock_labels=None, data_version='syn-1'):
    variables = variables or ['industry', 'export', 'price']
    n = len(variables)
    rng = random.Random(seed)
    ar = ar or [[0.4, 0.15, 0.05],
                [0.2, 0.25, -0.1],
                [0.1, 0.05, 0.3]][:n]
    values = [0.0] * n
    rows = []
    for t in range(1, months + 1):
        shocks = [rng.gauss(0.0, 0.3) for _ in range(n)]
        new = [1.0 + 0.5 * i + sum(ar[i][j] * values[j] for j in range(n)) + shocks[i]
               for i in range(n)]
        values = new
        rows.append({'id': 'm%d' % t, 'revision': 1, 'month': t,
                     'values': {v: values[i] for i, v in enumerate(variables)}})
    metadata = {'data_version': data_version, 'variables': variables, 'lags': lags,
                'positive_margin': margin,
                'restrictions': restrictions if restrictions is not None else []}
    if shock_labels is not None:
        metadata['shock_labels'] = shock_labels
    return {'schema_version': 1, 'metadata': metadata,
            'collections': {'months': rows}}


def snapshot_of(payload):
    return load(payload, SCHEMAS_UNDER_TEST)


def three_shock_signs(margin=0.0):
    return [
        {'shock': 0, 'variable': 'industry', 'horizon': 0, 'sign': '+'},
        {'shock': 1, 'variable': 'export', 'horizon': 0, 'sign': '+'},
        {'shock': 2, 'variable': 'price', 'horizon': 0, 'sign': '+'},
    ]


def two_variable_payload(**kwargs):
    return synthetic_payload(variables=['industry', 'export'],
                             ar=[[0.4, 0.15], [0.2, 0.25]],
                             **kwargs)


def two_variable_estimate(payload):
    from app import reduced
    return reduced.estimate(snapshot_of(payload))


def one_variable_payload(**kwargs):
    return synthetic_payload(variables=['industry'], ar=[[0.4]], **kwargs)
