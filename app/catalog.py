"""读取业务资料的修订快照，保留撤回与冲突语义。"""
import copy
import hashlib
import json
import math
from dataclasses import dataclass

def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)

def _finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError('资料包含非有限数')
    if isinstance(value, dict):
        for v in value.values(): _finite(v)
    elif isinstance(value, list):
        for v in value: _finite(v)

@dataclass(frozen=True)
class Snapshot:
    _data: dict
    digest: str

    @property
    def metadata(self):
        return copy.deepcopy(self._data['metadata'])

    def rows(self, collection):
        return copy.deepcopy(self._data['collections'][collection])

def load(payload, schemas):
    if not isinstance(payload, dict) or payload.get('schema_version') != 1 or isinstance(payload.get('schema_version'), bool):
        raise ValueError('schema_version应为整数1')
    _finite(payload)
    collections = payload.get('collections')
    if not isinstance(collections, dict) or set(collections) != set(schemas):
        raise ValueError('业务集合不完整或包含未知集合')
    active = {}
    for name, fields in schemas.items():
        source = collections[name]
        if not isinstance(source, list):
            raise ValueError('业务集合应为数组')
        winners = {}
        identities = {}
        for row in source:
            if not isinstance(row, dict):
                raise ValueError('记录应为对象')
            key, revision = row.get('id'), row.get('revision')
            if not isinstance(key, str) or not key or type(revision) is not int or revision < 1:
                raise ValueError('记录身份或修订号无效')
            identity = key, revision
            encoded = canonical(row)
            if identity in identities and identities[identity] != encoded:
                raise ValueError('同身份同修订内容冲突')
            identities[identity] = encoded
            if 'deleted' in row and type(row['deleted']) is not bool:
                raise ValueError('撤回标志应为布尔值')
            if not row.get('deleted', False) and any(field not in row for field in fields):
                raise ValueError('记录缺少业务字段')
            if key not in winners or winners[key]['revision'] < revision:
                winners[key] = row
        active[name] = [copy.deepcopy(winners[key]) for key in sorted(winners) if not winners[key].get('deleted', False)]
    data = {'metadata': copy.deepcopy(payload.get('metadata', {})), 'collections': active}
    if not isinstance(data['metadata'], dict):
        raise ValueError('metadata应为对象')
    digest = hashlib.sha256(canonical(data).encode('utf-8')).hexdigest()
    return Snapshot(data, digest)
