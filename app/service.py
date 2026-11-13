"""发布带资料身份的已有业务摘要。"""
import json
import sys
from pathlib import Path
from .catalog import load
from .baseline import analyze

SCHEMAS = {'months': ['month', 'values']}

def inspect(payload):
    snapshot = load(payload, SCHEMAS)
    return {'schema_version': 1, 'input_digest': snapshot.digest,
            'data_version': snapshot.metadata.get('data_version'), 'summary': analyze(snapshot)}

def main():
    if len(sys.argv) != 2:
        raise SystemExit('请指定资料JSON路径')
    payload = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    print(json.dumps(inspect(payload), ensure_ascii=False, allow_nan=False, indent=2))

if __name__ == '__main__':
    main()
