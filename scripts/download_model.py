import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_DISABLE_XET'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import json
import shutil
import time
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

root = Path(os.environ.get('NTL_ROOT', Path(__file__).resolve().parents[1]))
out = root / 'research/highscore_20261113'
out.mkdir(parents=True, exist_ok=True)
target = root / 'models/Qwen3-8B-Base'
status = out / 'qwen8b_preparation_status.json'
state = {'model_id': 'Qwen/Qwen3-8B-Base', 'status': 'preparing', 'gpu_used': False,
         'target': str(target), 'started_at': time.strftime('%Y-%m-%d %H:%M:%S')}
try:
    if shutil.disk_usage(root).free < 40 * 1024**3:
        raise RuntimeError('Fewer than 40 GiB free; stopping model download')
    status.write_text(json.dumps(state, indent=2))
    info = HfApi().model_info(state['model_id'], revision='49e3418fbbbca6ecbdf9608b4d22e5a407081db4')
    state.update(status='downloading', revision=info.sha)
    status.write_text(json.dumps(state, indent=2))
    print('PINNED', info.sha, flush=True)
    snapshot_download(state['model_id'], revision=info.sha, local_dir=target, max_workers=4,
                      allow_patterns=['*.json', 'tokenizer*', '*.model', '*.txt'])
    index = json.loads((target / 'model.safetensors.index.json').read_text())
    state['expected_weight_bytes'] = index['metadata']['total_size']
    state['weight_files'] = sorted(set(index['weight_map'].values()))
    status.write_text(json.dumps(state, indent=2))
    print('WEIGHT_BYTES', state['expected_weight_bytes'], flush=True)
    snapshot_download(state['model_id'], revision=info.sha, local_dir=target, max_workers=4,
                      allow_patterns=['*.safetensors'])
    if not all((target / filename).is_file() for filename in state['weight_files']):
        raise RuntimeError('A required weight shard is missing after download')
    state.update(status='ready', finished_at=time.strftime('%Y-%m-%d %H:%M:%S'))
    status.write_text(json.dumps(state, indent=2))
    print('MODEL_READY', str(target), flush=True)
except Exception as exc:
    state.update(status='failed', error=repr(exc))
    status.write_text(json.dumps(state, indent=2))
    raise
