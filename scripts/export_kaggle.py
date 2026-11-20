import os
import hashlib
import json
import zipfile
from pathlib import Path
import torch
from safetensors import safe_open

root=Path(os.environ.get('NTL_ROOT', Path(__file__).resolve().parents[1]))
base=root/'models/Qwen3-8B-Base'
checkpoint=root/'experiments/qwen3_8b_two_stage_20261113_v1/full_fit/final_epoch2'
destination=root/'kaggle_cli_upload/qwen8b_two_stage_dataset'
destination.mkdir(parents=True,exist_ok=True)
index=json.loads((base/'model.safetensors.index.json').read_text())['weight_map']
keys=[k for k in index if k=='model.embed_tokens.weight' or (k.startswith('model.layers.') and int(k.split('.')[2])<24)]
by_file={}
for key in keys:
    by_file.setdefault(index[key],[]).append(key)
manifest={'revision':'49e3418fbbbca6ecbdf9608b4d22e5a407081db4','canonical_dtype':'bfloat16','tensors':{}}
torch.set_num_threads(4)
for filename,items in by_file.items():
    with safe_open(str(base/filename),framework='pt',device='cpu') as weights:
        for key in items:
            tensor=weights.get_tensor(key).to(torch.bfloat16).contiguous()
            manifest['tensors'][key]={'shape':list(tensor.shape),'sha256_bf16':hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest()}
            del tensor
    print('HASHED',filename,'TENSORS',len(manifest['tensors']),flush=True)
provenance={'base_model':'Qwen/Qwen3-8B-Base','base_license':'Apache-2.0',
            'training_recipe_reference':'https://www.kaggle.com/code/nishantdahal/two-stage-lora-on-the-public-pool',
            'external_pool_source':'https://www.kaggle.com/datasets/nishantdahal/ntl-two-stage-lora',
            'training_run':'qwen3_8b_two_stage_20261113_v1','checkpoint':'full_fit/final_epoch2'}
bundle=destination/'qwen8b_two_stage_bundle.zip'
with zipfile.ZipFile(bundle,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
    for name in ['model_spec.json','trainables.safetensors']:
        z.write(checkpoint/name,name)
    z.write(base/'config.json','config.json')
    for p in (checkpoint/'tokenizer').rglob('*'):
        if p.is_file(): z.write(p,'tokenizer/'+str(p.relative_to(checkpoint/'tokenizer')))
    z.writestr('base_tensor_manifest.json',json.dumps(manifest,indent=2))
    z.writestr('provenance.json',json.dumps(provenance,indent=2))
metadata={'title':'Qwen3 8B Name That LLM Two Stage 20261119',
          'id':'yifeiorange/qwen3-8b-ntl-two-stage-20261119','licenses':[{'name':'CC-BY-SA-4.0'}]}
(destination/'dataset-metadata.json').write_text(json.dumps(metadata,indent=2))
print('BUNDLE_READY',str(bundle),'BYTES',bundle.stat().st_size,flush=True)
