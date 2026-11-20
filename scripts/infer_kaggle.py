import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import gc
import hashlib
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModel, AutoTokenizer


class TaskLoRALinear(nn.Module):
    def __init__(self, base, a, b, scale):
        super().__init__()
        self.base = base
        self.register_buffer('a', a.to(device=base.weight.device, dtype=torch.float32))
        self.register_buffer('b', b.to(device=base.weight.device, dtype=torch.float32))
        self.scale = scale

    def forward(self, x):
        result = self.base(x)
        delta = F.linear(F.linear(x.to(self.a.dtype), self.a), self.b)
        return result + (delta * self.scale).to(result.dtype)


class TaskClassifier(nn.Module):
    def __init__(self, backbone, spec, state):
        super().__init__()
        self.backbone = backbone
        self.backbone.norm = nn.Identity()
        self.depths = spec['depths']
        self.width = backbone.config.hidden_size
        self.head_device = backbone.embed_tokens.weight.device
        self.norms = nn.ModuleList([nn.LayerNorm(self.width, dtype=torch.float32) for _ in self.depths]).to(self.head_device)
        self.plain = nn.Linear(self.width * len(self.depths), len(spec['labels']), dtype=torch.float32).to(self.head_device)
        expected = set()
        replaced = 0
        prefix = 'encoder.base_model.model.'
        for key, value in state.items():
            if '.lora_A.default.weight' not in key:
                continue
            assert key.startswith(prefix)
            module_path = key[len(prefix):].split('.lora_A.default.weight')[0]
            b_key = key.replace('.lora_A.default.weight', '.lora_B.default.weight')
            original = backbone.get_submodule(module_path)
            assert tuple(value.shape) == (spec['rank'], original.in_features)
            assert tuple(state[b_key].shape) == (original.out_features, spec['rank'])
            parent_path, name = module_path.rsplit('.', 1)
            setattr(backbone.get_submodule(parent_path), name,
                    TaskLoRALinear(original, value, state[b_key], spec['alpha']/spec['rank']))
            expected.update([key, b_key])
            replaced += 1
        assert replaced == spec['layers'] * 7, replaced
        for index, norm in enumerate(self.norms):
            norm.load_state_dict({name: state[f'norms.{index}.{name}'] for name in ('weight','bias')})
            expected.update(f'norms.{index}.{name}' for name in ('weight','bias'))
        self.plain.load_state_dict({name: state[f'plain.{name}'] for name in ('weight','bias')})
        expected.update(['plain.weight','plain.bias','centered.weight','centered.bias'])
        assert expected == set(state), sorted(set(state)-expected)
        self.eval()

    def forward(self, input_ids, attention_mask):
        states = self.backbone(input_ids=input_ids, attention_mask=attention_mask,
                               output_hidden_states=True, return_dict=True, use_cache=False).hidden_states
        features = []
        for depth, norm in zip(self.depths, self.norms):
            hidden = states[depth]
            mask = attention_mask.to(hidden.device).unsqueeze(-1).float()
            pooled = (hidden.float() * mask).sum(1)/mask.sum(1).clamp_min(1)
            features.append(norm(pooled.to(self.head_device)))
        return self.plain(torch.cat(features, dim=-1))


def find_base(root):
    choices = []
    for config_file in root.rglob('config.json'):
        try:
            config = json.loads(config_file.read_text())
        except (ValueError,OSError):
            continue
        if config.get('model_type') == 'qwen3' and config.get('hidden_size') == 4096 and config.get('num_hidden_layers') == 36:
            if (config_file.parent/'model.safetensors.index.json').is_file() or (config_file.parent/'model.safetensors').is_file():
                choices.append(config_file.parent)
    assert len(choices) == 1, choices
    return choices[0]


def verify_backbone(base, manifest):
    index_file = base/'model.safetensors.index.json'
    if index_file.is_file():
        index = json.loads(index_file.read_text())['weight_map']
    else:
        with safe_open(str(base/'model.safetensors'),framework='pt',device='cpu') as f:
            index = {key:'model.safetensors' for key in f.keys()}
    by_file = {}
    for key in manifest['tensors']:
        actual = key if key in index else key.removeprefix('model.')
        assert actual in index, key
        by_file.setdefault(index[actual], []).append((key,actual))
    count = 0
    for filename, keys in by_file.items():
        with safe_open(str(base/filename),framework='pt',device='cpu') as f:
            for key, actual in keys:
                tensor = f.get_tensor(actual).to(torch.bfloat16).contiguous()
                digest = hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest()
                expected = manifest['tensors'][key]
                assert list(tensor.shape) == expected['shape'] and digest == expected['sha256_bf16'], key
                count += 1
                del tensor
        print('BASE_SHARD_VERIFIED',filename,'tensors_so_far',count,flush=True)
        gc.collect()
    print('ALL_BASE_TENSORS_VERIFIED',count,flush=True)


def predict(model, tokenizer, texts, max_length, batch_size=2):
    encoded = tokenizer(list(texts), add_special_tokens=False, truncation=True, max_length=max_length-1, padding=False)
    ids = [tokens+[tokenizer.eos_token_id] for tokens in encoded['input_ids']]
    order = sorted(range(len(ids)), key=lambda i:len(ids[i]))
    result = np.zeros((len(ids),model.plain.out_features),dtype=np.float32)
    with torch.inference_mode():
        for start in range(0,len(order),batch_size):
            chosen = order[start:start+batch_size]
            batch = tokenizer.pad({'input_ids':[ids[i] for i in chosen]},padding=True,pad_to_multiple_of=8,return_tensors='pt')
            batch = {key:value.to(model.head_device) for key,value in batch.items() if key in ('input_ids','attention_mask')}
            scores = model(**batch)
            assert torch.isfinite(scores).all(), 'Non-finite model output'
            result[chosen] = scores.softmax(-1).cpu().numpy()
            if start % 100 == 0:
                print('PREDICTED',min(start+len(chosen),len(ids)),'/',len(ids),flush=True)
    assert np.isfinite(result).all() and np.allclose(result.sum(1),1,atol=1e-5)
    return result


def main():
    start = time.monotonic()
    torch.set_num_threads(4)
    print('TORCH',torch.__version__,'GPUS',torch.cuda.device_count(),flush=True)
    root = Path('/kaggle/input')
    archives = list(root.rglob('qwen8b_two_stage_bundle.zip'))
    if archives:
        assert len(archives) == 1, archives
        assets = Path('/kaggle/working/qwen8b_assets')
        assets.mkdir(exist_ok=True)
        with zipfile.ZipFile(archives[0]) as archive:
            archive.extractall(assets)
    else:
        extracted = [p.parent for p in root.rglob('trainables.safetensors')
                     if (p.parent/'model_spec.json').is_file() and (p.parent/'base_tensor_manifest.json').is_file()]
        assert len(extracted) == 1, extracted
        assets = extracted[0]
    spec = json.loads((assets/'model_spec.json').read_text())
    manifest = json.loads((assets/'base_tensor_manifest.json').read_text())
    base = find_base(root)
    print('BASE',str(base),'SPEC',spec,flush=True)
    verify_backbone(base,manifest)
    canonical = Path('/kaggle/input/competitions/name-that-llm/test.csv')
    candidates = [canonical] if canonical.is_file() else list(root.rglob('test.csv'))
    assert len(candidates) == 1, candidates
    test = pd.read_csv(candidates[0], dtype={'id':str})
    labels = np.asarray(spec['labels'])
    assert len(labels) == 18 and len(set(labels)) == 18
    assert test.id.is_unique and test.text.notna().all() and len(test) > 0
    tokenizer = AutoTokenizer.from_pretrained(assets/'tokenizer',local_files_only=True)
    tokenizer.padding_side = 'right'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    assert tokenizer.eos_token_id is not None
    config = AutoConfig.from_pretrained(assets,local_files_only=True)
    config.num_hidden_layers = spec['layers']
    config.use_cache = False
    config.pad_token_id = tokenizer.pad_token_id
    devices = torch.cuda.device_count()
    assert devices >= 1
    memory = {i: int(torch.cuda.get_device_properties(i).total_memory)-3*1024**3 for i in range(devices)}
    assert sum(memory.values()) >= 21*1024**3, memory
    backbone = AutoModel.from_pretrained(base,config=config,torch_dtype=torch.float32,
                 attn_implementation='sdpa',device_map='balanced',max_memory=memory,low_cpu_mem_usage=True,local_files_only=True)
    device_map = getattr(backbone,'hf_device_map',{})
    print('DEVICE_MAP',device_map,flush=True)
    assert all(str(value) not in ('cpu','disk') for value in device_map.values()), device_map
    model = TaskClassifier(backbone,spec,load_file(str(assets/'trainables.safetensors'),device='cpu'))
    probabilities = predict(model,tokenizer,test.text.astype(str).tolist(),spec['max_length'])
    submission = pd.DataFrame({'id':test.id,'model':labels[probabilities.argmax(1)]})
    assert len(submission)==len(test) and submission.id.is_unique and submission.model.isin(labels).all()
    output = Path('/kaggle/working/submission.parquet')
    submission.to_parquet(output,index=False)
    check = pd.read_parquet(output)
    assert check.equals(submission)
    report = dict(model='Qwen3-8B-Base first24 multi-depth two-stage LoRA',checkpoint='full_fit/final_epoch2',
                  oof_balanced_accuracy=.5564031817051951,rows=len(test),dtype='float32',
                  base_tensors_verified=len(manifest['tensors']),
                  seconds=time.monotonic()-start,gpu_names=[torch.cuda.get_device_name(i) for i in range(devices)])
    Path('/kaggle/working/inference_report.json').write_text(json.dumps(report,indent=2))
    print('FINISHED',report,flush=True)


if __name__ == '__main__': main()
