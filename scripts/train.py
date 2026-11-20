from __future__ import annotations

import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '4'

import argparse
import gc
import hashlib
import json
import math
import random
import signal
import subprocess
import sys
import time
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from peft import LoraConfig, TaskType, get_peft_model
from safetensors.torch import load_file, save_file
from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.model_selection import GroupKFold
from transformers import AutoConfig, AutoModel, AutoTokenizer, Qwen3Config, Qwen3Model, get_cosine_schedule_with_warmup


ROOT = Path(os.environ.get('NTL_ROOT', Path(__file__).resolve().parents[1]))
REVISION = '49e3418fbbbca6ecbdf9608b4d22e5a407081db4'
DEPTHS = (2, 4, 6, 8, 12, 16, 20, 24)
STOP = False


def log(message):
    print(datetime.now(timezone.utc).isoformat(timespec='seconds'), message, flush=True)


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)


def handle_signal(*_):
    global STOP
    STOP = True


def cancelled(out):
    if STOP or (out / 'STOP').exists():
        raise InterruptedError('Stopped by signal or STOP file')


def gpu_snapshot():
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.total,memory.used,utilization.gpu',
                             '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True)
    rows = []
    for line in result.stdout.strip().splitlines():
        index, total, used, util = [int(v.strip()) for v in line.split(',')]
        rows.append(dict(index=index, total_mib=total, used_mib=used, utilization=util))
    return rows


def idle_gpu(rows):
    eligible = [r for r in rows if r['used_mib'] < 2048 and r['utilization'] < 10
                and r['total_mib'] - r['used_mib'] >= 27000]
    return min(eligible, key=lambda r: (r['used_mib'], r['index'])) if eligible else None


def status(out, phase, **kwargs):
    data = dict(phase=phase, pid=os.getpid(), updated_utc=datetime.now(timezone.utc).isoformat(), **kwargs)
    write_json(out / 'status.json', data)


def wait_for_resources(out, root, timeout_hours):
    start = time.monotonic()
    preparation = root / 'research/highscore_20261113/qwen8b_preparation_status.json'
    previous = None
    consecutive = 0
    logged_phase = None
    while time.monotonic() - start < timeout_hours * 3600:
        cancelled(out)
        try:
            state = json.loads(preparation.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            state = {'status': 'missing'}
        if state.get('status') == 'failed':
            raise RuntimeError('Model preparation failed: ' + state.get('error', 'unknown'))
        if state.get('status') != 'ready':
            phase = 'waiting_for_weights'
            status(out, phase, preparation_state=state.get('status'), gpu_training_started=False)
            previous = None
            consecutive = 0
        else:
            if state.get('revision') != REVISION:
                raise RuntimeError('Base-model revision differs from registered protocol')
            rows = gpu_snapshot()
            chosen = idle_gpu(rows)
            phase = 'waiting_for_idle_gpu'
            status(out, phase, gpu_snapshot=rows, gpu_training_started=False)
            if chosen is not None:
                consecutive = consecutive + 1 if previous == chosen['index'] else 1
                previous = chosen['index']
                if consecutive >= 3:
                    return chosen['index']
            else:
                previous = None
                consecutive = 0
        if logged_phase != phase:
            log(phase)
            logged_phase = phase
        time.sleep(20)
    raise TimeoutError('Resource wait exceeded configured limit; no busy GPU was used')


class MultiDepthClassifier(nn.Module):
    def __init__(self, backbone, depths, classes, rank=16):
        super().__init__()
        self.depths = tuple(depths)
        backbone.norm = nn.Identity()
        spec = LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, r=rank, lora_alpha=2 * rank,
                          lora_dropout=0.05, bias='none',
                          target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'])
        self.encoder = get_peft_model(backbone, spec)
        width = backbone.config.hidden_size
        self.norms = nn.ModuleList([nn.LayerNorm(width, dtype=torch.float32) for _ in depths])
        self.plain = nn.Linear(width * len(depths), classes, dtype=torch.float32)
        self.centered = nn.Linear(width * len(depths), classes, dtype=torch.float32)

    def features(self, input_ids, attention_mask):
        states = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                              output_hidden_states=True, return_dict=True, use_cache=False).hidden_states
        mask = attention_mask.unsqueeze(-1).float()
        denominator = mask.sum(1).clamp_min(1)
        pooled = [(states[d].float() * mask).sum(1) / denominator for d in self.depths]
        return torch.cat([norm(x) for norm, x in zip(self.norms, pooled)], dim=-1)

    def forward(self, input_ids, attention_mask):
        z = self.features(input_ids, attention_mask)
        return self.plain(z), z

    def triplet_loss(self, batch, class_weights=None):
        y = batch['labels']
        logits, z = self(batch['input_ids'], batch['attention_mask'])
        if len(y) % 3:
            raise ValueError('A training microbatch must consist of complete triplets')
        centered = z.reshape(-1, 3, z.shape[-1])
        centered = (centered - centered.mean(1, keepdim=True)).reshape_as(z)
        a = F.cross_entropy(logits.float(), y, weight=class_weights)
        b = F.cross_entropy(self.centered(centered).float(), y, weight=class_weights)
        return a + b


def trainable_state(model):
    return {n: p.detach().cpu().contiguous().clone() for n, p in model.named_parameters() if p.requires_grad}


def restore_state(model, state):
    parameters = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if set(parameters) != set(state):
        raise ValueError('Checkpoint trainable names do not match architecture')
    with torch.no_grad():
        for name, parameter in parameters.items():
            parameter.copy_(state[name].to(parameter.device, parameter.dtype))


def save_checkpoint(model, tokenizer, path, labels, stage, epoch, args):
    path.mkdir(parents=True, exist_ok=True)
    save_file(trainable_state(model), str(path / 'trainables.safetensors'))
    if tokenizer is not None:
        tokenizer.save_pretrained(path / 'tokenizer')
    write_json(path / 'model_spec.json', dict(base_model='Qwen/Qwen3-8B-Base', revision=REVISION,
                layers=24, depths=list(DEPTHS), final_backbone_norm='Identity', rank=16, alpha=32,
                dropout=.05, stage=stage, epoch=epoch, labels=labels, max_length=args.max_length,
                fold=args.fold, torch_version=torch.__version__, validation_is_exploratory=True))


class TokenRows(Dataset):
    def __init__(self, frame, tokenizer, labels, max_length):
        label_map = {label: i for i, label in enumerate(labels)}
        encoded = tokenizer(frame.text.astype(str).tolist(), add_special_tokens=False,
                            truncation=True, max_length=max_length-1, padding=False)
        eos = tokenizer.eos_token_id
        self.tokens = [ids + [eos] for ids in encoded['input_ids']]
        self.y = np.array([label_map[label] for label in frame.model], dtype=np.int64)
        self.groups = frame.prompt_id.astype(str).to_numpy()
        self.truncated = sum(len(ids) >= max_length for ids in self.tokens)

    def __len__(self): return len(self.y)

    def __getitem__(self, i): return dict(input_ids=self.tokens[i], labels=int(self.y[i]))


def make_triplets(dataset, seed):
    rng = np.random.default_rng(seed)
    grouped = defaultdict(lambda: defaultdict(list))
    for index, (group, label) in enumerate(zip(dataset.groups, dataset.y)):
        grouped[group][int(label)].append(index)
    triplets = []
    skipped = 0
    for by_label in grouped.values():
        labels = list(by_label)
        count = sum(map(len, by_label.values()))
        if len(labels) < 3:
            skipped += count
            continue
        for _ in range(math.ceil(count / 3)):
            chosen = rng.choice(labels, 3, replace=False)
            triplets.append([int(rng.choice(by_label[int(label)])) for label in chosen])
    rng.shuffle(triplets)
    if not triplets:
        raise ValueError('No same-prompt triplets with three distinct labels')
    return triplets, skipped


def collator(tokenizer):
    def collate(rows):
        batch = tokenizer.pad({'input_ids': [r['input_ids'] for r in rows]}, padding=True,
                              pad_to_multiple_of=8, return_tensors='pt')
        batch['labels'] = torch.tensor([r['labels'] for r in rows], dtype=torch.long)
        return batch
    return collate


def optimizer_for(model, lora_lr, head_lr):
    adapters, heads = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (adapters if 'lora_' in name else heads).append(parameter)
    if not adapters or not heads:
        raise ValueError('Missing trainable adapters or task heads')
    return torch.optim.AdamW([dict(params=adapters, lr=lora_lr), dict(params=heads, lr=head_lr)],
                              weight_decay=.01, foreach=False)


def to_cuda(batch): return {name: value.to('cuda:0', non_blocking=True) for name, value in batch.items()}


def memory_preflight(model, args, out):
    original = trainable_state(model)
    optimizer = optimizer_for(model, 1e-4, 1e-3)
    torch.cuda.reset_peak_memory_stats()
    model.train()
    batch = dict(input_ids=torch.randint(0, model.encoder.config.vocab_size, (3,args.max_length), device='cuda'),
                 attention_mask=torch.ones((3,args.max_length), dtype=torch.long, device='cuda'),
                 labels=torch.tensor([0,1,2], device='cuda'))
    with torch.autocast('cuda', dtype=torch.bfloat16):
        loss = model.triplet_loss(batch)
    if not torch.isfinite(loss):
        raise FloatingPointError('Non-finite loss in memory preflight')
    loss.backward()
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
    optimizer.step()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_reserved()/1024**3
    write_json(out/'memory_preflight.json', dict(peak_reserved_gib=peak, sequence_length=args.max_length,
                                                microbatch=3, trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)))
    log(f'PREFLIGHT peak_reserved={peak:.2f} GiB')
    if peak > 24:
        raise MemoryError('Preflight exceeds 24 GiB training budget')
    optimizer.zero_grad(set_to_none=True)
    del optimizer, loss, batch
    restore_state(model, original)
    del original
    gc.collect()
    torch.cuda.empty_cache()


def evaluate(model, dataset, tokenizer, args, out, phase):
    model.eval()
    probabilities = []
    total_loss = 0.
    loader = DataLoader(dataset, batch_size=args.eval_batch_size, shuffle=False,
                        collate_fn=collator(tokenizer), num_workers=0, pin_memory=True)
    with torch.inference_mode():
        for step, batch in enumerate(loader):
            cancelled(out)
            batch = to_cuda(batch)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits, _ = model(batch['input_ids'], batch['attention_mask'])
            if not torch.isfinite(logits).all():
                raise FloatingPointError('Non-finite evaluation logits')
            total_loss += F.cross_entropy(logits.float(), batch['labels'], reduction='sum').item()
            probabilities.append(logits.float().softmax(-1).cpu().numpy())
            if (step+1) % 200 == 0:
                status(out, phase, evaluation_batch=step+1, evaluation_batches=len(loader), gpu_training_started=True)
    p = np.concatenate(probabilities)
    return dict(ba=float(balanced_accuracy_score(dataset.y, p.argmax(1))), loss=total_loss/len(dataset)), p


def fit_stage(model, dataset, valid, tokenizer, labels, args, out, stage, epochs, history, run_out=None):
    progress = run_out or out
    lora_lr, head_lr = (1e-4,1e-3) if stage=='stage1' else (5e-5,5e-4)
    optimizer = optimizer_for(model, lora_lr, head_lr)
    initial_triplets, _ = make_triplets(dataset, 20261113)
    total_steps = math.ceil(len(initial_triplets)/6)*epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=max(1,int(.1*total_steps)),
                                               num_training_steps=total_steps)
    weights = None
    if stage=='stage1':
        counts = np.bincount(dataset.y, minlength=len(labels))
        weights = torch.tensor(len(dataset)/(len(labels)*counts), dtype=torch.float32, device='cuda')
    best = -1.
    for epoch in range(1,epochs+1):
        triplets, skipped = make_triplets(dataset, 20261113+epoch)
        loader = DataLoader(dataset, batch_sampler=triplets, collate_fn=collator(tokenizer),
                            num_workers=0, pin_memory=True)
        log(f'{stage} EPOCH {epoch}/{epochs} triplets={len(triplets)} skipped_rows={skipped}')
        write_json(out/f'{stage}_sampling.json', dict(triplets=len(triplets), rows_without_three_distinct_labels=skipped,
                   note='Random same-prompt triplets have a passage-equivalent epoch budget, not exact once-only row coverage.'))
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.
        torch.cuda.reset_peak_memory_stats()
        for step,batch in enumerate(loader):
            cancelled(progress)
            if step % 50==0:
                free,total = torch.cuda.mem_get_info()
                if free < 3*1024**3:
                    raise MemoryError('Less than 3 GiB physical VRAM remains; stopping own task')
            batch = to_cuda(batch)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                loss = model.triplet_loss(batch,weights)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite training loss')
            divisor = min(6,len(loader)-(step//6)*6)
            (loss/divisor).backward()
            total_loss += loss.detach().item()
            if (step+1)%6==0 or step+1==len(loader):
                norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
                optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True)
            if (step+1)%50==0:
                log(f'{stage} epoch={epoch} batch={step+1}/{len(loader)} loss={total_loss/(step+1):.4f}')
                status(progress,'training_'+stage,fold=args.fold,epoch=epoch,batch=step+1,batches=len(loader),gpu_training_started=True)
        checkpoint = out/f'{stage}_epoch{epoch}'
        save_checkpoint(model,tokenizer,checkpoint,labels,stage,epoch,args)
        status(progress,'evaluating_'+stage,fold=args.fold,epoch=epoch,gpu_training_started=True)
        train_metrics,_ = evaluate(model,dataset,tokenizer,args,progress,'evaluating_'+stage+'_train')
        if valid is None:
            val_metrics,p = {'ba':None,'loss':None},None
        else:
            val_metrics,p = evaluate(model,valid,tokenizer,args,progress,'evaluating_'+stage+'_validation')
        row = dict(stage=stage,fold=args.fold,epoch=epoch,train_size=len(dataset),validation_size=len(valid) if valid is not None else 0,
                   train_set_ba=train_metrics['ba'],val_ba=val_metrics['ba'],train_set_loss=train_metrics['loss'],
                   val_loss=val_metrics['loss'],optimization_loss=total_loss/len(loader),
                   peak_gpu_reserved_gib=torch.cuda.max_memory_reserved()/1024**3)
        history.append(row)
        pd.DataFrame(history).to_csv(progress/'training_history.csv',index=False)
        if valid is not None:
            np.savez_compressed(out/f'{stage}_epoch{epoch}_validation.npz',probabilities=p,labels=valid.y,
                                label_names=np.asarray(labels),fold=np.full(len(valid),args.fold))
            write_json(out/f'{stage}_epoch{epoch}_classification_report.json',classification_report(valid.y,p.argmax(1),
                             labels=list(range(len(labels))),target_names=labels,output_dict=True,zero_division=0))
        log('EPOCH_RESULT '+json.dumps(row))
        if val_metrics['ba'] is not None: best=max(best,val_metrics['ba'])
    optimizer.zero_grad(set_to_none=True)
    del optimizer,scheduler
    gc.collect();torch.cuda.empty_cache()
    return best


def self_test():
    torch.manual_seed(73)
    cfg=Qwen3Config(vocab_size=128,hidden_size=64,intermediate_size=128,num_hidden_layers=4,
                   num_attention_heads=4,num_key_value_heads=2,head_dim=16,max_position_embeddings=128,use_cache=False)
    cfg._attn_implementation='eager'
    model=MultiDepthClassifier(Qwen3Model(cfg),(1,2,4),18,rank=4)
    model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.train()
    batch=dict(input_ids=torch.randint(0,128,(3,16)),attention_mask=torch.ones((3,16),dtype=torch.long),labels=torch.tensor([0,1,2]))
    frozen={n:p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad}
    opt=optimizer_for(model,1e-4,1e-3)
    loss=model.triplet_loss(batch)
    loss.backward()
    assert torch.isfinite(loss) and any(p.grad is not None for n,p in model.named_parameters() if 'lora_' in n)
    assert model.plain.weight.grad is not None and model.centered.weight.grad is not None
    opt.step();opt.zero_grad(set_to_none=True)
    assert all(torch.equal(dict(model.named_parameters())[n],v) for n,v in frozen.items())
    model.eval()
    with torch.no_grad(): expected=model(batch['input_ids'],batch['attention_mask'])[0].clone()
    state=trainable_state(model)
    with tempfile.TemporaryDirectory(prefix='ntl-q8-checkpoint-check-') as temporary:
        checkpoint=Path(temporary)/'trainables.safetensors'
        save_file(state,str(checkpoint))
        state=load_file(str(checkpoint))
    with torch.no_grad(): model.plain.weight.zero_()
    restore_state(model,state)
    with torch.no_grad(): actual=model(batch['input_ids'],batch['attention_mask'])[0]
    assert torch.equal(expected,actual)
    from types import SimpleNamespace
    sample=SimpleNamespace(groups=np.array(['a']*6+['b']*2),y=np.array([0,1,2,0,1,2,4,4]))
    triplets,skipped=make_triplets(sample,73)
    assert skipped==2 and len(triplets)==2
    assert all(len(set(sample.y[triplet]))==3 and len(set(sample.groups[triplet]))==1 for triplet in triplets)
    assert idle_gpu([dict(index=0,total_mib=32607,used_mib=16906,utilization=100)]) is None
    assert idle_gpu([dict(index=1,total_mib=32607,used_mib=300,utilization=0)])['index']==1
    log('CPU_SELF_TEST_PASSED: both heads and adapters receive gradients; frozen base unchanged; checkpoint restoration exact; busy GPU rejected')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--fold',type=int,default=1,choices=range(1,6))
    parser.add_argument('--max-length',type=int,default=384)
    parser.add_argument('--eval-batch-size',type=int,default=8)
    parser.add_argument('--wait-hours',type=float,default=24)
    parser.add_argument('--physical-gpu',type=int)
    parser.add_argument('--pilot-only',action='store_true')
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(4)
    if args.self_test:
        self_test();return
    if args.out is None:
        parser.error('--out is required for training')
    out=args.out
    out.mkdir(parents=True,exist_ok=args.physical_gpu is not None)
    signal.signal(signal.SIGTERM,handle_signal)
    signal.signal(signal.SIGINT,handle_signal)
    history=[]
    try:
        host=pd.read_csv(args.root/'data/official/train.csv')
        pool_path=args.root/'experiments/public_pool_strategy_pilot_20261113_v1/screened_public_pool.parquet'
        pool=pd.read_parquet(pool_path)
        pool['prompt_id']=pool.prompt_key.astype(str)
        labels=sorted(host.model.unique())
        assert set(pool.model)==set(labels)
        ti,vi=list(GroupKFold(5).split(host,host.model,host.prompt_id))[args.fold-1]
        assert not set(host.iloc[ti].prompt_id)&set(host.iloc[vi].prompt_id)
        reference_fold=np.zeros(len(host),dtype=np.int8)
        for number,(_,indices) in enumerate(GroupKFold(5).split(host,host.model,host.prompt_id),1): reference_fold[indices]=number
        assert np.all(reference_fold[vi]==args.fold)
        host[['id','prompt_id','model']].iloc[vi].to_csv(out/'validation_manifest.csv',index=False)
        write_json(out/'run_manifest.json',dict(strategy='two-stage truncated Qwen3-8B multi-depth LoRA',
                   model_revision=REVISION,pool_rows=len(pool),pool_sha256=hashlib.sha256(pool_path.read_bytes()).hexdigest(),
                   host_sha256=hashlib.sha256((args.root/'data/official/train.csv').read_bytes()).hexdigest(),
                   fold=args.fold,train_rows=len(ti),validation_rows=len(vi),
                   pool_audit=json.loads((pool_path.parent/'data_audit.json').read_text()),
                   score_is_single_fold_not_oof=True,reference='https://www.kaggle.com/code/nishantdahal/two-stage-lora-on-the-public-pool'))
        if args.physical_gpu is None:
            selected=wait_for_resources(out,args.root,args.wait_hours)
            environment=dict(os.environ,CUDA_VISIBLE_DEVICES=str(selected))
            os.execve(sys.executable,[sys.executable,'-u',str(Path(__file__).resolve()),
                      *sys.argv[1:],'--physical-gpu',str(selected)],environment)
        selected=args.physical_gpu
        if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(selected):
            raise RuntimeError('Worker GPU visibility does not match selected physical GPU')
        if idle_gpu([r for r in gpu_snapshot() if r['index']==selected]) is None:
            raise RuntimeError('GPU became busy during reservation; stopped before loading')
        torch.cuda.set_device(0)
        torch.cuda.set_per_process_memory_fraction(.8,0)
        random.seed(20261113);np.random.seed(20261113);torch.manual_seed(20261113)
        base=args.root/'models/Qwen3-8B-Base'
        tokenizer=AutoTokenizer.from_pretrained(base,local_files_only=True)
        tokenizer.padding_side='right'
        if tokenizer.pad_token_id is None: tokenizer.pad_token=tokenizer.eos_token
        external=TokenRows(pool,tokenizer,labels,args.max_length)
        official_train=TokenRows(host.iloc[ti],tokenizer,labels,args.max_length)
        validation=TokenRows(host.iloc[vi],tokenizer,labels,args.max_length)
        config=AutoConfig.from_pretrained(base,local_files_only=True)
        config.num_hidden_layers=24;config.use_cache=False
        config.pad_token_id=tokenizer.pad_token_id
        status(out,'loading_and_preflight',physical_gpu=selected,gpu_training_started=False)
        backbone=AutoModel.from_pretrained(base,config=config,dtype=torch.bfloat16,
                    attn_implementation='sdpa',local_files_only=True,low_cpu_mem_usage=True)
        model=MultiDepthClassifier(backbone,DEPTHS,len(labels))
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        model.to('cuda:0')
        memory_preflight(model,args,out)
        log(f'MODEL_READY physical_gpu={selected} parameters={sum(p.numel() for p in model.parameters())} trainable={sum(p.numel() for p in model.parameters() if p.requires_grad)}')
        fit_stage(model,external,validation,tokenizer,labels,args,out,'stage1',1,history)
        public_state=load_file(str(out/'stage1_epoch1/trainables.safetensors'))
        best=fit_stage(model,official_train,validation,tokenizer,labels,args,out,'stage2',2,history)
        pilot_final=history[-1]['val_ba']
        if args.pilot_only or args.fold!=1 or pilot_final<.46:
            status(out,'completed',physical_gpu=selected,gpu_training_started=True,pilot_best_val_ba=best,
                       pilot_final_val_ba=pilot_final,pilot_goal=.46,pilot_goal_met=pilot_final>=.46,single_fold_exploratory=True)
            log('PILOT_COMPLETED '+str(pilot_final))
            return
        log('PILOT_GATE_PASSED; continuing fixed-two-epoch fivefold confirmation')
        oof=np.zeros((len(host),len(labels)),dtype=np.float32)
        fold_ids=np.zeros(len(host),dtype=np.int8)
        oof[vi]=np.load(out/'stage2_epoch2_validation.npz')['probabilities']
        fold_ids[vi]=1
        fold_scores=[pilot_final]
        for fold,(train_indices,valid_indices) in enumerate(GroupKFold(5).split(host,host.model,host.prompt_id),1):
            if fold==1: continue
            cancelled(out)
            assert not set(host.iloc[train_indices].prompt_id)&set(host.iloc[valid_indices].prompt_id)
            assert np.all(reference_fold[valid_indices]==fold)
            args.fold=fold
            torch.manual_seed(20261113+fold)
            restore_state(model,public_state)
            fold_out=out/f'fold{fold}'
            fold_out.mkdir()
            host[['id','prompt_id','model']].iloc[valid_indices].to_csv(fold_out/'validation_manifest.csv',index=False)
            train_ds=TokenRows(host.iloc[train_indices],tokenizer,labels,args.max_length)
            valid_ds=TokenRows(host.iloc[valid_indices],tokenizer,labels,args.max_length)
            fit_stage(model,train_ds,valid_ds,tokenizer,labels,args,fold_out,'stage2',2,history,run_out=out)
            oof[valid_indices]=np.load(fold_out/'stage2_epoch2_validation.npz')['probabilities']
            fold_ids[valid_indices]=fold
            fold_scores.append(history[-1]['val_ba'])
        assert (fold_ids>0).all() and np.allclose(oof.sum(1),1,atol=1e-4)
        predicted=np.asarray(labels)[oof.argmax(1)]
        score=float(balanced_accuracy_score(host.model,predicted))
        np.savez_compressed(out/'qwen3_8b_two_stage_oof.npz',probabilities=oof,labels=host.model.to_numpy(),
                            label_names=np.asarray(labels),fold=fold_ids,ids=host.id.astype(str).to_numpy())
        write_json(out/'cv_metrics.json',dict(oof_balanced_accuracy=score,fold_scores=fold_scores,
                   epochs_per_fold=2,epoch_selection='fixed in advance, not best-of-validation',
                   source_data_caveats=json.loads((pool_path.parent/'data_audit.json').read_text())['limitation']))
        write_json(out/'cv_classification_report.json',classification_report(host.model,predicted,
                   labels=labels,output_dict=True,zero_division=0))
        log('FIVEFOLD_COMPLETED '+str(score))
        final_checkpoint=None
        if score>=.46:
            restore_state(model,public_state)
            args.fold=0
            torch.manual_seed(20261113)
            full_out=out/'full_fit'
            full_out.mkdir()
            full=TokenRows(host,tokenizer,labels,args.max_length)
            fit_stage(model,full,None,tokenizer,labels,args,full_out,'final',2,history,run_out=out)
            final_checkpoint=str(full_out/'final_epoch2')
        status(out,'completed',physical_gpu=selected,gpu_training_started=True,pilot_final_val_ba=pilot_final,
                   oof_balanced_accuracy=score,fold_scores=fold_scores,final_checkpoint=final_checkpoint,
                   final_training_scores_are_in_sample=True)
        log('COMPLETED '+str(score))
    except BaseException as exc:
        status(out,'stopped' if isinstance(exc,InterruptedError) else 'failed',error=repr(exc),history=history)
        log('STOPPED '+repr(exc))
        raise


if __name__=='__main__': main()
