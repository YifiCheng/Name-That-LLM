import os
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ[key] = '4'
import gc
import json
import re
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits

ROOT=Path('/storage/users/cyf/Name_That_LLM')
OUT=ROOT/'experiments/public_pool_strategy_pilot_20261113_v1'
WORDS=set(('a an the and or but if while although because as so than that which who whom whose this these those '
           'it its they them their he him his she her hers we us our you your i me my in on at to from with without '
           'for of by over under between through into about is are was were be been being have has had do does '
           'did can could will would shall should may might must not no nor all any some each every both either '
           'neither other another such only also just very too more most less least here there when where why '
           'how what whether then now still yet even much many few little').split())

def log(s): print(time.strftime('%Y-%m-%d %H:%M:%S'),s,flush=True)
def canonical(s): return re.sub(r'\s+',' ',str(s)).strip().casefold()
def view(t,name):
    if name=='shape': return re.sub(r'[^\W\d_]+',lambda m:'X' if m[0].isupper() else 'x',t)
    if name=='function':
        return ' '.join(w.lower() if w.lower() in WORDS or not w.isalnum() else 'CONTENT' for w in re.findall(r'\w+|[^\w\s]',t))
    return t

def main():
    OUT.mkdir(parents=True,exist_ok=False)
    host=pd.read_csv(ROOT/'data/official/train.csv')
    pool=pd.read_parquet(ROOT/'data/external/two_stage_public/pool_v1.parquet')
    extra=pd.read_csv(ROOT/'data/external/extended_public/passages_clean.csv',low_memory=False)
    audit={'pool_input':len(pool),'extended_input':len(extra),'extended_train_regen_excluded':int(extra['split'].eq('train_regen').sum())}
    known_regen=set(extra.loc[extra['split'].eq('train_regen'),'text'].map(canonical))
    extra=extra.loc[extra['split'].isin(['new','proxy_v1','new2'])].copy()
    extra['prompt_key']='prvsiyan:'+extra.prompt_id.str.replace('v1_','',regex=False)
    extra['source']='prvsiyan_clean_20261111'
    combined=pd.concat([pool[['text','model','prompt_key','source']],extra[['text','model','prompt_key','source']]],ignore_index=True)
    normalized=combined.text.map(canonical)
    excluded=normalized.isin(set(host.text.map(canonical))|known_regen)|combined.prompt_key.str.contains(r'tr_p\d+|train_regen|reconstruct',regex=True,case=False,na=False)
    audit['known_overlap_or_regen_removed']=int(excluded.sum())
    combined=combined.loc[~excluded & combined.model.isin(host.model.unique())].copy()
    combined['normalized']=combined.text.map(canonical)
    conflict=combined.groupby('normalized').model.nunique()
    combined=combined.loc[~combined.normalized.isin(conflict[conflict>1].index)].drop_duplicates('normalized').reset_index(drop=True)
    audit['after_exact_dedup']=len(combined)
    av=TfidfVectorizer(stop_words='english',ngram_range=(1,2),min_df=2,max_features=40000,sublinear_tf=True,dtype=np.float32)
    ax=av.fit_transform(host.text.tolist()+combined.text.tolist())
    hc,hn=pd.factorize(host.prompt_id)
    pc,pn=pd.factorize(combined.prompt_key)
    hg=normalize(csr_matrix((np.ones(len(host)),(hc,np.arange(len(host)))),shape=(len(hn),len(host)))@ax[:len(host)])
    pg=normalize(csr_matrix((np.ones(len(combined)),(pc,np.arange(len(combined)))),shape=(len(pn),len(combined)))@ax[len(host):])
    similarity=(pg@hg.T).toarray().max(1)
    removed_keys=set(pn[similarity>=0.45])
    audit.update({'possible_prompt_overlap_threshold':0.45,'possible_overlap_groups_removed':len(removed_keys),
                  'possible_overlap_rows_removed':int(combined.prompt_key.isin(removed_keys).sum()),
                  'limitation':'Lexical overlap screening cannot establish semantic prompt independence. Public pool preparation also used official-train typography filters. Results are exploratory, not independent confirmation.'})
    combined=combined.loc[~combined.prompt_key.isin(removed_keys)].copy()
    audit['remaining_rows']=len(combined)
    audit['class_counts']=combined.model.value_counts().to_dict()
    (OUT/'data_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    combined.drop(columns='normalized').to_parquet(OUT/'screened_public_pool.parquet',index=False)
    log('AUDIT '+json.dumps(audit))

if __name__=='__main__': main()
