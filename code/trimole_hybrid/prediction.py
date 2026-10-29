"""Predict with a fixed, checksum-verified serialized classical model."""
from pathlib import Path
import json
import numpy as np
from .artifacts import file_sha256,checkpoint_manifest
from .composition import transform

def predict_classical(checkpoint,features,task_config,expected_sha256=None):
    import joblib
    path=Path(checkpoint)
    if expected_sha256 is None:
        sidecar=path.with_suffix(path.suffix+'.json')
        if sidecar.is_file():expected_sha256=json.loads(sidecar.read_text())['sha256']
    if not expected_sha256:raise ValueError('Provide the published SHA256 or the sidecar written by the training command.')
    if file_sha256(path)!=expected_sha256:raise ValueError('Checkpoint SHA256 does not match.')
    bundle=joblib.load(path)
    with np.load(features,allow_pickle=False) as data:
        if isinstance(bundle,dict) and bundle.get('format')=='trimole-classical-v1':
            if bundle['kind']=='kernel_rank_mean':
                return np.mean([transform(m['model'].decision_function(data[m['feature']]),'rank_n') for m in bundle['members']],axis=0)
            X=data['X'];preds=[]
            for member in bundle['members']:
                model=member['model'];x=X if member['columns'] is None else X[:,member['columns']]
                preds.append(model.predict_proba(x)[:,1] if bundle['classification'] else model.predict(x))
            return np.mean(preds,axis=0)
        if task_config['task_id']=='cyp3a4_substrate_carbonmangels' and isinstance(bundle,dict):
            members=task_config['training_parameters']['members']
            return np.mean([transform(bundle[c['id']].decision_function(data[c['feature']]),'rank_n') for c in members],axis=0)
    raise ValueError('Unsupported checkpoint format. Use the matching model implementation listed in the task configuration.')

def predict_head(checkpoint,features,record):
    """Predict from prepared C/K/E embeddings with a bound feature-head model."""
    import torch
    from trimole.models.model import MultiModalFusionMLP
    if file_sha256(checkpoint)!=record['sha256']:raise ValueError('Checkpoint SHA256 does not match.')
    meta=record['model_metadata'];cfg=meta['config'];dims=meta['dims']
    classification=meta['task_type']=='classification'
    model=MultiModalFusionMLP(dim_smiles=int(dims['chemberta']),dim_3d=int(dims['ept']),dim_graph=int(dims['kpgt']),out_dim=2 if classification else 1,hidden_dim=int(cfg['hidden_dim']),dropout_proj=float(cfg['dropout_proj']),dropout_head=float(cfg['dropout_head']),fusion_type=str(cfg['fusion_type']),task_context_dim=int(cfg.get('task_context_dim',3)))
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True);model.eval()
    masks={'all':None,'chemberta':(True,False,False),'kpgt':(False,True,False),'ept':(False,False,True),'chemberta_kpgt':(True,True,False),'ept_kpgt':(False,True,True),'chemberta_ept':(True,False,True)}
    mask=masks[cfg['modalities']]
    with np.load(features,allow_pickle=False) as z:
        # E_normalized explicitly denotes the fitted training normalization,
        # so query-batch statistics are never substituted for training stats.
        arrays=[np.asarray(z[n],dtype=np.float32) for n in ['C','E_normalized','K']]
    if len({len(x) for x in arrays})!=1 or any(x.ndim!=2 or not np.isfinite(x).all() for x in arrays):raise ValueError('Embedding blocks must be finite, aligned matrices.')
    for x,key in zip(arrays,['chemberta','ept','kpgt']):
        if x.shape[1]!=dims[key]:raise ValueError('Feature width differs from the fitted '+key+' width.')
    context=meta.get('task_context_vector');context=None if context is None else torch.tensor(context,dtype=torch.float32)
    output=[]
    with torch.inference_mode():
        for start in range(0,len(arrays[0]),256):
            c,e,k=[torch.from_numpy(a[start:start+256]) for a in arrays]
            logits=model(c,e,k,modality_mask=mask,task_context=context)
            result=torch.softmax(logits,dim=1)[:,1] if classification else logits.flatten()
            output.append(result.cpu().numpy().astype(float))
    result=np.concatenate(output)
    if not classification:result=result*float(meta['label_std'])+float(meta['label_mean'])
    return result
