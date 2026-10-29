"""Fit a supplied fixed configuration; this module has no configuration search."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import numpy as np

def _save(path,bundle,config):
    import joblib
    path=Path(path)
    if path.exists():raise FileExistsError('Choose a new checkpoint output path.')
    path.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump(bundle,path)
    record=dict(task=config['task_id'],format='trimole-classical-v1',sha256=hashlib.sha256(path.read_bytes()).hexdigest(),fixed_configuration=config)
    path.with_suffix(path.suffix+'.json').write_text(json.dumps(record,indent=2)+'\n')

def _kernel_models(config,data,seed):
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.svm import SVC
    members=[]
    for c in config['training_parameters']['members']:
        X=np.asarray(data[c['feature']],dtype=float)
        pre=[SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler()]
        if c['preprocess']=='pca64':pre.append(PCA(n_components=64,svd_solver='randomized',random_state=seed))
        width=64 if c['preprocess']=='pca64' else X.shape[1]
        model=make_pipeline(*pre,SVC(C=c['C'],kernel='rbf',gamma=c['gamma_factor']/width,class_weight=c['class_weight'],random_state=seed))
        model.fit(X,np.asarray(data['y']))
        members.append(dict(model=model,feature=c['feature'],response='decision_function'))
    return dict(format='trimole-classical-v1',kind='kernel_rank_mean',members=members)

def train(config,features,output,seed,run_index):
    if features is None:raise ValueError('--features must supply fixed training features and labels.')
    from .fixed_backends import build_scaffold_folds,fit_selector,fit_backend,fit_backend_xl
    with np.load(features,allow_pickle=False) as data:
        if config['task_id']=='cyp3a4_substrate_carbonmangels':
            _save(output,_kernel_models(config,data,seed),config);return
        if 'X' not in data or 'y' not in data:raise ValueError('Training NPZ requires X and y.')
        X=np.asarray(data['X'],dtype=np.float32);y=np.asarray(data['y'])
        if X.ndim!=2 or y.ndim!=1 or len(X)!=len(y) or not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError('Provide aligned, finite features on the recorded feature basis.')
        task=config['task_id'];p=config.get('training_parameters',{})
        classification=config['metric'] in ['AUROC','AUPRC']
        if task=='bbb_martins':
            if seed not in p['random_seeds']:raise ValueError('Choose a seed listed in the task record.')
            from sklearn.ensemble import ExtraTreesClassifier
            model=ExtraTreesClassifier(n_estimators=800,min_samples_leaf=1,max_features='sqrt',class_weight='balanced_subsample',random_state=seed,n_jobs=8)
            model.fit(X,y);bundle=dict(format='trimole-classical-v1',kind='fold_mean',classification=True,members=[dict(model=model,columns=None)])
        elif task=='ames' or p.get('backend') in ['extratrees','xgb']:
            if 'smiles' not in data:raise ValueError('Scaffold fold training requires the original aligned SMILES array.')
            smiles=np.asarray(data['smiles']).astype(str).tolist();models=[]
            if task=='ames':
                from xgboost import XGBClassifier
                if not 1<=run_index<=5:raise ValueError('--run-index must be 1..5 for the five recorded AMES groups.')
                inner=p['seed_groups'][run_index-1]
            else:inner=[seed]
            for fold_seed in inner:
                for fold,valid in enumerate(build_scaffold_folds(smiles,3,fold_seed)):
                    train_mask=np.ones(len(X),dtype=bool);train_mask[valid]=False
                    if task=='ames':
                        pos=float(np.sum(y[train_mask]==1));neg=float(train_mask.sum()-pos)
                        columns=np.arange(X.shape[1])
                        model=XGBClassifier(n_estimators=500,max_depth=5,learning_rate=.04,subsample=.85,colsample_bytree=.85,min_child_weight=2,objective='binary:logistic',eval_metric='logloss',tree_method='hist',device='cpu',random_state=fold_seed*100+fold,n_jobs=1,scale_pos_weight=max(1.,neg/max(pos,1.)))
                        model.fit(X[train_mask],y[train_mask],verbose=False)
                    else:
                        k=0 if str(p['top_k']).lower()=='all' else int(p['top_k'])
                        kind='classification' if classification else 'regression'
                        columns=fit_selector(X[train_mask],y[train_mask],kind,k)
                        counts=p['source_default_estimator_counts']
                        args=SimpleNamespace(xgb_estimators=counts['xgb'],tree_estimators=counts['extratrees'],cat_estimators=counts['catboost'],n_jobs=1)
                        fit=fit_backend_xl if p['backend_implementation']=='XL' else fit_backend
                        model=fit(p['backend'],X[train_mask][:,columns],y[train_mask],kind,config['metric'],seed+fold,args)
                    models.append(dict(model=model,columns=columns))
            bundle=dict(format='trimole-classical-v1',kind='fold_mean',classification=classification,members=models)
        else:
            raise ValueError('This task combines independently fitted components. Use the recorded component metadata and neural training entry point; predict.py --component-predictions applies its final weights.')
        _save(output,bundle,config)

def train_neural(config,component,seed,data_dir,output):
    """Use the original feature-head trainer with one recorded per-run config."""
    from dataclasses import fields
    from trimole.training.trainer import TrainConfig,fit_on_task
    matches=[r for c in config['components_bound'] if c['id']==component for r in c.get('fitted_runs',[]) if r['seed']==seed]
    if len(matches)!=1:raise ValueError('Choose an explicitly bound component and seed from train.py --task TASK --inspect.')
    values=matches[0]['config'];known={f.name for f in fields(TrainConfig)}
    unknown=set(values)-known
    if unknown:raise ValueError('Recorded trainer configuration needs its original implementation: '+str(sorted(unknown)))
    output=Path(output)
    if output.exists():raise FileExistsError('Choose a new feature-head output directory.')
    return fit_on_task(Path(data_dir),output,TrainConfig(**values))
