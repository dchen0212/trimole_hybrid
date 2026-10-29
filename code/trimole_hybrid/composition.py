"""Apply a fixed recorded combination. No fitting, ranking of models or scoring."""
import numpy as np
from scipy.stats import rankdata

def transform(values,mode):
    x=np.asarray(values,dtype=np.float64)
    if x.ndim!=1 or not len(x) or not np.isfinite(x).all():
        raise ValueError('Each component must be a nonempty finite one-dimensional prediction array.')
    if mode=='raw':return x
    if mode=='zscore':
        x=x.astype(np.float32)
        sd=float(np.std(x))
        return np.zeros_like(x) if sd<1e-12 else ((x-float(np.mean(x)))/sd).astype(np.float32)
    if mode=='rank':
        return rankdata(x,method='average').astype(np.float32)
    if mode=='rank_n':return rankdata(x,method='average')/len(x)
    if mode=='logit':
        x=np.clip(x.astype(np.float32),1e-6,1-1e-6)
        return np.log(x/(1-x)).astype(np.float32)
    raise ValueError('Unknown fixed transformation: '+str(mode))

def combine(parts,weights,mode='raw'):
    if len(parts)!=len(weights) or not parts:raise ValueError('One fixed weight is required per component.')
    arrays=[transform(p,mode) for p in parts]
    if len({len(x) for x in arrays})!=1:raise ValueError('Components must use the same rows and order.')
    w=np.asarray(weights,dtype=float)
    if not np.isfinite(w).all():raise ValueError('Weights must be finite.')
    return sum(float(a)*b for a,b in zip(w,arrays))

def apply_graph(node,predictions):
    if isinstance(node,str):
        if node not in predictions:raise ValueError('Missing component predictions: '+node)
        return transform(predictions[node],'raw')
    return combine([apply_graph(n,predictions) for n in node['inputs']],node['weights'],node['mode'])

def task_prediction(config,predictions):
    if 'combination_graph' in config:return apply_graph(config['combination_graph'],predictions)
    rule=config.get('fixed_combination')
    if rule:
        result=combine([predictions[n] for n in rule['component_order']],rule['weights'],rule['mode'])
        if rule.get('output_transform')=='sigmoid':
            from scipy.special import expit
            result=expit(result)
        return result
    if 'prediction' in predictions:return transform(predictions['prediction'],'raw')
    raise ValueError('Use a recorded fixed combination or provide the complete task-model array as `prediction`.')
