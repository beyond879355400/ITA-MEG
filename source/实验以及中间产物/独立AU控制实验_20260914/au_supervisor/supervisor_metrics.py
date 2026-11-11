"""Raw 0--5 AU diagnostics and a preregistered pilot qualification gate."""
import math
import numpy as np
from scipy.stats import spearmanr

AUS = [1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26]


def supported_aus(labels):
    y = np.asarray(labels, dtype=float)
    return [i for i in range(y.shape[1]) if len(np.unique(y[:, i])) >= 2 and int((y[:, i] > 0).sum()) >= 10]


def metrics(labels, predictions, subjects=None):
    y, p = np.asarray(labels, dtype=float), np.asarray(predictions, dtype=float)
    if y.shape != p.shape or y.ndim != 2 or y.shape[0] == 0 or y.shape[1] != 12 or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError('Invalid labels/predictions')
    rows = []
    for j, au in enumerate(AUS):
        a, b = y[:, j], p[:, j]
        e = np.abs(a-b)
        variable = float(a.std()) > 1e-10 and float(b.std()) > 1e-10
        rows.append({'au': au, 'n': len(a), 'active_n': int((a>0).sum()), 'nonactive_n': int((a==0).sum()),
                     'mae': float(e.mean()), 'active_mae': float(e[a>0].mean()) if (a>0).any() else None,
                     'nonactive_mae': float(e[a==0].mean()) if (a==0).any() else None,
                     'pearson': float(np.corrcoef(a,b)[0,1]) if variable else None,
                     'spearman': float(spearmanr(a,b).statistic) if variable else None,
                     'prediction_std': float(b.std()), 'label_std': float(a.std())})
    active=[r['active_mae'] for r in rows if r['active_mae'] is not None]
    nonactive=[r['nonactive_mae'] for r in rows if r['nonactive_mae'] is not None]
    result = {'raw_intensity_scale': [0,5], 'count':len(y), 'macro_mae':float(np.abs(y-p).mean(axis=0).mean()),
              'active_mae':float(np.mean(active)) if active else None,
              'nonactive_mae':float(np.mean(nonactive)) if nonactive else None,
              'aggregation':'equal AU weight; masks defined by true label >0 and =0', 'per_au':rows}
    if subjects is not None:
        subs = np.asarray(subjects)
        if len(subs) != len(y): raise ValueError('Subject count mismatch')
        result['per_subject'] = {s:metrics(y[subs==s],p[subs==s]) for s in sorted(set(subjects))}
        yc, pc = y.copy(), p.copy()
        for s in sorted(set(subjects)):
            m = subs==s
            yc[m] -= y[m].mean(axis=0)
            pc[m] -= p[m].mean(axis=0)
        result['within_subject_centered_pearson'] = {
            str(au):float(np.corrcoef(yc[:,j],pc[:,j])[0,1]) if yc[:,j].std()>1e-10 and pc[:,j].std()>1e-10 else None
            for j,au in enumerate(AUS)}
    return result


def constant_baselines(train_labels, dev_labels, subjects):
    train = np.asarray(train_labels, dtype=float)
    constants = {'train_mean':train.mean(axis=0), 'train_median':np.median(train,axis=0), 'zero':np.zeros(12)}
    return {name:{'constant_raw':v.tolist(), 'metrics':metrics(dev_labels,np.tile(v,(len(dev_labels),1)),subjects)} for name,v in constants.items()}


def qualify(result, baselines, support, min_improvement=.05):
    strongest_macro = min(v['metrics']['macro_mae'] for v in baselines.values())
    active_values=[v['metrics']['active_mae'] for v in baselines.values() if v['metrics']['active_mae'] is not None]
    strongest_active = min(active_values) if active_values else None
    pearsons = [result['per_au'][i]['pearson'] for i in support]
    positive = sum(x is not None and x>0 for x in pearsons)
    # Undefined correlations count as zero in the gate, not silently excluded.
    median = float(np.median([0 if x is None else x for x in pearsons])) if pearsons else 0.0
    tests = {'macro_improves_5pct':result['macro_mae'] <= strongest_macro*(1-min_improvement),
             'active_improves_5pct':strongest_active is not None and result['active_mae'] is not None and result['active_mae'] <= strongest_active*(1-min_improvement),
             'two_thirds_supported_au_positive':bool(support) and positive >= math.ceil(2*len(support)/3),
             'supported_median_pearson_ge_0_2':median>=.2}
    return {'passed':all(tests.values()), 'checks':tests, 'strongest_constant_macro_mae':strongest_macro,
            'strongest_constant_active_mae':strongest_active, 'supported_au_numbers':[AUS[i] for i in support],
            'positive_supported_au_count':positive, 'supported_median_pearson':median,
            'interpretation':'predeclared engineering pilot gate, not statistical significance or final test evidence'}
