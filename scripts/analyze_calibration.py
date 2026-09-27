"""Descriptive calibration analysis; never fits or changes the saved model."""
import json
import numpy as np
from experiment_common import ARTIFACTS, LABELS, read_csv, load_locked_data, sha256


def calibration(probabilities, targets, bins=10):
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(targets, dtype=int)
    if p.ndim != 2 or p.shape != (len(y),4) or not len(y):
        raise ValueError('Expected nonempty N x 4 probabilities')
    if not np.isfinite(p).all() or (p<0).any() or (p>1).any() or not np.allclose(p.sum(1),1,atol=1e-6):
        raise ValueError('Invalid probabilities')
    if ((y<0)|(y>3)).any():
        raise ValueError('Invalid targets')
    confidence = p.max(1)
    correct = p.argmax(1) == y
    edges = np.array([i/bins for i in range(bins+1)])
    indices = np.minimum(np.searchsorted(edges,confidence,side='right')-1,bins-1)
    rows=[]
    ece=0.0
    for i in range(bins):
        mask=indices==i
        n=int(mask.sum())
        acc=float(correct[mask].mean()) if n else None
        conf=float(confidence[mask].mean()) if n else None
        contribution=n/len(y)*abs(acc-conf) if n else 0.0
        ece+=contribution
        rows.append({'lower':float(edges[i]),'upper':float(edges[i+1]),'count':n,
                     'accuracy':acc,'mean_confidence':conf,
                     'confidence_minus_accuracy':conf-acc if n else None,
                     'ece_contribution':contribution})
    return {'count':len(y),'method':'10 equal-width [lower,upper) bins; final bin includes 1',
            'bins':rows,'ece':ece,'mean_confidence':float(confidence.mean()),
            'accuracy':float(correct.mean()),
            'mean_confidence_minus_accuracy':float(confidence.mean()-correct.mean()),
            'multiclass_brier_score':float(((p-np.eye(4)[y])**2).sum(1).mean()),
            'brier_definition':'Mean sum of four squared errors; not divided by four',
            'model_recalibrated':False}


def build():
    splits=load_locked_data()
    rows=read_csv(ARTIFACTS/'finetuned_predictions.csv')
    assert [r['comment_id'] for r in rows]==[r['comment_id'] for r in splits['test']]
    result=calibration([[float(r['probability_'+label]) for label in LABELS] for r in rows],
                       [LABELS.index(r['true_label']) for r in rows])
    result.update(source_sha256=sha256(ARTIFACTS/'finetuned_predictions.csv'),
                  label_order=LABELS, limitation='43 test cases; sparse bins; descriptive, not proof of population calibration')
    return result


def plot(result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,(ax,counts)=plt.subplots(2,1,figsize=(8,7),height_ratios=[3,1],sharex=True)
    populated=[b for b in result['bins'] if b['count']]
    ax.plot([0,1],[0,1],linestyle='--',color='gray',label='Perfect calibration')
    ax.plot([b['mean_confidence'] for b in populated],[b['accuracy'] for b in populated],
            'o-',color='#176b9b',label='Observed nonempty bins')
    for b in populated:
        ax.annotate('n='+str(b['count']),(b['mean_confidence'],b['accuracy']),
                    xytext=(7,8),textcoords='offset points')
    ax.set(ylabel='Observed accuracy',ylim=(-.03,1.07),
           title=f"DistilBERT reliability on 43 locked test comments\nECE = {result['ece']:.4f}; Brier = {result['multiclass_brier_score']:.4f}")
    ax.legend(loc='upper left')
    ax.grid(alpha=.2)
    counts.bar([(b['lower']+b['upper'])/2 for b in result['bins']],
               [b['count'] for b in result['bins']],width=.09,color='#176b9b')
    counts.set(xlabel='Confidence (maximum class probability)',ylabel='Count',xlim=(0,1))
    fig.tight_layout()
    fig.savefig(ARTIFACTS/'calibration_curve.png',dpi=160)
    plt.close(fig)


if __name__=='__main__':
    result=build()
    (ARTIFACTS/'calibration_results.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    plot(result)
    print(json.dumps(result,indent=2))
