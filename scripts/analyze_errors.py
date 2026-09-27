"""Reproduce descriptive error diagnostics from frozen saved predictions."""
import csv
import json
import re
from collections import Counter
import numpy as np
from experiment_common import ARTIFACTS, LABELS, load_locked_data, read_csv, sha256

TECHNICAL_PATTERN=r'\b(print\w*|filament\w*|nozzle\w*|layer\w*|bed|slicer\w*|petg|pla|tpu|support\w*)\b'
INTERPRETATIONS={
 'pbz923l': 'The comment supplies a procedure: adjust z offset during a large first layer. It does not develop why the adjustment solves a problem. Mistaking a multi-step technical procedure for supported reasoning is consistent with this error, but no causal feature attribution was performed.',
 'pc1nuul': 'The brief orientation assertion supplies neither a mechanism nor a concrete next step. Predicting an explanation for this six-word claim shows that the dominant class extends to short unsupported assertions; technical vocabulary or length alone cannot establish the cause.',
 'gyxhfex': 'The Mongolia association and smiling emoji express a social reaction rather than a technical claim. Its bare_claim prediction illustrates a missed social function. This interpretation follows the text and taxonomy, not a causal explanation of model internals.'
}


def stats(values):
    if not values: return {'count':0,'mean':None,'median':None,'min':None,'max':None}
    return {'count':len(values),'mean':float(np.mean(values)),'median':float(np.median(values)),
            'min':float(min(values)),'max':float(max(values))}


def build():
    splits=load_locked_data()
    lookup={r['comment_id']:r for group in splits.values() for r in group}
    fine=read_csv(ARTIFACTS/'finetuned_predictions.csv')
    baseline=read_csv(ARTIFACTS/'baseline_predictions.csv')
    assert [r['comment_id'] for r in fine]==[r['comment_id'] for r in splits['test']]
    assert [r['comment_id'] for r in baseline]==[r['comment_id'] for r in fine]
    base={r['comment_id']:r for r in baseline}
    joined=[]
    for p in fine:
        source=lookup[p['comment_id']]
        assert p['true_label']==source['final_label']
        joined.append({**p,'confidence':float(p['confidence']),
                       'probabilities':{label:float(p['probability_'+label]) for label in LABELS},
                       'text':source['text'],'source_url':source['source_url'],
                       'word_count':len(source['text'].split()),'character_count':len(source['text']),
                       'correct':p['true_label']==p['predicted_label'],
                       'baseline_prediction':base[p['comment_id']]['predicted_label'],
                       'baseline_correct':base[p['comment_id']]['predicted_label']==p['true_label']})
    errors=[r for r in joined if not r['correct']]
    assert {r['comment_id'] for r in errors}=={r['comment_id'] for r in read_csv(ARTIFACTS/'finetuned_errors.csv')}
    paired={name:[r['comment_id'] for r in joined if predicate(r)] for name,predicate in {
        'both_correct':lambda r:r['correct'] and r['baseline_correct'],
        'baseline_only_correct':lambda r:not r['correct'] and r['baseline_correct'],
        'finetuned_only_correct':lambda r:r['correct'] and not r['baseline_correct'],
        'both_wrong':lambda r:not r['correct'] and not r['baseline_correct']}.items()}
    features={
        'technical_lexicon':lambda t:bool(re.search(TECHNICAL_PATTERN,t,re.I)),
        'question_mark':lambda t:'?' in t,'multiline':lambda t:'\n' in t,
        'at_most_20_words':lambda t:len(t.split())<=20}
    feature_stats={}
    for name,predicate in features.items():
        feature_stats[name]={}
        for flag in [True,False]:
            group=[r for r in joined if predicate(r['text'])==flag]
            feature_stats[name][str(flag).lower()]={'count':len(group),
                'correct':sum(r['correct'] for r in group),
                'reasoned_predictions':sum(r['predicted_label']=='reasoned_explanation' for r in group)}
    pairs=Counter((r['true_label'],r['predicted_label']) for r in errors)
    examples=[]
    for comment_id,interpretation in INTERPRETATIONS.items():
        example=next(r for r in errors if r['comment_id']==comment_id)
        examples.append({**example,'interpretation':interpretation,
                         'interpretation_status':'AI-assisted hypothesis checked against saved text and taxonomy; not independently student-verified'})
    training=json.loads((ARTIFACTS/'training_results.json').read_text())
    result={
        'count':43,'total_finetuned_errors':len(errors),
        'prediction_distribution':{name:{label:sum(r['predicted_label']==label for r in rows) for label in LABELS}
                                   for name,rows in [('baseline',baseline),('finetuned',fine)]},
        'errors_by_true_label':dict(Counter(r['true_label'] for r in errors)),
        'errors_by_predicted_label':dict(Counter(r['predicted_label'] for r in errors)),
        'confusion_pairs':[{'true_label':a,'predicted_label':b,'count':n} for (a,b),n in pairs.most_common()],
        'confidence_overall':stats([r['confidence'] for r in joined]),
        'by_correctness':{str(c).lower():{'words':stats([r['word_count'] for r in joined if r['correct']==c]),
             'characters':stats([r['character_count'] for r in joined if r['correct']==c]),
             'confidence':stats([r['confidence'] for r in joined if r['correct']==c])} for c in [True,False]},
        'test_length_by_true_class':{label:{'words':stats([r['word_count'] for r in joined if r['true_label']==label]),
                    'characters':stats([r['character_count'] for r in joined if r['true_label']==label])} for label in LABELS},
        'dataset_words_by_split_and_class':{name:{label:stats([len(r['text'].split()) for r in rows if r['final_label']==label])
                    for label in LABELS} for name,rows in splits.items()},
        'highest_confidence_errors':sorted(errors,key=lambda r:r['confidence'],reverse=True)[:5],
        'paired_correctness_ids':paired,'paired_correctness_counts':{k:len(v) for k,v in paired.items()},
        'prediction_agreement':sum(r['predicted_label']==r['baseline_prediction'] for r in joined),
        'prediction_disagreement':sum(r['predicted_label']!=r['baseline_prediction'] for r in joined),
        'prediction_pair_counts':[{'baseline':a,'finetuned':b,'count':n}
              for (a,b),n in Counter((r['baseline_prediction'],r['predicted_label']) for r in joined).most_common()],
        'feature_definitions':{'technical_lexicon_regex':TECHNICAL_PATTERN,'word_count':'whitespace split; descriptive, not tokenizer length'},
        'feature_statistics':feature_stats,'representative_errors':examples,
        'training_losses':[{'epoch':h['epoch'],'train_loss':h['train_loss'],'validation_loss':h['validation_loss']} for h in training['history']],
        'interpretation_limits':['Exploratory post-test descriptions, not causal attribution.',
            'Correctness/length association is confounded by true class and prediction collapse.',
            '197 training examples and class imbalance may limit learning; no causal ablation was run.',
            'Only 39 optimization steps occurred; decreasing loss does not demonstrate semantic separation.',
            'Technical lexicon is shared across functions, but dominant predictions also occur without those words.'],
        'source_sha256':{name:sha256(ARTIFACTS/name) for name in ['finetuned_predictions.csv','baseline_predictions.csv','finetuned_errors.csv','training_results.json']}
    }
    return result,joined


if __name__=='__main__':
    result,rows=build()
    (ARTIFACTS/'error_analysis.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    with (ARTIFACTS/'error_analysis.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=[k for k in rows[0] if k!='probabilities'])
        writer.writeheader(); writer.writerows({k:v for k,v in row.items() if k!='probabilities'} for row in rows)
    print(json.dumps({k:result[k] for k in ['total_finetuned_errors','prediction_distribution','paired_correctness_counts']},indent=2))
