"""Substantive read-only core audit plus reproducible stretch/report checks."""
import importlib.metadata
import json
import re
import subprocess
import sys
from pathlib import Path
from dotenv import dotenv_values
from packaging.requirements import Requirement
from experiment_common import ROOT, ARTIFACTS, STRETCH_PLAN_CHECKPOINT, load_locked_data, read_csv, sha256
from validate_experiment import validate
from analyze_calibration import build as build_calibration
from analyze_errors import build as build_errors

CORE = 'a55b3ec73bb33b22314c82f8ebf7aa220c977d30'


def git(*args):
    return subprocess.check_output(['git',*args],cwd=ROOT)


def main():
    core=validate()
    assert core['passed']
    protected=git('ls-tree','-r','--name-only',CORE,'data','artifacts','app.py','requirements.txt').decode().splitlines()
    for name in protected:
        old=git('show',f'{CORE}:{name}')
        new=(ROOT/name).read_bytes()
        assert old.replace(b'\r\n',b'\n')==new.replace(b'\r\n',b'\n'), name
    assert git('diff-tree','--no-commit-id','--name-only','-r',STRETCH_PLAN_CHECKPOINT).decode().splitlines()==['planning.md']
    subprocess.run(['git','merge-base','--is-ancestor',STRETCH_PLAN_CHECKPOINT,'HEAD'],cwd=ROOT,check=True)
    stretch_paths=['classifier_app.py','scripts/analyze_calibration.py','scripts/analyze_errors.py',
                   'src/takemeter/classifier.py','artifacts/calibration_results.json','artifacts/error_analysis.json']
    assert not git('ls-tree','-r','--name-only',STRETCH_PLAN_CHECKPOINT,*stretch_paths).strip()
    calibration=json.loads((ARTIFACTS/'calibration_results.json').read_text())
    assert calibration==build_calibration()
    analysis,rows=build_errors()
    assert analysis==json.loads((ARTIFACTS/'error_analysis.json').read_text(encoding='utf-8'))
    exported=read_csv(ARTIFACTS/'error_analysis.csv')
    assert len(exported)==43
    for a,b in zip(rows,exported):
        assert all(str(v)==b[k] for k,v in a.items() if k!='probabilities')
    smoke=json.loads((ARTIFACTS/'classifier_smoke_test.json').read_text())
    assert {r['device'] for r in smoke['records']}=={'cpu','cuda:0'}
    assert smoke['groq_module_imported'] is False and smoke['groq_key_in_environment'] is False
    assert all(abs(sum(r['probabilities'].values())-1)<1e-5 for r in smoke['records'])
    readme=(ROOT/'README.md').read_text(encoding='utf-8')
    baseline=json.loads((ARTIFACTS/'baseline_results.json').read_text())
    final=json.loads((ARTIFACTS/'finetuned_results.json').read_text())
    for result in [baseline,final]:
        assert f"{result['accuracy']:.2%}" in readme and f"{result['macro_f1']:.4f}" in readme
    assert f"{calibration['ece']:.4f}" in readme and f"{calibration['multiclass_brier_score']:.4f}" in readme
    for label in final['label_order']:
        numbers=[baseline['per_class'][label][k] for k in ['precision','recall','f1-score']]
        numbers += [final['per_class'][label][k] for k in ['precision','recall','f1-score']]
        expected='| '+label+' | '+' | '.join(f'{v:.3f}'.lstrip('0') or '0' for v in numbers)+' |'
        # A recall of exactly one is represented explicitly as 1.000.
        assert expected in readme, label
    assert 'To be recorded after final repository audit.' in readme
    assert not re.search(r'https?://(?:www\.)?(?:youtu\.be|youtube\.com|loom\.com)',readme)
    for case in json.loads((ROOT/'data/difficult_cases.json').read_text())['cases']:
        assert case['text'] in readme and case['comment_id'] in readme
    for case in analysis['representative_errors']:
        assert case['text'] in readme and not case['correct']
    for link in re.findall(r'\]\(([^)]+)\)',readme):
        if not link.startswith(('http://','https://','#')) and link!='artifacts/repository_audit.md':
            assert (ROOT/link).exists(), link
    versions={}
    for line in (ROOT/'requirements.txt').read_text().splitlines():
        if not line.strip() or line.startswith('#'): continue
        requirement=Requirement(line)
        version=importlib.metadata.version(requirement.name)
        assert requirement.specifier.contains(version), (requirement.name,version)
        versions[requirement.name]=version
    assert not git('ls-files','--','.env').strip()
    subprocess.run(['git','check-ignore','-q','.env'],cwd=ROOT,check=True)
    subprocess.run(['git','check-ignore','-q','checkpoints/takemeter-distilbert/model.safetensors'],cwd=ROOT,check=True)
    paths=set(git('ls-files').decode().splitlines()+git('ls-files','--others','--exclude-standard').decode().splitlines())
    secret=dotenv_values(ROOT/'.env').get('GROQ_API_KEY','')
    for name in paths:
        data=(ROOT/name).read_bytes()
        assert not secret or secret.encode() not in data, 'Secret check failed'
        assert not re.search(rb'gsk_[A-Za-z0-9_-]{20,}',data), 'Secret pattern check failed'
        assert len(data)<5_000_000, 'Unexpected large tracked/staged candidate'
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],
                         cwd=ROOT,capture_output=True,text=True,encoding='utf-8')
    output=tests.stdout+'\n'+tests.stderr
    if tests.returncode:
        print(output)
        raise RuntimeError('Test suite failed')
    count=int(re.search(r'Ran (\d+) tests',output).group(1))
    splits=load_locked_data()
    threads={name:{r['thread_id'] for r in values} for name,values in splits.items()}
    result={'status':'non-demo checks passed with documented limitations',
            'planning_checkpoint':STRETCH_PLAN_CHECKPOINT,'planning_only_commit_verified':True,
            'stretch_files_absent_at_planning_commit':True,'core_checkpoint':CORE,
            'unchanged_core_files':protected,'tests_passed':count,'tests_failed':0,
            'calibration_reproduced':True,'error_analysis_reproduced':True,
            'core_validation_passed':True,'readme_metric_case_and_link_checks_passed':True,
            'secret_scan_passed':True,'model_weights_ignored':True,'package_versions':versions,
            'final_unique_threads':len(set.union(*threads.values())),
            'shared_train_test_threads':len(threads['train']&threads['test']),
            'limitations':['Poor classifier; all registered success thresholds failed.',
                'Only 43 test comments; calibration bins sparse.',
                'Comment-level, not thread-held-out split; 19 shared train/test thread IDs.',
                'Question-like accepted comment gxiw1ug is a sampling-frame tension; label preserved.',
                'AI-assisted labels, high acceptance and model overlap may introduce anchoring.',
                'Fine-tuned weights local/ignored; no hosted checkpoint or deployed public URL.',
                'Qualitative interpretations are AI-assisted, not independently student-verified.',
                'Demo recording remains a student task.'],
            'remaining_student_action':'Record a 3-5 minute demo; no demo link or completion claimed.'}
    (ARTIFACTS/'repository_audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    report=f"""# Non-demo repository audit

Audited against the assignment checklist supplied by the student. Status:
**non-demo checks passed with documented limitations**. The demo is pending.
No test-set rerun, retraining, relabeling, or split replacement was performed.

| Area | Substantive evidence and verdict |
|---|---|
| Planning | Original community rationale, four definitions, two examples per class, explicit hard boundary, sampling/balance targets, metric rationale, success thresholds and AI plan remain intact. Planning-only commit `{STRETCH_PLAN_CHECKPOINT}` precedes implementation; new stretch paths are absent from that tree. |
| Dataset | 282 reviewed examples; counts 89/50/83/60; unique IDs/normalized text, provenance, and finalized projection verified by core checks. All data files unchanged from the experiment checkpoint. No class dominates. Three real post-audit difficult cases, two corrections, initial-versus-audit timing disclosed. |
| Collection limitation | Original retrieval/deduplication summaries and supplemental archived collection documented. Question-like accepted `gxiw1ug` remains a sampling-frame tension. No human label was changed during this audit. |
| Baseline | Actual Scout 404 evidence preserved; explicit authorized gpt-oss-20b substitution disclosed. Request hashes reconstruct prompt+text, with no truth metadata; all 43 correct locked IDs, zero unresolved parse failures. Saved metrics recomputed. |
| Training | Recorded pretrained revision, seed/config, actual CUDA environment, 39 optimization steps, train-only optimization IDs, validation-only selection and model hashes verified. No retraining in this run. |
| Evaluation | Both saved prediction files contain identical 43 IDs. Accuracy, all class metrics, macro-F1 and confusion matrices independently recomputed; probability rows valid. README metrics match artifacts; poor result and failed thresholds explicit. |
| Calibration | Reproduced 10-bin ECE and multiclass Brier; bin endpoints, empty bins and known synthetic cases tested. Reliability plot visually inspected. No calibration transform or test tuning. |
| Error analysis | All 29 errors reproduced; prediction/confusion counts, lengths, confidence, paired-model results and declared lexical/structural measurements. Three exact errors and all four probabilities included. Interpretations distinguish observations from hypotheses. |
| Local interface | Separate classifier_app.py preserves annotation app byte-for-byte. Real CPU/CUDA offline smoke inference; Streamlit submit/empty/missing-model tests passed. Four probabilities, stable labels, truncation notice, poor-result disclosure and recovery instructions present. |
| README | Covers dataset/annotation/audit history, methods, observed comparisons, actual examples, learned-vs-intended reflection, calibration, stretch scope, setup, AI disclosure, limits and pending demo. Local links checked. No invented video URL. |
| Reproducibility | Pinned installed requirements checked. Setup's stale empty-project wording fixed. EXPERIMENTS explains fixed revision and guarded reproduction. Cache/weights ignored; local checkpoint required for actual inference and full model-hash audit. |
| Integrity | {len(protected)} pre-existing core data/artifact/app/requirements files unchanged. Planning validator now verifies only the authorized append-only plan while retaining the historical original-plan identity. No metrics or predictions altered. |
| Security | Exact-key and key-pattern scan over tracked/untracked deliverable candidates passed; .env untracked/ignored; model weights ignored; no oversized deliverable files. |
| Tests | **{count} tests passed**, zero failures, including actual local Streamlit inference. Core artifact validation also passed. |
| Remaining deliverable | Student records the 3–5 minute demo; no recording or upload performed. |

Known limitations remain visible in README. In particular, this split is disjoint
by comment ID, not by thread: **19 thread IDs overlap train and test**. This limits
independence but does not change recorded optimization membership. Small-sample
calibration, AI-label anchoring, local-only weights, and the poor model result are
not silently presented as resolved. These are study limitations, not evidence of
a computational bug invalidating the saved metrics.

Final staging/commit/push and clean-tree checks are performed after this audit.
Their observed status is reported to the student; this document does not preclaim
a future commit hash or successful push.
"""
    (ARTIFACTS/'repository_audit.md').write_text(report,encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['status','tests_passed','core_validation_passed','calibration_reproduced','error_analysis_reproduced']},indent=2))


if __name__=='__main__': main()
