"""Exercise real offline CPU/CUDA inference on new smoke text, not test data."""
import json
import os
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from takemeter.classifier import load_classifier, classify_text, choose_device

if __name__=='__main__':
    os.environ.pop('GROQ_API_KEY',None)
    examples=['Try printing a small calibration cube before the next large part.',
              'That little model looks wonderful!']
    records=[]
    for device in dict.fromkeys(['cpu',choose_device()]):
        bundle=load_classifier(device=device)
        for text in examples:
            result=classify_text(text,bundle)
            assert abs(sum(result['probabilities'].values())-1)<1e-5
            records.append({'input':text,**result})
    result={'purpose':'Actual inference smoke test on newly authored input; not graded evaluation',
            'groq_key_in_environment':False,'groq_module_imported':'groq' in sys.modules,
            'records':records}
    (ROOT/'artifacts/classifier_smoke_test.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))
