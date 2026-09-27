"""Local Streamlit classifier, separate from the annotation application."""
import os
import sys
from pathlib import Path
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from takemeter.classifier import DEFAULT_MODEL, classify_text, load_classifier


@st.cache_resource
def cached_model(path):
    return load_classifier(path)


def main():
    st.set_page_config(page_title='TakeMeter classifier', page_icon='🖨️')
    st.title('TakeMeter')
    st.write('What kind of contribution does this 3D-printing comment make?')
    st.caption('Local fine-tuned DistilBERT · Experimental course classifier')
    st.warning('This model achieved 32.56% accuracy on 43 test comments. Treat its predictions as fallible suggestions.')
    with st.expander('The four labels'):
        st.markdown('**reasoned_explanation:** supports a conclusion with meaningful reasoning.\n\n'
                    '**actionable_advice:** offers a concrete action without substantial rationale.\n\n'
                    '**bare_claim:** states an unsupported answer or viewpoint without a concrete next step.\n\n'
                    '**reaction:** primarily a social or emotional response.')
    model_path = os.getenv('TAKEMETER_MODEL_DIR', str(DEFAULT_MODEL))
    if not (Path(model_path) / 'model.safetensors').exists():
        st.info('Checkpoint not found. Restore the saved checkpoints/takemeter-distilbert folder, '
                'or follow the pinned-revision reproduction instructions in EXPERIMENTS.md. '
                'Weights are intentionally excluded from Git; no hosted download is provided.')
    with st.form('classify'):
        text = st.text_area('Comment', height=160, placeholder='Paste or write a comment…', max_chars=20000)
        submitted = st.form_submit_button('Classify', type='primary')
    if submitted:
        if not text.strip():
            st.warning('Enter a comment before classifying.')
            return
        try:
            with st.spinner('Classifying locally…'):
                result = classify_text(text, cached_model(model_path))
        except (FileNotFoundError, OSError, ValueError, RuntimeError) as error:
            st.error(f'Unable to classify ({type(error).__name__}). Check the saved model folder and its label mapping. '
                     'See EXPERIMENTS.md for checkpoint recovery and setup instructions.')
            return
        st.subheader(result['predicted_label'])
        st.metric('Raw confidence', f"{result['confidence']:.2%}")
        st.caption('Softmax scores, not calibrated guarantees. Inference runs locally without Groq.')
        st.dataframe([{'Label':label, 'Probability':f'{p:.2%}'} for label,p in result['probabilities'].items()],
                     hide_index=True, width='stretch')
        if result['truncated']:
            st.info('Long comment: only the first 256 tokens were used.')
        st.caption('Device: '+result['device'])


if __name__ == '__main__':
    main()
