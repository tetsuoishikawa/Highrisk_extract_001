import streamlit as st
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

@st.cache_resource
def load_model():
    model = AutoModelForTokenClassification.from_pretrained("Tetsuo3003/highrisk_medical_japanese")
    tokenizer = AutoTokenizer.from_pretrained("Tetsuo3003/highrisk_medical_japanese")
    nlp = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
    return nlp

def highlight_entities(text, entities):
    highlighted_text = ""
    last_idx = 0
    for ent in entities:
        start = ent['start']
        end = ent['end']
        label = ent['entity_group']
        score = ent['score']
        # 前のテキスト
        highlighted_text += text[last_idx:start]
        # 強調表示
        highlighted_text += f'<mark style="background-color: #f9dc5c;">{text[start:end]} <sub>{label}</sub></mark>'
        last_idx = end
    highlighted_text += text[last_idx:]
    return highlighted_text

st.title("ハイリスク抽出アプリ")
text = st.text_area("文章を入力してください：", height=200)

if text:
    with st.spinner("モデルが予測中..."):
        nlp = load_model()
        entities = nlp(text)
        st.subheader("抽出結果")
        st.markdown(highlight_entities(text, entities), unsafe_allow_html=True)
