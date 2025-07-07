import streamlit as st
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import torch.nn.functional as F

# モデルとトークナイザーの読み込み（初回のみ）
@st.cache_resource
def load_model():
    model_name = "Tetsuo3003/highrisk_medical_japanese"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()
label_list = model.config.id2label

# ラベルの B- / I- を取り除く
def strip_label_prefix(label):
    return label.replace("B-", "").replace("I-", "")

# ラベルが同じものをマージ（B- / I- 無視）
def merge_same_label_entities(tokens, labels, scores):
    merged = []
    current_tokens = []
    current_label = None
    current_scores = []

    for token, label, score in zip(tokens, labels, scores):
        if token in tokenizer.all_special_tokens:
            continue
        token_stripped = token.lstrip("▁")
        clean_label = strip_label_prefix(label)

        if clean_label == "O":
            if current_tokens:
                word = "".join(current_tokens)
                avg_score = sum(current_scores) / len(current_scores)
                merged.append((word, current_label, avg_score))
                current_tokens = []
                current_scores = []
                current_label = None
            continue

        if clean_label == current_label:
            current_tokens.append(token_stripped)
            current_scores.append(score)
        else:
            if current_tokens:
                word = "".join(current_tokens)
                avg_score = sum(current_scores) / len(current_scores)
                merged.append((word, current_label, avg_score))
            current_tokens = [token_stripped]
            current_scores = [score]
            current_label = clean_label

    if current_tokens:
        word = "".join(current_tokens)
        avg_score = sum(current_scores) / len(current_scores)
        merged.append((word, current_label, avg_score))
    return merged

# 推論処理
def ner_predict(text):
    encoded = tokenizer(text, return_tensors="pt", truncation=True)
    input_ids = encoded["input_ids"]
    with torch.no_grad():
        logits = model(**encoded).logits

    probs = F.softmax(logits, dim=-1)
    scores, predictions = torch.max(probs, dim=2)

    token_ids = input_ids.squeeze().tolist()
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    predicted_labels = [label_list[pred.item()] for pred in predictions.squeeze()]
    confidence_scores = scores.squeeze().tolist()

    merged = merge_same_label_entities(tokens, predicted_labels, confidence_scores)
    return merged

# --- Streamlit UI ---

st.title("注目語抽出アプリ（highrisk_medical_japanese）")

input_text = st.text_area("解析したい文章を入力してください", height=150)

if st.button("推論する"):
    if not input_text.strip():
        st.warning("文章を入力してください。")
    else:
        with st.spinner("推論中..."):
            results = ner_predict(input_text)

        if results:
            st.markdown("### 抽出結果")
            for word, label, score in results:
                st.markdown(f"- **{word}**  : `{label}`（確信度: `{score:.2f}`）")
        else:
            st.info("注目語は見つかりませんでした。")
