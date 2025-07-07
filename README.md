# 注目語抽出（医療リスク）Streamlitアプリ

日本語の医療・リスク関連文書から、重要語句（注目語）を抽出する NER（名前付きエンティティ認識）モデルを使った Streamlit アプリです。

本アプリは、Hugging Face 上に公開したファインチューニング済みモデル [`tetsuo-ishikawa/highrisk_medical_japanese`](https://huggingface.co/Tetsuo3003/highrisk_medical_japanese) を使用して、文章中の「high_risk」「risk」「hazard」「state」といった注目語を抽出・可視化します。

---

## 🔍 アプリの概要

- **入力**：日本語の文章
- **出力**：抽出された注目語とそのラベル（色付きでハイライト表示）
- **モデル**：`tsmatz/xlm-roberta-ner-japanese` をベースにファインチューニング
- **Streamlit Cloud対応**：ボタン一つでWebアプリとして公開可能

---

## 🚀 デモ（Streamlit Cloud）

👉 [アプリを開く](https://ftrobertaner002-mbqwgh9r4aahvd8zwpg66v.streamlit.app/) ← デモURL（※必要に応じて差し替えてください）

---

## 💻 ローカルでの実行方法

### 1. リポジトリをクローン

```bash
git clone https://github.com/yourname/ner-risk-streamlit-app.git
cd ner-risk-streamlit-app
