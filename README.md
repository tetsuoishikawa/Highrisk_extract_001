# 注目語抽出（医療リスク）Streamlitアプリ

日本語の医療・リスク関連文書から、重要語句（注目語）を抽出する NER（名前付きエンティティ認識）モデルを使った Streamlit アプリです。

このアプリでは、Hugging Face に公開されたファインチューニング済みモデル [`Tetsuo3003/highrisk_medical_japanese`](https://huggingface.co/Tetsuo3003/highrisk_medical_japanese) を使い、文章中の「high_risk」「risk」「hazard」「state」などの語句を可視化抽出します。

---

## 🔍 アプリの概要

- **入力**：日本語の任意文章
- **出力**：注目語＋ラベル（色付きで強調表示）
- **使用モデル**：`tsmatz/xlm-roberta-ner-japanese` をベースにファインチューニング
- **構築環境**：Streamlit + Hugging Face Transformers

---

## 🚀 デモ（Streamlit Cloud）

👉 [アプリを開く](https://ftrobertaner002-mbqwgh9r4aahvd8zwpg66v.streamlit.app/)

---

## 💻 ローカル実行手順

### 1. リポジトリをクローン

```bash
git clone https://github.com/yourname/ner-risk-streamlit-app.git
cd ner-risk-streamlit-app
ファイル構成
text
コピーする
編集する
ner-risk-streamlit-app/
├── app.py                # Streamlit アプリ本体
├── requirements.txt      # 依存ライブラリ一覧
└── README.md             # この説明ファイル

📚 使用モデル
モデル名：Tetsuo3003/highrisk_medical_japanese

ベースモデル：tsmatz/xlm-roberta-ner-japanese

タスク：日本語 NER（トークン分類）

抽出ラベル：

high_risk（重大なリスク）

risk（一般的なリスク）

hazard（危険性）

state（身体状態・兆候）

モデルサイズ：約1.1GB

📄 ライセンス
このリポジトリは MITライセンス に基づいて提供されます。
商用・非商用問わず自由に使用・改変・再配布が可能ですが、著作権表示とライセンス文を含める必要があります。

🙋‍♂️ 作者
名前：Tetsuo Ishikawa

Hugging Face：Tetsuo3003
