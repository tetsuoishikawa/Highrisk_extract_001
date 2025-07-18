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
## 📄 ライセンス

このリポジトリは [MITライセンス](https://opensource.org/licenses/MIT) に基づいて提供されます。  
著作権表示およびライセンス文は [`LICENSE`](./LICENSE) ファイルに記載されています。  
商用・非商用問わず自由に使用・改変・再配布が可能です。

## 📚 使用モデル
モデル名：Tetsuo3003/highrisk_medical_japanese

ベースモデル：tsmatz/xlm-roberta-ner-japanese

タスク：日本語 NER（トークン分類）

抽出ラベル：

high_risk：非常に危険、非常に悪い結果、一刻も早く注目しなければならない危害や損害とその影響

risk：危険、悪い結果、人やモノなどに対する危害や損害とその影響

hazard：危険の原因になり得る「モノ」

state：hazard が risk や high_risk に変わる「条件」、「状況・状態」、「きっかけ」、「原因」


モデルサイズ：約1.1GB

## 🙋‍♂️ 作者
名前：Tetsuo Ishikawa

Hugging Face：Tetsuo3003

## 🛑 注意事項
本アプリおよびモデルは、研究・開発・教育用途での利用を想定しています。

実際の医療診断や治療などの 臨床目的での直接使用は推奨されません。

抽出された語句に基づく判断・対応は、必ず専門の医療従事者によって行ってください。
