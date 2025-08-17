# Visualizing Causal Chain of Heat Stress: A Three-Layer Network Analysis of Climate, Energy, and Socioeconomic Systems in South Asia

熱ストレスがもたらす因果連鎖の可視化：南アジアにおける気候・エネルギー・社会経済の三層ネットワーク分析

## 概要

本リポジトリは、ERA5 由来の気候指標（WBGT）と World Bank WDI のエネルギー・社会経済指標を統合し、  
**気候 → エネルギー → 社会経済**の **三層因果ネットワーク**を推定・検証・可視化するためのコードと再現手順を提供します。

- **因果探索**：三層の部分順序をマスク行列で課し、NOTEARS（線形）で DAG を学習  
- **知識整合**：LLM（Mistral 7B / Ollama）で背景知識と照合し、矛盾エッジを除去  
- **介入評価**：do 演算に基づく全因果効果（総効果）で政策介入点を特定  
- **成果物**：国別・プールのエッジリスト、中心性・クラスター係数、ヒートマップ・ネットワーク図

対象：**南アジア 5 か国（India, Bangladesh, Pakistan, Nepal, Sri Lanka）**、**2000–2021 年**  
（欠損のため **Bhutan** と **2022–2023 年**は主分析から除外）

---

## セットアップ

### 1) Python 環境
- 推奨：**Python 3.10 以上**  
- 依存関係は `requirements.txt` を参照  
```bash
  pip install -r requirements.txt
```
注：NOTEARS は GitHub 実装を `notears/linear.py` に同梱し、
**mask（部分順序制約）** に対応するよう最小限の拡張を加えています（呼び出し例は `04_*` 参照）。

### 2) ERA5 の取得（CDS API）

`~/.cdsapirc` もしくは `.env` に CDS API キーを設定
```ini
url: https://cds.climate.copernicus.eu/api/v2
key: <UID>:<API_KEY>
```

- 2m 気温・露点は 別リクエストで取得し、Magnus 近似 → RH → Stull(2011) 近似で湿球温度 Tw →
WBGT ≈ 0.7·Tw + 0.3·T（日射なし近似）を採用
- 国別集計は国境ポリゴンでクリップし、面積加重平均で年平均値を算出

### 3) WDI の取得
- 5 か国の 2000–2023 年を取得し、以下の 10 指標で三層化
  - 気候層：mean_wbgt
  - エネルギー層：renewable_energy_pct, fossil_fuel_pct, electricity_per_capita, co2_per_capita
  - 社会経済層：gdp_per_capita, unemployment_rate, health_expenditure_pct, agri_valueadded_pct, urbanization_pct

### 4) 前処理
- 年次でマージ → Z スコア標準化
- 欠損は前後年の線形補間 + 端点の ffill/bfill
- 欠損多の Bhutan と 2022–2023 年は除外（主分析は 2000–2021）

---

## 再現手順（最短ルート）

1. `01_download_era5.ipynb`：ERA5 ダウンロード → RH → WBGT → 国別年平均 CSV
2. `02_build_wdi.ipynb`：WDI 指標の取得・整形
3. `03_merge_and_standardize.ipynb`：パネル結合・標準化・補間
4. `04_causal_discovery_notears.ipynb`：マスク付き NOTEARS で 𝑊 推定、しきい値で疎化、エッジ出力
5. `05_llm_validation.ipynb`：Ollama(Mistral) で背景知識照合 → 矛盾エッジ除去・再構築
6. `06_do_interventions.ipynb`：𝑇=(𝐼−𝑊)^{−1}−𝐼を計算、介入効果のヒートマップ
7. `07_visualizations.ipynb`：国別ネットワーク図・指標比較図の生成

---

## 主要アウトプット
- `data/processed/panel/edges_notears_masked.csv`：三層制約下の推定エッジ（プール）
- `data/processed/panel/edges_per_country.csv`：国別エッジ
- `data/processed/panel/edges_no_conflicts*.csv`：LLM 整合性検証後のエッジ
- `results/`：ネットワーク図、中心性・クラスター係数、介入効果ヒートマップ

---

## 既知の注意点
- 線形・同時点 DAG：年平均により極端現象の影響は希釈されます。ラグ・非線形は今後拡張。
- 符号の不均質性：国により同一エッジの符号が異なるため、政策含意は国別解釈が必要。
- LLM 照合：回答のばらつき対策でプロンプトを固定化し、判定と理由をログ化しています。

---

## 引用（Citation）
Odaka, M. (2025). **Visualizing Causal Chain of Heat Stress: A Three-Layer Network Analysis of Climate, Energy, and Socioeconomic Systems in South Asia.** *Society for Environmental Economics and Policy Studies Annual Meeting 2025 (Tokyo, Japan)*.

---

## ライセンス / 連絡先
- License: MIT
- Contact: Mitsuhiro Odaka / harutosyura[at]i[dot]softbank[dot]jp

---

## データソース
- ERA5 © ECMWF/Copernicus Climate Change Service (C3S)
- World Development Indicators (World Bank)
