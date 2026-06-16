# マイルストーン1 報告：既存SGG出力に基づく意味情報飽和判定の基礎検討

> 観測品質の改善（解像度の段階的回復）に伴い、既存Scene Graph Generation（EGTR）の出力から
> **意味情報がどこで飽和するか**を推定できるかを検討したPoCの報告。

---

## 0. 要約（先に結論）

- **解像度が上がるほどノード・関係情報は増える**（NodeRecall 0.112→0.569、TripletRecall 0.000→0.286、SGScore 0.056→0.427、L1→L20）。単調増加を確認。Bilinear補間を採用し、Nearest と比べて低解像度での回復が早い（例: L10 NodeRecall 0.341 vs Nearest 0.122）。
- **飽和は解像度レンジの中〜後半に分布**：TopKオラクル（coef 0.9）の平均停止 L16.36（中央値 17）、71/100 が L20 前に飽和、L5–20 に広がる。Bilinear補間の採用により飽和点が解像度レンジ内に留まる（Nearest では L18–20 に天井集中）。
- **ノード確信度だけでは飽和判定に不十分**：EGTRの平均物体確信度は約0.40で頭打ちし、絶対閾値（0.5以上）はほぼ到達不能。確信度のみの baseline（MAE 5.40）は提案手法に大きく劣後。
- **上位2手法は実質タイ**：TopKオラクル（coef 0.9）基準で、1位 llm_sgg_4b_h5_k2（MAE **2.99**）と2位 hybrid_4b_k2（提案手法、MAE **3.06**）はΔMAE 0.07・MedAE 共に 2.0 で**実質的に並ぶ**。提案手法（hybrid）は物体検出確信度とLLM関係性信頼度の2モジュール融合だが、本動作点ではLLMモジュール単体で同等性能を示す。確信度モジュールの追加効果はLLM閾値が低い場合にのみ顕在化する（llm=0.7 では hybrid 3.06 ≫ LLM単体 3.82、llm=0.8 で差が消失）。ただし **per-image 精度は限定的**（Pearson r ≈ 0.19–0.25、手法は中心傾向を捉えるが正確な per-image 予測は困難）。
- **構造化SGG＋LLMが画像VLMを上回る**：同一モデル `qwen3.5` で比較し、SGG系上位手法（llm_sgg_4b_h5_k2 MAE 2.99、hybrid_4b_k2 MAE 3.06）が最良VLM vlm_9b_single_k2（MAE 3.63）を上回る。4b VLM（MAE 6.22–9.66）は大きく劣後。
- **Bilinear補間を採用**（光学的忠実性）。Nearest 結果は比較として併記。

---

## 1. はじめに

### 1.1 背景

家庭等の人と共生する実環境において、ロボットは人・物の状況を理解して適切に行動する必要がある。効率的な理解には、対象に接近する等の追加観測が求められるが、既に十分に観測済みで接近しても情報が増えない場合もある。ロボットは現在の観測から「追加観測で新たな情報が得られるか」を判断する能力を持つべきである。

### 1.2 目的

本研究は、シーングラフ生成（SGG）において、観測品質に応じた追加観測の要否を判断するプロセスの基礎検討を行う。追加観測の要否は2つの観点から判断する：(1) 今見えているものに対する確信の程度、(2) 関係性まで十分に捉えられているか・未観測の物体がありそうか。

### 1.3 アプローチ

上記2つの観点に対応する2つのモジュールで判断を行う：

- **物体検出確信度**：SGGが検出した物体に対する確信度。画像品質が低い段階では確信度が低く不安定で、品質向上に伴い安定する。
- **LLMによる関係性信頼度**：生成されたSG（物体・関係の系列）をLLMに与え、常識知識に基づいて「関係が十分か」「未出現の物体・関係がありそうか」を飽和スコアとして出力させる。

### 1.4 実験設定

実際のロボット接近の代わりに、観測品質の改善を画像解像度の段階的回復で再現する。20段階で劣化・復元し、低解像度を遠距離の観測、原解像度を最接近時の観測とみなすことで、ロボットの接近過程を近似する。この設定のもとで意味情報の飽和を定量化し、どの段階で観測を停止できるかを基礎的に検討する。

- **データセット**：Visual Genome（VG150, test split）、100画像（GT物体≥5、GT関係≥3、seed=42のランダムサンプル）
- **SGG手法**：EGTR（CVPR2024、Deformable-DETRベース、finetune済みckpt）
- **補間法**：Bilinear（scale 0.05→1.00 の20段階、縮小→bilinear補間再拡大）
- **LLM**：ollama `qwen3.5:4b` / `qwen3.5:9b`（画像は見せず、構造化SG系列のみ入力）

> **スコープ明記**：接近行動そのものではなく、接近に伴う観測品質改善のうち**解像度改善のみ**を扱う（視点変化・遮蔽・移動計画は対象外）。LLMは画像理解ではなく、**SGG出力系列の増分・安定性の評価**に用いる。

---

## 2. 関連研究

Object Goal Navigation を中心に、実世界での環境探索を行う研究群がある。特に UniGoal は、質問から対象のシーングラフを生成し、探索しながら環境のシーングラフを構築して、質問SGと環境SGが合致した時点で探索を終了する手法である。

しかし、既存手法は「意味情報の飽和による観測停止」を明示的に扱わない。UniGoal の停止条件は質問SGとのマッチングであり、「十分に観測したので追加情報は得られない」という判断とは異なる。本研究は、SGGの出力から意味情報の飽和を推定し、観測の停止を判断するという、これらの手法が扱わない問題を設定する。

---

## 3. 問題設定

### 3.1 観測列の生成

20段階（Level 1: scale=0.05 の劣化画像 〜 Level 20: scale=1.00 の原画像）の観測列を生成する。各段階でEGTR推論を行い、ノード集合（検出物体）とトリプレット集合（subject-predicate-object の関係三つ組）を取得する。

### 3.2 擬似真値の定義

アノテーション真値（VG-GT）とSGG出力には大きな乖離がある（EGTR の TripletRecall@L20 = 0.286）。このため、**Level 20 のSGG出力の高確信コア（top-10 ノード / top-5 トリプレット）を擬似真値**として採用する。

**top-10 ノード / top-5 トリプレットの根拠（2つの独立なアンカーが収束）**：
- **VG150 アノテーション規模**：GT ユニークカテゴリ中央値 8（6–10）、ユニークトリプレット中央値 5（3–6）→ 1シーン分の意味コンテンツに相当。
- **EGTR 確信度の崖**：物体確信度は rank 10 でまだ ~0.36 だが、triplet\_score は rank 5 以降 ~0.001 に崩壊 → 確信度の高いコアは ~10 ノード / ~5 トリプレット。
- **非対称 N が正しい**：関係は物体より疎（全物体ペアが関係を持つわけではない）。対称の 10/10 ではトリプレット top-10 ≈ 全トリプレット（L20平均~8本）となり "コア" 選択の意味が消失。

平均参照集合サイズ：ノード 5.8 ラベル、トリプレット 3.5 本。

### 3.3 評価指標：SGScore

Level 20 未満の各レベルの集合と擬似真値から、以下の指標で飽和度を測定する：

`SGScore(t) = 0.5 * NodeRecall(t) + 0.5 * TripletRecall(t)`

ここで NodeRecall / TripletRecall はそれぞれ擬似真値（top-10/5）に対する recall（label-only カテゴリ集合一致）。RelSGScore(L20) = 1.0 by construction（参照 ⊂ L20）。

### 3.4 理想停止位置

**理想停止位置 = SGScore ≥ 0.9 を最初に達成するレベル**（TopK RelOracle, coef 0.9）。

**coef 0.9 を主基準とする根拠**：参照集合が L20 の高確信コア（上位N）であるため、該当項目は早いレベルで確実に回収され RelSGScore が高くなりやすい。低い閾値（0.8）では飽和が早く立ちすぎる。0.9 は「擬似真値のほぼ全回収」を狙う値であり、これより高い閾値（≥ 0.95）は量子化により 1.0 と等価となり 43/100 が L20 に集中して分布が崩壊する。0.9 は分布を保つ最も厳しい閾値である。

coef 0.8 は感度分析として併記する。

### 3.5 SGScore vs SGDet@20：主指標の選択根拠

**SGScore（label-only カテゴリ集合一致）を主指標に採用し、SGDet@20 を per-image 飽和指標として使用しない**。理由：

1. **IoU は飽和タイミングに新情報を足さない**：per-image 停止レベルを label-only と SGDet@20 で比較したところ、71/96 画像で一致。bbox 精度はラベル一致と同時に飽和する（独立の飽和シグナルを持たない）。
2. **per-image 標準 SGDet@20 は退化**：100画像中 31画像が R@20(L20)=0（フル解像度でも strict IoU+class+predicate マッチングで1件もヒットしない）。これを飽和基準にすると「飽和しない」が標準的な結果になり評価として機能しない。

**標準 SGDet R@20 はコーパスレベルの EGTR 妥当性検証値としてのみ報告**：本実験100画像で 0.186（cf. 自前500画像実測 20.4、著者評価 23.54）。

---

## 4. 提案手法

### 4.1 物体検出確信度モジュール

EGTRの検出結果のうち確信度 0.3 以上のもののみを抽出し、その確信度の平均をスコアとする。

### 4.2 LLMによる関係性信頼度モジュール

SGGのノード・トリプレットをLLMに入力し、常識知識に基づいて未観測物体の有無・関係性の十分さを判断させ、飽和スコア（saturation\_score）を出力させる。直近5段階（history\_length=5）のSG系列を入力する。

### 4.3 提案手法（hybrid）：2モジュールの融合

上記2つのモジュールの出力の平均（0.5 * 物体検出確信度 + 0.5 * LLM関係性信頼度）を計算し、閾値を**2回連続**で超えた時点で停止する。

### 4.4 閾値体系

3つの閾値は独立に定めたものではなく、一貫した凸結合の関係にある：

| モジュール / 手法 | 停止閾値 | 根拠 |
|---|---|---|
| 物体検出確信度（ablation = baseline） | **0.4** | EGTR の平均物体確信度が ~0.40 でプラトーに達する水準（≥0.5 は 7/100、≥0.6 は 1/100 でしか到達しない）。プラトー水準そのものを閾値とする。 |
| LLM 関係性信頼度（ablation = llm\_sgg） | **0.8** | 高確信での飽和判定。経験的にも最良の動作点。 |
| **hybrid = mean(conf, LLM)（提案手法）** | **0.6** | **導出値: 0.5 * 0.4 + 0.5 * 0.8 = 0.6** — 2つの成分閾値の 0.5/0.5 凸結合であり、融合重みと整合する。2回連続超過（k=2）で停止。 |

`configs/config.yaml` 対応：`baseline.mean_object_conf_threshold=0.4`、`llm.saturation_threshold=0.8`、`ablation.hybrid.threshold=0.6`。

### 4.5 アブレーション構成

- **物体検出確信度単体**（baseline）：確信度 ≥ 0.4 で停止（k=1）。
- **LLM関係性信頼度単体**（llm\_sgg）：LLM飽和スコア ≥ 0.8 で停止（k=2）。
- **提案手法（hybrid）**：2モジュール融合 ≥ 0.6 で停止（k=2）。

### 4.6 比較参考：VLMベースライン

同じ `qwen3.5`（vision対応）に**画像を直接**見せて飽和判定（single/sequence）。SG結果を使わないため提案手法との直接比較は不可であり、参考として位置づける。VLMがEGTRより高精度の可能性があり、最適停止位置が異なりうる点に留意。

---

## 5. 結果

### 5.1 Q1. 解像度改善でノード・関係情報は増えるか → はい

平均（n=100, bilinear, L1→L20）：NodeRecall 0.112→0.569、TripletRecall 0.000→0.286、SGScore 0.056→0.427。単調増加を確認。

代表的なレベル別推移（100画像平均、bilinear）：

| Level | scale | NodeRecall | TripletRecall | SGScore | mean\_obj\_conf |
|-------|-------|-----------|---------------|---------|---------------|
| 1  | 0.050 | 0.112 | 0.000 | 0.056 | 0.166 |
| 5  | 0.094 | 0.161 | 0.021 | 0.091 | 0.247 |
| 10 | 0.242 | 0.341 | 0.105 | 0.223 | 0.356 |
| 15 | 0.455 | 0.494 | 0.199 | 0.347 | 0.391 |
| 18 | 0.729 | 0.547 | 0.240 | 0.393 | 0.399 |
| 20 | 1.000 | 0.569 | 0.286 | 0.427 | 0.405 |

L20 は原画像そのものであり補間法に依存しない。Bilinear は低解像度帯（L1–L15）で Nearest より回復が早い（§5.5 で比較）。

### 5.2 飽和の分布（TopK RelOracle）

**TopK RelOracle 閾値別（n=100, bilinear）**：

| coef | 平均停止 | 中央値 | sat<20 | 分布 |
|------|----------|--------|--------|------|
| **0.9（主基準）** | **16.36** | **17** | 71/100 | L5–20 に展開、peak L15（14画像）、L20 に29画像 |
| 0.8（感度） | 15.01 | 14.5 | 80/100 | L11–20 に展開、peak L14（17画像）、L20 に20画像 |

飽和は解像度レンジの中〜後半に分布しており、観測品質の改善過程で意味情報の飽和を検出する余地がある。

参考 — GT-based Oracle（絶対基準）の退化：

| θ | 真に飽和 |
|---|----------|
| 0.4 | 63/100 |
| 0.5 | 40/100 |
| 0.6 | 21/100 |

絶対閾値では多数の画像が「永遠に飽和しない」退化した評価になるため、TopK RelOracle を採用した。

### 5.3 停止手法の比較（主結果）

TopK RelOracle（coef 0.9）基準の停止誤差（n=100, bilinear, MAE昇順, 全13手法）：

| Rank | 手法 | 平均停止 | MAE | MedAE | Early% | Late% | Pearson r |
|------|------|----------|-----|-------|--------|-------|-----------|
| 1 | **llm\_sgg\_4b\_h5\_k2**（LLM単体） | 16.17 | **2.99** | 2.0 | 0.44 | 0.40 | 0.227 |
| 2 | **hybrid\_4b\_k2**（提案手法） | 16.34 | **3.06** | 2.0 | 0.42 | 0.43 | 0.193 |
| 3 | vlm\_9b\_single\_k2 | 14.41 | 3.63 | 3.0 | 0.64 | 0.26 | 0.176 |
| 4 | hybrid\_9b\_k2 | 15.08 | 3.84 | 3.0 | 0.47 | 0.40 | 0.071 |
| 5 | hybrid\_4b\_k1 | 12.77 | 4.91 | 4.0 | 0.69 | 0.19 | 0.079 |
| 6 | vlm\_9b\_single\_k1 | 11.95 | 5.03 | 4.5 | 0.82 | 0.13 | 0.120 |
| 7 | baseline（conf≥0.4） | 13.38 | 5.40 | 5.0 | 0.58 | 0.26 | 0.133 |
| 8 | vlm\_4b\_single\_k2 | 12.98 | 6.22 | 5.0 | 0.61 | 0.32 | −0.076 |
| 9 | hybrid\_9b\_k1 | 10.76 | 6.26 | 6.0 | 0.82 | 0.13 | 0.173 |
| 10 | llm\_sgg\_9b\_h5\_k2 | 11.19 | 6.33 | 6.0 | 0.72 | 0.18 | 0.089 |
| 11 | llm\_sgg\_4b\_h5\_k1 | 9.95 | 6.93 | 7.0 | 0.84 | 0.09 | −0.065 |
| 12 | llm\_sgg\_9b\_h5\_k1 | 6.95 | 9.43 | 10.0 | 0.96 | 0.01 | 0.034 |
| 13 | vlm\_4b\_single\_k1 | 7.00 | 9.66 | 10.0 | 0.94 | 0.04 | −0.115 |

**主な知見**：

1. **上位2手法は実質タイ**：llm\_sgg\_4b\_h5\_k2（MAE 2.99）と提案 hybrid\_4b\_k2（MAE 3.06）はΔMAE 0.07、MedAE 共に 2.0 で実質的に並ぶ。構造化SGG＋LLM at 4b with consecutive-k2 が上位を占める。
2. **アブレーション解釈（提案手法 = 2モジュール融合）**：LLM関係性信頼度モジュール単体（llm\_sgg\_4b\_k2, MAE 2.99）が提案 hybrid（3.06）と並ぶ。**本動作点（llm=0.8）では、物体検出確信度モジュールの追加効果は限定的**である。確信度モジュールの効果が顕在化するのはLLM閾値を低く設定した場合に限られる：llm=0.7 では hybrid\_4b\_k2（MAE 3.06）が llm\_sgg\_4b\_k2（MAE 3.82）を明確に上回るが、llm=0.8 で差が消失する。
3. **consecutive-k=2 が一貫して有効**：hybrid\_4b k1→k2 で MAE 4.91→3.06、llm\_sgg\_4b で 6.93→2.99。
4. **小モデル(4b)が構造化SGGで優位**：LLM-SGG MAE 4b\_k2=2.99 vs 9b\_k2=6.33。9bは飽和を過信し早期停止。VLMでは逆に 9b > 4b（MAE 3.63 vs 6.22）で大モデルが必要。
5. **baseline（確信度のみ）は7位**（MAE 5.40）。上位6手法が全て上回る。

### 5.4 感度分析（coef 0.8）

TopK RelOracle coef 0.8 基準での上位（n=100, bilinear）：

| Rank | 手法 | MAE | MedAE |
|------|------|-----|-------|
| 1 | llm\_sgg\_4b\_h5\_k2 | 3.26 | 3.0 |
| 2 | hybrid\_4b\_k2 | 3.31 | 3.0 |
| 3 | vlm\_9b\_single\_k2 | 3.34 | 3.0 |
| — | baseline | 4.97 | 4.0 |

coef を 0.8 に下げるとオラクルが早まり（平均停止 15.01 vs 16.36）上位が拮抗するが、上位のタイ関係とランキングファミリは安定している。

### 5.5 Bilinear 補間と Nearest 補間の比較

**補間法の選択根拠**：光学センサによる遠方観測は面積積分的な低域通過（滑らかなボケ）であり、bilinear がこれを忠実に再現する。Nearest-neighbor はポイントサンプリングによるブロック状エイリアシング（デジタルズーム的アーティファクト）を生じるが、これは光学系では発生しない。

Nearest-neighbor 補間のレベル別推移（100画像平均、参考）：

| Level | scale | NodeRecall | TripletRecall | SGScore | mean\_obj\_conf |
|-------|-------|-----------|---------------|---------|---------------|
| 1  | 0.050 | 0.034 | 0.000 | 0.017 | 0.162 |
| 5  | 0.094 | 0.063 | 0.000 | 0.031 | 0.116 |
| 10 | 0.242 | 0.122 | 0.002 | 0.062 | 0.158 |
| 15 | 0.455 | 0.223 | 0.062 | 0.143 | 0.354 |
| 18 | 0.729 | 0.463 | 0.188 | 0.325 | 0.389 |
| 20 | 1.000 | 0.569 | 0.286 | 0.427 | 0.405 |

Bilinear は低解像度帯で Nearest を大幅に上回る（例: L10 NodeRecall 0.341 vs 0.122、SGScore 0.223 vs 0.062）。

**Nearest での TopK 評価は天井集中を示す**：TopK coef 0.8 で平均停止 18.5、中央値 19 と、ほぼ全画像が L18–20 に集中する。飽和点が解像度レンジの上端に張り付くため手法間の停止誤差がほとんど分離されず、評価の弁別力が失われる（天井効果）。**Bilinear を採用する根拠**は、光学的忠実性に加え、飽和が解像度レンジの中〜後半に分布し手法間の差を評価できる点にある。

### 5.6 構造化SGG出力 vs VLM

同一モデル `qwen3.5` で「構造化SGテキスト（画像なし）」vs「生画像」を比較（TopK coef 0.9 基準）。

- 最良 SGG 系: llm\_sgg\_4b\_h5\_k2 MAE **2.99**（1位）、hybrid\_4b\_k2 MAE **3.06**（2位）
- 最良 VLM: vlm\_9b\_single\_k2 MAE **3.63**（3位）
- 能力統制比較（4b）：llm\_sgg\_4b\_h5\_k2 MAE **2.99** / hybrid\_4b\_k2 MAE **3.06** vs vlm\_4b\_single\_k2 MAE **6.22**
- 9b VLM は3位と健闘するが、4b VLM は大きく劣後 → 画像直接入力は大モデルが必要

構造化SGG出力は、モデルサイズの影響を受けにくく、意味情報の飽和判定に安定して適する。ただしVLMはSG結果を使わないため直接比較は不可であり、参考として位置づける。

---

## 6. 評価枠組みの設計検討（方法論的貢献）

### 6.1 TopKオラクルの導入

L20 の高確信コア（top-10 ノード / top-5 トリプレット）を擬似GTとし、ノイジーな裾の完全再現要求を外すことで、飽和が解像度レンジの中〜後半（coef 0.9 で平均 16.36、中央値 17）に分布する評価枠組みを実現。full参照オラクル（擬似GT = L20 全出力）は L20 のノイジーな低確信度出力の完全再現を要求するため飽和が解像度天井近傍に集中しやすい（平均停止 16.71）が、TopK ではコアに絞ることで飽和点がレンジ内に収まり、手法間の停止誤差を評価できる。ただし per-image の弁別は依然として弱い（上位手法でも相関 r ≈ 0.19–0.25）。

### 6.2 RelSGScore が測っているもの

RelSGScore は VG アノテーションではなく Level-20 EGTR 出力を参照グラフとする。測定対象は「意味的正確さ」ではなく「**モデル出力の収束性**（低解像度出力がフル解像度出力にどれだけ近いか）」である。この自己参照設計は研究目的から正当化される：本研究が問うのは「固定の知覚モジュールの出力が、これ以上接近しても変わらなくなるのはいつか」であり、実機ロボットは真の GT を持たず自身のモデル出力のみを参照できるためである。

### 6.3 ollama `num_ctx` 不具合の発見と修正

既定 `num_ctx=4096` が長プロンプト（履歴5×top20/30 で最大~9.2kトークン）を無言で切り詰め、生成枠0で1トークン停止 → LLMの「パース失敗27%」の正体。`num_ctx=12288` で **0%に解消**。修正前は不当にVLM>LLMに見えていた。

### 6.4 関係エクスポートの過剰フィルタの修正

人物ボックスは確信度~0.2で物体閾値0.3に届かず、その関係（man-wearing-shirt 等）が連鎖的に消え、23/100の擬似GTが関係0件になっていた。**EGTRの失敗ではなく出力規約の問題**。本来のSGGの使い方（端点足切りなしのtop-K）に修正し **23→4/100**、真GT triplet recall@L20 0.250→0.286。

---

## 7. EGTRの妥当性（論文整合性）

「triplet recallが低い／述語の大半が当たらない」のは**EGTRの不具合ではなく既知の長尾現象**であることを確認：

- 同梱の著者評価JSONが論文Table 1を再現（R@20/50/100 = 23.54/30.21/34.35、mR@50=7.94、AP50=30.83）。我々の500画像実測（RTX5080）も想定レンジ内（R@20/50/100 = 20.4/26.3/30.5）。
- VG150は述語が極端な長尾（`on` だけで全関係の約40%）。R@50≈30 に対し **mR@50≈7.9**（約4倍差）は全SGG手法共通＝**多くの述語が低recallなのは構造的に想定通り**。
- 我々のPoC指標が低めなのは label-only・IoU無し・n=100・top-K・劣化画像という**指標/設定差**であり、モデル欠陥ではない。
- 本実験100画像でのコーパスレベル SGDet R@20@L20 = 0.186。
- L20 での関係 decouple 統計：zero-rel 4/100、TripletRecall@L20 0.286、NodeRecall@L20 0.569。

---

## 8. 限界

- 100画像・label-only・カテゴリ集合一致（インスタンス区別なし、bbox IoU無し）。
- **per-image 飽和予測精度は限定的**：Pearson r ≈ 0.19–0.25（正だが弱い相関）。手法は中心傾向を捉えるが、各画像の正確な飽和レベル予測は困難。
- **提案 hybrid と LLM 単体が並ぶ**（本動作点で確信度モジュールの追加効果が限定的）。確信度アンカーの有用性は LLM 閾値が低い場合にのみ顕在化する。
- 人物ノード自体は確信度~0.25で物体閾値0.3に届かず node 集合から落ちる（関係トラックは復活済みだが node-level の人物弱点は残存）。
- 観測品質を解像度のみで近似（視点・遮蔽・移動は対象外）。
- TopKオラクルの top-10/5 は VG150 + EGTR に calibrate されており、他データセット・他SGG手法への転移性は未検証。

---

## 9. 成果物

- 指標CSV：`outputs/metrics/{all_metrics, oracle_stop, baseline_stop, relative_metrics, relative_oracle_stop, method_comparison_relative_100, method_comparison_convergence_100}.csv`
- TopKオラクル評価：`outputs/metrics/topk_oracle/{method_comparison_topk_c08, method_comparison_topk_c09, oracle_topk_per_image, summary}.{csv,json,md}`
- 図：`outputs/figures/{relative_method_comparison_mae_100, stop_distribution_all_100, saturation_rate_100, convergence_vs_absolute_mae_100, representative_*}.png`
- 研究ノート：`.claude/docs/research/{m1-analysis, m1-evaluation, m1-relative-saturation, m1-scale100, m1-convergence, egtr-paper-claims, egtr-repro, m1-code-review, bilinear_topk_datasheet}.md`
- コード：`src/milestone1/*`、`scripts/01..17`（実行手順は `MILESTONE1.md`）
- TopKオラクル評価スクリプト：`scripts/eval_topk_oracle.py`
- 補間法別結果アーカイブ：`outputs/runs/{nearest,bilinear}/`（各 metrics/figures/analysis/egtr_predictions を含む）

### 出力 JSON スキーマ

**EGTR 予測 JSON**（`outputs/egtr_predictions/{image_id}/level_{t:02d}.json`）:

| フィールド | 所属 | 説明 |
|-----------|------|------|
| `objects[].label` | object | 物体ラベル（VG150） |
| `objects[].confidence` | object | 物体確信度 |
| `objects[].bbox` | object | `[x1, y1, x2, y2]` |
| `objects[].class_id` | object | VG150 クラスID |
| `relationships[].subject_id` | relationship | subject の query index |
| `relationships[].subject_label` | relationship | subject ラベル |
| `relationships[].subject_class_id` | relationship | subject の VG150 クラスID |
| `relationships[].subject_bbox` | relationship | subject の bbox `[x1, y1, x2, y2]` |
| `relationships[].subject_conf` | relationship | subject の確信度 |
| `relationships[].predicate` | relationship | 述語ラベル |
| `relationships[].predicate_id` | relationship | VG150 述語ID |
| `relationships[].object_id` | relationship | object の query index |
| `relationships[].object_label` | relationship | object ラベル |
| `relationships[].object_class_id` | relationship | object の VG150 クラスID |
| `relationships[].object_bbox` | relationship | object の bbox `[x1, y1, x2, y2]` |
| `relationships[].object_conf` | relationship | object の確信度 |
| `relationships[].confidence` | relationship | 関係確信度（rel\_score） |
| `relationships[].triplet_score` | relationship | rel\_score × subj\_conf × obj\_conf（SGDet R@K ランキング用） |

**GT JSON**（`outputs/gt/{image_id}.json`）:

| フィールド | 所属 | 説明 |
|-----------|------|------|
| `objects[].object_id` | object | objects 内インデックス |
| `objects[].label` | object | 物体ラベル（VG150） |
| `objects[].class_id` | object | VG150 クラスID |
| `objects[].bbox` | object | `[x1, y1, x2, y2]` |
| `relationships[].subject_id` | relationship | objects へのインデックス |
| `relationships[].subject_label` | relationship | subject ラベル |
| `relationships[].subject_class_id` | relationship | subject の VG150 クラスID |
| `relationships[].subject_bbox` | relationship | subject の bbox |
| `relationships[].predicate` | relationship | 述語ラベル |
| `relationships[].predicate_id` | relationship | VG150 述語ID |
| `relationships[].object_id` | relationship | objects へのインデックス |
| `relationships[].object_label` | relationship | object ラベル |
| `relationships[].object_class_id` | relationship | object の VG150 クラスID |
| `relationships[].object_bbox` | relationship | object の bbox |

---

## 10. 次マイルストーンへの接続

本段階で「観測品質改善に伴う意味情報の収束は、構造化SGG出力＋LLM（特にLLM-SGG系/hybrid系）で定量的に推定可能」「TopKオラクルで飽和を解像度レンジ中盤に分布させ評価できる」ことを確認した。次段階では、ここに**質問・依頼の意図**をLLMプロンプトに条件として加え、「その意図に答えるのに十分な観測か」へと停止判定を拡張する。本マイルストーンのTopKオラクル・hybrid/LLM停止がその基盤となる。
