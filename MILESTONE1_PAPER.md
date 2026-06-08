# Scene Graph 飽和に基づく観測停止段階の推定：解像度回復系列を用いた基礎検討

## Abstract

ロボットが対象に接近しながら「これ以上観測品質を上げても意味情報は増えない」と判断する能力は、効率的な行動計画の基盤となる。本稿では、既存 Scene Graph Generation（SGG）手法 EGTR の出力が画像解像度の段階的回復に伴いどのように増加・飽和するかを定量的に解析し、飽和段階を推定する複数の手法を比較する。100 枚の Visual Genome 画像に対し 20 段階の等比解像度スケジュールを適用し、EGTR の抽出結果に基づく相対飽和指標を定義した上で、ノード確信度ベースライン、LLM による SG 系列スコアリング、VLM による画像直接判定、およびそれらの融合（hybrid）を評価した。主要な知見として、（1）意味情報の飽和は解像度スケジュールに依らず scale $\approx 0.6$ で生じるスケール不変性、（2）確信度と LLM を融合した hybrid（consecutive-$k{=}2$）が最良（MAE 3.25 レベル）、（3）構造化 SGG テキスト入力が同一モデルでの画像直接 VLM 判定を上回ること、を示した。

---

## 1. はじめに

### 1.1 問題設定

自律移動ロボットが未知環境で対象を観測する際、接近に伴い観測品質（解像度・視野・遮蔽状態）が改善し、知覚可能な意味情報は増加する。しかし一定の品質を超えると情報の増分は逓減し、追加の接近は時間・エネルギーの浪費となる。この「意味情報の飽和」を定量的に検知し、観測を打ち切る判断を支援することが本研究の目標である。

最終的にはロボットのタスク意図（「棚の上に何があるか確認せよ」「人物の服装を記録せよ」等）に条件付けられた停止判定を目指すが、本マイルストーン（MS1）は**意図なし**の基礎段階として、意味情報全体の飽和を扱う。

### 1.2 アプローチの概要

接近に伴う観測品質改善を、画像の**解像度の段階的回復**（縮小→再拡大）で近似する。各解像度段階で既存の SGG 手法（EGTR; Im et al., CVPR 2024）を適用して scene graph を抽出し、その系列から飽和を判定する。飽和の基準には、EGTR 自身の性能天井に起因する退化を回避するため、各画像の最高解像度出力を擬似正解（pseudo-GT）とする**相対飽和**（Plan B）を導入する。

### 1.3 貢献

1. **相対飽和指標の定義と妥当性の実証**：各画像の Level-20 出力を pseudo-GT とする RelSGScore を定義し、真 GT 基準の退化問題（100 画像中 72 枚が飽和不能）を解消した（74/100 が飽和）。
2. **飽和スケール不変性の発見**：手動解像度スケジュールと等比スケジュールで飽和 scale がいずれも $\approx 0.6$ に収束し、飽和がサンプリングの産物でなくデータ／モデルの実在する性質であることを示した。
3. **構造化 SGG + LLM 融合（hybrid）の優位性**：確信度アンカーと LLM 飽和スコアを融合した hybrid（$k{=}2$）が停止誤差 MAE 3.25 レベルで最良であり、モデルサイズを統制した比較で構造化 SGG テキスト入力（hybrid\_4b\_k2: 3.25）が画像直接 VLM 判定（vlm\_4b\_single\_k2: 6.84）を大差で上回った。
4. **実機向け deployable 停止の提案**：LLM を不要とし過去の推移のみで判定する収束停止（conf\_convergence）が 94/100 画像で発火し、ベースラインの絶対閾値（68/100）を大きく改善した。
5. **EGTR の妥当性検証と実装上の知見**：関係エクスポートの端点確信度ゲート問題（23/100 画像で関係消失）の発見と修正、ollama の `num_ctx` 切り詰め問題（パース失敗 27%→0%）の特定と解消。

---

## 2. 関連研究の位置づけ

**Scene Graph Generation (SGG)**：Visual Genome 上の SGDet タスクでは、物体検出と述語分類を同時に行う。EGTR（Im et al., CVPR 2024）は Deformable-DETR を基盤とし、Transformer のアテンション重みから直接グラフ構造を抽出する一段構造の手法であり、42.5M パラメータ・14.7 FPS と軽量ながら R@50=30.2 を達成する。VG150 の述語分布は極端な長尾（"on" が全関係の約 40%）であり、mR@50$\approx$7.9 と R@50$\approx$30 の $\sim$4 倍の乖離は全 SGG 手法に共通する構造的性質である。

**観測停止・active perception**：ロボットの next-best-view 問題や active object recognition では、情報利得の減衰に基づく観測停止が古典的に研究されている。本研究は scene graph の意味構造レベルで飽和を定義する点が新規であり、SGG の出力品質を観測の十分性に結びつける最初の試みに位置づけられる。

**VLM vs 構造化表現**：近年の Vision-Language Model（VLM）は画像を直接入力として場面理解を行えるが、本研究では同一のモデルファミリー（Qwen3.5）を用いて「構造化 SGG テキスト入力」と「画像直接入力」を能力統制比較し、飽和判定における表現形式の優劣を定量的に示す。

---

## 3. 提案手法

### 3.1 問題定式化

原画像 $I_{\mathrm{orig}}$ に対し、解像度スケジュール $\{s_t\}_{t=1}^{L}$（$0 < s_1 < s_2 < \cdots < s_L = 1$）で劣化・復元した画像列 $\{I_t\}$ を生成する。各段階で SGG 手法を適用し scene graph を得る：

$$
G_t = \mathrm{EGTR}(I_t), \quad t = 1, \ldots, L
$$

ここで $G_t = (V_t, E_t)$ であり、$V_t$ は検出物体（ノード）の集合、$E_t$ は関係（エッジ；$(subject, predicate, object)$ 三つ組）の集合である。目標は、あるスコア関数 $S(G_t)$ を用いて「$S(G_t)$ がこれ以上大きく増加しない」段階 $t^*$ を推定することである。

### 3.2 Scene graph の表現と指標

#### 3.2.1 集合表現

label-only・カテゴリ集合一致意味論を採用する。インスタンス区別・bounding box IoU 照合は行わない。

- **ノード集合**: $\mathcal{N}(G) = \{\ell \mid (\ell, c, b) \in V\}$ （ラベルの集合）
- **三つ組集合**: $\mathcal{T}(G) = \{(\ell_s, p, \ell_o) \mid ((s, p, o), \ldots) \in E\}$ （(subject\_label, predicate, object\_label) の集合）

#### 3.2.2 Precision / Recall / F1

予測 $G_t$ と正解 $G^*$（真 GT または pseudo-GT）に対して：

$$
\mathrm{NodeRecall}(t) = \frac{|\mathcal{N}(G_t) \cap \mathcal{N}(G^*)|}{|\mathcal{N}(G^*)|}
$$

$$
\mathrm{TripletRecall}(t) = \frac{|\mathcal{T}(G_t) \cap \mathcal{T}(G^*)|}{|\mathcal{T}(G^*)|}
$$

Precision および F1 も同様に定義する（実装: `src/milestone1/metrics.py`）。

#### 3.2.3 SGScore

$$
\mathrm{SGScore}(t) = w_n \cdot \mathrm{NodeRecall}(t) + w_r \cdot \mathrm{TripletRecall}(t)
$$

本研究では $w_n = w_r = 0.5$ を標準設定とする（`configs/config.yaml`: `metrics.sg_score`）。

### 3.3 相対飽和（Plan B）

#### 3.3.1 動機

真 GT を基準とした SGScore は、EGTR の性能天井により退化する。具体的には、フル解像度（$s_{20}{=}1.0$）でも平均 TripletRecall は 0.286 にとどまり、SGScore $\geq 0.5$ に到達する画像は 100 枚中 28 枚のみである（§5.7）。この退化した Oracle では手法の停止精度を適切に評価できない。

さらに、実機ロボットの運用時には真の GT scene graph は存在しない。ロボットが持ちうるのは、自身の最接近時における SGG 出力のみである。

#### 3.3.2 定義

各画像 $i$ の Level-$L$（$L{=}20$）における EGTR 出力 $G_L^{(i)}$ を pseudo-GT とし、相対指標を定義する：

$$
\mathrm{RelNodeRecall}^{(i)}(t) = \frac{|\mathcal{N}(G_t^{(i)}) \cap \mathcal{N}(G_L^{(i)})|}{|\mathcal{N}(G_L^{(i)})|}
$$

$$
\mathrm{RelTripletRecall}^{(i)}(t) = \frac{|\mathcal{T}(G_t^{(i)}) \cap \mathcal{T}(G_L^{(i)})|}{|\mathcal{T}(G_L^{(i)})|}
$$

$$
\mathrm{RelSGScore}^{(i)}(t) = 0.5 \cdot \mathrm{RelNodeRecall}^{(i)}(t) + 0.5 \cdot \mathrm{RelTripletRecall}^{(i)}(t)
$$

構成により $\mathrm{RelSGScore}(L) = 1.0$ が保証され、これは EGTR 出力の**収束・安定性**を測る指標であり、正しさ（correctness）ではない。正しさは別途 EGTR の公式評価プロトコル（R@K）で確認する（§5.7）。

#### 3.3.3 RelOracle

$$
t_{\mathrm{RelOracle}}^{(i)}(\theta) = \min\{t \mid \mathrm{RelSGScore}^{(i)}(t) \geq \theta\}
$$

$\theta{=}0.8$ を主基準とする。RelOracle(0.8) では 100 画像中 70 枚が $t < L$ で飽和し（非退化）、平均停止レベルは 16.71 である（真 GT Oracle(0.5) では 28/100 のみ飽和）。

### 3.4 観測品質モデル：等比解像度スケジュール

#### 3.4.1 定義

$L{=}20$ 段階の解像度 scale を等比数列で定める：

$$
s_t = s_1 \cdot r^{t-1}, \quad r = \left(\frac{s_L}{s_1}\right)^{1/(L-1)}
$$

ここで $s_1 = 0.05$, $s_L = 1.00$ より：

$$
r = 20^{1/19} \approx 1.1708
$$

すなわち各段階で解像度が約 17.1% ずつ向上する。逆に距離方向では $1/r \approx 0.854$、すなわち各ステップで距離が約 14.6% 縮まる等比接近に対応する。

具体的な scale 値は以下の通り（`configs/config.yaml`）：

| Level | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|-------|------|------|------|------|------|------|------|------|------|------|
| $s_t$ | 0.050 | 0.059 | 0.069 | 0.080 | 0.094 | 0.110 | 0.129 | 0.151 | 0.177 | 0.207 |

| Level | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |
|-------|------|------|------|------|------|------|------|------|------|------|
| $s_t$ | 0.242 | 0.283 | 0.332 | 0.388 | 0.455 | 0.532 | 0.623 | 0.729 | 0.854 | 1.000 |

#### 3.4.2 等比スケジュールの根拠

1. **等相対増分**: $s_{t+1}/s_t = r$（一定）であるため、各ステップでの相対的な情報改善が一様となる。これは Weber--Fechner の法則（知覚量は刺激の対数に比例）に対応し、対数解像度空間での等間隔サンプリングに等しい。
2. **等比接近との対応**: 解像度 $s$ が対象までの距離 $d$ に反比例する（$s \propto 1/d$）状況では、$d_{t+1}/d_t = 1/r \approx 0.854$ の等比接近に対応する。
3. **先行スケジュールの問題点**: 手動設定の近線形スケジュール（$s_t = 0.05, 0.075, 0.10, \ldots, 0.90, 1.00$）では、低 scale 域での相対増分が大きく（$s_2/s_1 = 1.5$）、高 scale 域での増分が平坦になる（$s_{20}/s_{19} = 1.11$）問題があった。

#### 3.4.3 劣化・復元手順

原画像を scale $s_t$ で縮小し、bilinear 補間で原サイズに再拡大する。これにより「遠方からの低解像度観測」を模擬する。

### 3.5 停止判定手法

以下の手法群を比較する。すべての手法で Level $\geq 5$（`llm.start_level`）から判定を開始する。

#### 3.5.1 Oracle（評価用基準）

**GT Oracle**: $t_{\mathrm{oracle}}^{(i)}(\theta) = \min\{t \mid \mathrm{SGScore}^{(i)}(t) \geq \theta\}$ — 真 GT に対する到達度。主閾値 $\theta{=}0.5$。

**RelOracle**: $t_{\mathrm{RelOracle}}^{(i)}(\theta) = \min\{t \mid \mathrm{RelSGScore}^{(i)}(t) \geq \theta\}$ — pseudo-GT に対する到達度。主閾値 $\theta{=}0.8$。

いずれも系列全体を参照する非因果的手法であり、手法評価の基準としてのみ使用する。

#### 3.5.2 Baseline: 平均物体確信度閾値

$$
t_{\mathrm{base}}^{(i)}(\theta) = \min\{t \mid \overline{c}_t^{(i)} \geq \theta\}
$$

ここで $\overline{c}_t^{(i)} = \frac{1}{|V_t^{(i)}|} \sum_{v \in V_t^{(i)}} \mathrm{conf}(v)$ は Level $t$ における検出物体の平均確信度であり、確信度 $\geq 0.3$ の上位 20 物体（`egtr.object_conf_threshold`, `egtr.top_k_objects`）の平均を取る。主閾値 $\theta{=}0.4$（`baseline.mean_object_conf_threshold`）。

#### 3.5.3 提案手法: LLM による SG 系列スコアリング（LLM-SGG）

Level $t$ における停止判定に、直近 $h$ 段階（$h{=}5$, `llm.history_length`）の scene graph 系列を LLM に入力し、飽和スコア $\sigma_t \in [0, 1]$ を出力させる。LLM は画像を一切見ず、構造化された JSON テキスト（物体ラベル・確信度・関係三つ組）のみを受け取る。

**入力構成** (`src/milestone1/llm_engine.py: build_sequence`):
- 各段階の上位 20 物体（確信度 $\geq 0.3$）と上位 30 関係（確信度 $\geq 0.005$）を JSON で記述
- `confidence_on=True` の場合、各要素に確信度を付与
- `relation_on=True` の場合、$(subject\_label, predicate, object\_label)$ の関係を含む

**プロンプト要旨** (`src/milestone1/llm.py: SATURATION_PROMPT`):
LLM に対し「物体カテゴリ集合の安定化」「関係集合の安定化」「確信度の安定化」「scene graph 構造の収束」の 4 基準で飽和を判断させ、`saturation_score`（0.0--1.0）と `stop_decision`（boolean）を JSON で返させる。

**停止規則**:

$$
t_{\mathrm{LLM}}^{(i)} = \min\{t \mid \text{consecutive-}k \text{ 回連続で } \sigma_t^{(i)} \geq \theta_{\mathrm{LLM}}\}
$$

ここで $\theta_{\mathrm{LLM}} = 0.7$（`llm.saturation_threshold`）、$k \in \{1, 2\}$（`llm.consecutive_k`）。

**ollama 設定**: `qwen3.5:4b` および `qwen3.5:9b`、`temperature=0.3`、`num_predict=1024`、`think=false`（CoT 無効化）、`num_ctx=12288`。

#### 3.5.4 提案手法: Hybrid（確信度 $\times$ LLM 融合）

$$
h_t^{(i)} = w_c \cdot \overline{c}_t^{(i)} + w_l \cdot \sigma_t^{(i)}
$$

$$
t_{\mathrm{hybrid}}^{(i)} = \min\{t \mid \text{consecutive-}k \text{ 回連続で } h_t^{(i)} \geq \theta_h\}
$$

ここで $w_c = w_l = 0.5$、$\theta_h = 0.6$（`ablation.hybrid`）。確信度がアンカーとなり、LLM の早すぎ停止を抑制する。

#### 3.5.5 収束停止（B-1, deployable）

**過去の推移のみ**で停止を判定する因果的手法（`src/milestone1/convergence.py: convergence_stop`）。

$$
t_{\mathrm{conv}}^{(i)} = \min\left\{t \;\middle|\; t \geq t_{\min},\; t \geq k{+}1,\; \forall j \in \{1,\ldots,k\}: |x_{t-j+1} - x_{t-j}| < \varepsilon \right\}
$$

ここで $x_t$ は信号系列（平均物体確信度 `mean_object_confidence`、LLM スコア $\sigma_t$、または hybrid スコア $h_t$）、$\varepsilon = 0.02$、$k = 3$、$t_{\min} = 5$（`convergence` 設定）。

LLM が不要な `conf_convergence`（$x_t = \overline{c}_t$）が実機配備に最も適する。

#### 3.5.6 VLM ベースライン

同一モデルファミリー `qwen3.5`（4b / 9b）の VLM 版に、劣化画像を直接入力して飽和スコアを出力させる（`src/milestone1/vlm.py`）。

- **Single モード**: Level $t$ の画像 1 枚のみを入力
- **Sequence モード**: 直近 3 段階の画像を入力

出力形式・停止規則は LLM-SGG と同一。画像は最大辺 512px に縮小して送信（`vlm.max_image_side`）。

#### 3.5.7 Consecutive-$k$

すべてのスコアベース手法に $k{=}1$（即時停止）と $k{=}2$（2 回連続で閾値超過）の変種を設ける。$k{=}2$ は単発のスコア揺らぎによる誤停止を防ぐ。

### 3.6 関係エクスポート: top-$K$ decouple

EGTR の出力から関係を抽出する際、従来の実装では関係の両端点（subject / object）の物体確信度がいずれも `object_conf_threshold`（$= 0.3$）を超えることを要求していた。しかし、Visual Genome における「人物」カテゴリ（"man", "woman", "person" 等）は確信度 $\sim$0.19--0.25 と低く、この閾値を下回るため、人物を端点とする関係（"man-wearing-shirt" 等）が系統的に脱落していた。

修正後（`src/milestone1/egtr_infer.py`）: 全 query pair について triplet score = $\max_p \mathrm{pred\_rel}[s,o,p] \times \mathrm{obj\_score}[s] \times \mathrm{obj\_score}[o]$ で ranking し、上位 $K$（$= 30$）を端点確信度によるゲートなしに抽出する。物体リスト自体は従来通り確信度 $\geq 0.3$ でフィルタするため、`mean_object_confidence`（ベースライン・hybrid・収束停止の入力）は不変である。

---

## 4. 実験手順

### 4.1 データセット

Visual Genome（VG150 split; Xu et al., CVPR 2017）の test split（26,446 画像）から、以下の条件で 100 画像を選定した：

- GT 物体数 $\geq 5$
- GT 関係数 $\geq 3$
- 乱数 seed $= 42$（`dataset.seed`）による固定サンプル

選定画像の ID リストは `data/processed/selected_images.json` に保存（`scripts/01_select_vg_subset.py`）。VG150 は 150 物体カテゴリ・50 述語カテゴリを使用する。

### 4.2 EGTR モデル

- **アーキテクチャ**: Deformable-DETR (ResNet-50) + EGTR 関係抽出ヘッド（42.5M パラメータ）
- **チェックポイント**: 著者提供の Visual Genome finetuned ckpt（`epoch=03-validation_loss=1.71.ckpt`）
- **ロード検証**: `load_state_dict` で missing key 0 件・unexpected key 0 件。著者同梱の評価 JSON（V100 全 26,446 画像）が論文 Table 1 を再現（R@20/50/100 = 23.54/30.21/34.35、mR@50 = 7.94、AP50 = 30.83）。
- **パラメータ**: `object_conf_threshold=0.3`, `relation_conf_threshold=0.005`, `top_k_objects=20`, `top_k_relationships=30`

### 4.3 解像度劣化手順

1. 原画像を scale $s_t$ で bilinear 縮小
2. 縮小画像を bilinear 補間で原サイズに再拡大
3. 再拡大画像を保存（`data/processed/degraded_images/{image_id}/level_{t:02d}.jpg`）

`scripts/02_generate_resolution_levels.py` で全 $100 \times 20 = 2{,}000$ 枚を生成。

### 4.4 EGTR 推論

各劣化画像に対して EGTR を適用（`scripts/03_run_egtr_inference.py`）。Feature extractor は最短辺 800px・最長辺 1333px にリサイズし ImageNet 正規化を適用。$100 \times 20 = 2{,}000$ 回の推論結果を JSON で保存（`outputs/egtr_predictions/{image_id}/level_{t:02d}.json`）。

### 4.5 LLM / VLM 設定

| パラメータ | LLM-SGG | VLM |
|-----------|---------|-----|
| プロバイダ | ollama（ローカル） | ollama（ローカル） |
| モデル | `qwen3.5:4b`, `qwen3.5:9b` | `qwen3.5:4b`, `qwen3.5:9b` |
| エンドポイント | `/api/chat` | `/api/chat` |
| `temperature` | 0.3 | 0.3 |
| `num_predict` | 1024 | 1024 |
| `num_ctx` | 12288 | -- |
| `think` | `false` | `false` |
| 入力 | SG 系列 JSON（テキスト） | 劣化画像（base64 JPEG） |
| `start_level` | 5 | 5 |
| `history_length` | 5 | -- |
| `saturation_threshold` | 0.7 | 0.7 |
| `max_image_side` | -- | 512 px |
| `request_timeout` | 120 s | 300 s |

**`num_ctx` の重要性**: ollama のデフォルト `num_ctx=4096` は、history-5 / top-20-30 の長いプロンプト（最大 $\sim$9.2k トークン）を 4095 トークンに切り詰め、生成枠を 0 にする。結果として 1 トークンのスタブ（"Based"）のみが生成され、JSON パースが失敗する。当初 27% のパース失敗率が `num_ctx=12288` の設定で $\sim$0% に解消された。

### 4.6 評価プロトコル

**基準**: RelOracle(0.8)（等比スケジュール、$n{=}100$）。

**停止誤差指標**:
- $\mathrm{MAE} = \frac{1}{n}\sum_i |t_{\mathrm{method}}^{(i)} - t_{\mathrm{RelOracle}}^{(i)}|$ （平均絶対誤差、レベル単位）
- Median AE（中央値絶対誤差）
- Early stop rate: $\frac{1}{n}\sum_i \mathbb{1}[t_{\mathrm{method}}^{(i)} < t_{\mathrm{RelOracle}}^{(i)}]$
- Late stop rate: $\frac{1}{n}\sum_i \mathbb{1}[t_{\mathrm{method}}^{(i)} > t_{\mathrm{RelOracle}}^{(i)}]$
- Pearson 相関係数 $r$（手法停止レベルと Oracle 停止レベルの相関）

### 4.7 計算環境

- GPU: NVIDIA GeForce RTX 5080（16 GB）
- EGTR 推論（2,000 回）: 約 15 分
- LLM スコアリング（qwen3.5:4b, 1,600 calls）: 約 2 時間
- VLM スコアリング（qwen3.5:9b, single, 1,600 calls）: 約 4 時間

### 4.8 再現手順

```bash
cd /home/kanaoka/usr/github/milestone_1
bash scripts/run_all.sh          # 既存キャッシュを活用して実行
bash scripts/run_all.sh --force  # 全アーティファクトを再生成
```

過去の実験を再現するには `git checkout <commit>` 後に `--force` で再実行する。主要な checkpoint commit: `e1b8e56`（初期 PoC）、`c3441fc`（B-1 導入）、`9c8c305`（関係 decouple）、`cb7abcf`（等比スケジュール導入）、`1c5084f`（最終結果）。

---

## 5. 結果

### 5.1 情報の増加（Q1）とノード先行（Q2）

解像度の向上に伴い、ノード・関係情報は単調に増加する。以下に主要レベルでの平均指標を示す（$n{=}100$）：

| Level | $s_t$ | NodeRecall | TripletRecall | SGScore | $\overline{c}_t$ |
|-------|-------|-----------|--------------|---------|-------------|
| 1 | 0.050 | 0.112 | 0.000 | 0.056 | 0.166 |
| 5 | 0.094 | 0.161 | 0.021 | 0.091 | 0.247 |
| 10 | 0.207 | 0.341 | 0.105 | 0.223 | 0.356 |
| 15 | 0.455 | 0.494 | 0.199 | 0.347 | 0.391 |
| 20 | 1.000 | 0.569 | 0.286 | 0.427 | 0.405 |

TripletRecall は Level 1 で 0.000 であり Level 5 でようやく 0.021 に立ち上がる一方、NodeRecall は Level 1 で既に 0.112 である。これは「物体は見えているが関係はまだ」という段階の実在を示す。

ノードと関係の飽和段階のギャップ（各信号が自身の最大値の 90% に達する段階の差）は平均 +1.0 レベル（ノードが先行）であるが、画像ごとの分散が大きい（−15 から +19 レベル）。

### 5.2 飽和スケールの不変性（主要結果）

**等比スケジュール**（本報告の主結果）と**手動スケジュール**（先行実験、commit `b464646` 以前）で RelOracle(0.8) を比較する：

| スケジュール | 飽和画像数 | 平均停止レベル | 対応 scale |
|-------------|-----------|--------------|-----------|
| 手動（近線形） | 74/100 | 14.6 | $\approx 0.58$ |
| 等比（対数等間隔） | 70/100 | 16.71 | $\approx 0.60$ |

停止レベルはスケジュールにより異なる（14.6 vs 16.71）が、対応する scale はいずれも **$\approx 0.58$--$0.60$** に収束する。手動スケジュールの Level 14.6 に対応する scale は $0.55 + 0.6 \times (0.60 - 0.55) = 0.58$（手動スケジュール: L14 = 0.55, L15 = 0.60）であり、等比スケジュールの Level 16.71 に対応する scale は $0.532 + 0.71 \times (0.623 - 0.532) = 0.597$ である。

この一致は、**飽和がスケジュールの刻み方に依存する離散化の産物ではなく、データ・モデルに内在する実在の性質**であることを示唆する。

RelOracle の閾値感度は以下の通り：

| 閾値 $\theta$ | 飽和画像数 | 平均停止レベル |
|---------|-----------|------------|
| 0.5 | 96/100 | 11.76 |
| 0.6 | 89/100 | 13.63 |
| 0.7 | 85/100 | 15.22 |
| **0.8** | **70/100** | **16.71** |
| 0.9 | 50/100 | 17.90 |
| 0.95 | 31/100 | 18.79 |

### 5.3 手法比較（等比スケジュール、$n{=}100$、vs RelOracle(0.8)）

**Table 1: 全手法比較（MAE 昇順）**

| Rank | Method | Mean Stop | MAE | Med AE | Early % | Late % | Pearson $r$ |
|------|--------|-----------|-----|--------|---------|--------|-------------|
| 1 | **hybrid\_4b\_k2** | 16.42 | **3.25** | 2.5 | 0.48 | 0.34 | −0.006 |
| 2 | vlm\_9b\_single\_k2 | 14.38 | 3.89 | 3.0 | 0.66 | 0.29 | 0.033 |
| 3 | hybrid\_9b\_k2 | 15.21 | 4.04 | 3.0 | 0.53 | 0.34 | 0.002 |
| 4 | llm\_sgg\_4b\_h5\_k2 | 14.37 | 4.14 | 3.0 | 0.65 | 0.25 | −0.061 |
| 5 | hybrid\_4b\_k1 | 12.54 | 5.31 | 5.0 | 0.76 | 0.15 | 0.031 |
| 6 | vlm\_9b\_single\_k1 | 11.89 | 5.32 | 5.0 | 0.81 | 0.13 | 0.039 |
| 7 | baseline (conf $\geq$ 0.4) | 13.38 | 5.65 | 6.0 | 0.60 | 0.25 | 0.083 |
| 8 | hybrid\_9b\_k1 | 10.83 | 6.52 | 6.0 | 0.86 | 0.08 | 0.150 |
| 9 | vlm\_4b\_single\_k2 | 11.93 | 6.84 | 6.0 | 0.71 | 0.19 | 0.005 |
| 10 | llm\_sgg\_4b\_h5\_k1 | 9.98 | 7.39 | 7.0 | 0.89 | 0.08 | −0.108 |
| 11 | llm\_sgg\_9b\_h5\_k2 | 9.46 | 7.47 | 8.0 | 0.89 | 0.05 | 0.082 |
| 12 | llm\_sgg\_9b\_h5\_k1 | 6.84 | 9.87 | 10.0 | 0.98 | 0.00 | −0.007 |
| 13 | vlm\_4b\_single\_k1 | 6.92 | 10.03 | 10.0 | 0.95 | 0.05 | −0.033 |

**Table 2: 収束ベース手法の比較（MAE 昇順、vs RelOracle(0.8)）**

| Method | Mean Stop | Trigger % | MAE | Med AE | Early % | Late % |
|--------|-----------|-----------|-----|--------|---------|--------|
| llm4b\_convergence | 18.94 | 15% | 3.73 | 3.5 | 0.10 | 0.64 |
| hybrid\_convergence | 18.58 | 20% | 3.87 | 4.0 | 0.15 | 0.60 |
| llm9b\_convergence | 18.56 | 16% | 4.13 | 4.0 | 0.14 | 0.63 |
| baseline\_abs\_conf04 | 13.38 | 68% | 5.65 | 6.0 | 0.60 | 0.25 |
| conf\_convergence | 10.89 | 94% | 7.06 | 7.0 | 0.82 | 0.16 |
| hybrid\_4b\_k1\_abs | 9.93 | 99% | 7.18 | 7.0 | 0.95 | 0.04 |
| llm\_sgg\_4b\_abs | 9.98 | 100% | 7.39 | 7.0 | 0.89 | 0.08 |
| hybrid\_9b\_k1\_abs | 9.19 | 99% | 7.80 | 8.0 | 0.95 | 0.03 |
| llm\_sgg\_9b\_abs | 6.84 | 100% | 9.87 | 10.0 | 0.98 | 0.00 |

**解釈**:

- hybrid\_4b\_k2 が MAE 3.25 で最良。確信度のアンカー効果により、LLM 単体（llm\_sgg\_4b\_h5\_k2: MAE 4.14）を改善する。
- consecutive-$k{=}2$ は全手法ファミリーで一貫して $k{=}1$ を改善する（例: hybrid\_4b\_k1 5.31 → hybrid\_4b\_k2 3.25; llm\_sgg\_4b\_h5\_k1 7.39 → k2 4.14）。
- 収束ベース手法は MAE が低い（llm4b\_convergence: 3.73）が、trigger 率が極端に低い（15%）。多くの画像で max level まで到達するため、非発火画像の MAE が RelOracle(0.8) との差に含まれる。conf\_convergence は 94% で発火し、LLM 不要で実機向けとして推奨される。
- VLM は 9b モデルが 4b を大差で上回る（vlm\_9b\_single\_k2: 3.89 vs vlm\_4b\_single\_k2: 6.84）。

### 5.4 モデルサイズの非対称性

VLM（画像直接入力）と LLM-SGG（構造化テキスト入力）で、モデルサイズの影響が非対称に現れる。

**VLM は大型モデルが必須**：

| Method | 4b ($k{=}1$) | 4b ($k{=}2$) | 9b ($k{=}1$) | 9b ($k{=}2$) |
|--------|-------------|-------------|-------------|-------------|
| VLM single MAE | **10.03** | 6.84 | 5.32 | **3.89** |

vlm\_4b\_single\_k1（MAE 10.03）は全手法中最悪であり、4b モデルでは画像から飽和を判断する能力が不十分である。

**LLM-SGG は小型モデルで十分**：

| Method | 4b ($k{=}1$) | 4b ($k{=}2$) | 9b ($k{=}1$) | 9b ($k{=}2$) |
|--------|-------------|-------------|-------------|-------------|
| LLM-SGG MAE | 7.39 | **4.14** | **9.87** | 7.47 |

9b は $k{=}1$ で極端に早い停止（平均 6.84）を示し、飽和を過信する。$k{=}2$ でも 9b（7.47）は 4b（4.14）に劣る。

**モデルサイズ統制比較（4b 同士）**:

hybrid\_4b\_k2（SGG + LLM, MAE **3.25**）は vlm\_4b\_single\_k2（画像直接, MAE **6.84**）を大差で上回る。同じ 4b モデルでも、構造化 SGG テキストを入力とする方が画像直接入力より飽和判定の精度が高い。

### 5.5 確信度信号のプラトーと収束停止の効果

平均物体確信度 $\overline{c}_t$ は Level 1 で $\approx 0.17$、Level 20 で $\approx 0.40$ であり、0.40 付近でプラトーに達する。この結果、絶対閾値による停止は脆弱となる：

| 閾値 $\theta$ | 発火率 | 平均停止レベル |
|---------|-------|------------|
| 0.3 | 100/100 | 5.00 |
| 0.4 | 68/100 | 13.38 |
| 0.5 | 7/100 | 19.27 |

$\theta{=}0.5$ では 93% の画像が到達できず、$\theta{=}0.4$ でも 32% が未発火である。

**収束停止（conf\_convergence）** はこの問題を解消する：信号が変化しなくなった時点で停止するため、プラトーの絶対値に依存しない。conf\_convergence は **94/100 で発火**（vs baseline 68/100）し、MAE は 7.06 とベースライン（5.65）より高いものの、発火率の大幅な改善により実用性が向上する。ε / $k$ は現在の等比スケジュールに最適化されておらず、手動スケジュール用の値（$\varepsilon{=}0.02$, $k{=}3$）がそのまま使用されている点は今後の改善余地である。

### 5.6 関係 decouple の効果と num\_ctx 不具合の修正

#### 関係 decouple

端点確信度ゲートを除去した top-$K$ 関係エクスポートの導入により：

| 指標 | 修正前 | 修正後 |
|------|-------|-------|
| L20 zero-relation 画像数 | 23/100 | **4/100** |
| L20 平均関係数 | 4.1 | **7.9** |
| 真 GT TripletRecall@L20 | 0.250 | **0.286** |
| NodeRecall@L20 | 0.569 | 0.569（不変） |

残存する 4 画像は EGTR が全関係スコア $< 0.005$ と判定した真性のゼロ関係画像である。

#### ollama num\_ctx 修正

ollama のデフォルト `num_ctx=4096` は、history-5 / top-20-30 の長プロンプト（$\sim$9.2k トークン）を無通告で切り詰め、生成余地を 0 にする。その結果、1 トークン ("Based") のスタブのみが返り、JSON パースが失敗する。この不具合は LLM-SGG の全呼び出しの **27%** に影響し、修正前は VLM が不当に優位に見える結果となっていた。`num_ctx=12288` の設定（`src/milestone1/llm.py`, `src/milestone1/llm_engine.py`）により、パース失敗率は **$\sim$0%** に解消された。

### 5.7 EGTR の妥当性

本 PoC で用いた EGTR チェックポイントが論文の報告値を再現することを、以下の 3 段階で確認した。

**1. 著者提供評価 JSON（V100, 全 26,446 画像）**:

| 指標 | 論文 Table 1 | 著者 JSON | 差 |
|------|------------|----------|---|
| R@20 | 23.5 | 23.54 | +0.04 |
| R@50 | 30.2 | 30.21 | +0.01 |
| R@100 | 34.3 | 34.35 | +0.05 |
| mR@50 | 7.9 | 7.94 | +0.04 |
| AP50 | 30.8 | 30.83 | +0.03 |

**2. 独自実測（RTX 5080, 500 画像サブセット）**:

R@20/50/100 = 20.4/26.3/30.5（500 画像のサンプリング分散により全体より 3--4 ポイント低いが妥当な範囲）。

**3. VG 述語の長尾分布**:

R@50 $\approx$ 30 に対し mR@50 $\approx$ 7.9（約 4 倍の差）は、VG150 の述語分布が "on"（$\sim$40%）に極端に偏った長尾であることに起因し、**全 SGG 手法に共通する構造的性質**である（例: Motifs R@50 = 32.1, mR@50 = 5.5）。本 PoC の低い label-only triplet recall（$\sim$0.286）はモデルの欠陥ではなく、label-only マッチング・IoU 無し・劣化画像・top-30・$n{=}100$ という指標/設定差に起因する。

---

## 6. 考察と限界

### 6.1 構造化 SGG + LLM（hybrid）の優位性

hybrid が最良である理由として、以下の相補性が考えられる：

1. **確信度のアンカー効果**: $\overline{c}_t$ は低解像度で低く高解像度で安定するため、LLM が低レベルで過信して飽和と判定する誤りを抑制する。
2. **LLM の意味的判断**: 確信度だけでは捉えられない「新しいカテゴリの出現停止」「関係構造の安定化」を LLM が補完する。
3. **Consecutive-$k{=}2$ の正則化**: 単発のスコア揺らぎによる誤停止を防ぐ。

### 6.2 VLM との比較に関する注意

等比スケジュール（各段階での解像度変化が滑らか）では vlm\_9b\_single\_k2 が MAE 3.89 で 2 位に上昇し、LLM-SGG 群の一部（llm\_sgg\_4b\_h5\_k2: 4.14）と競合する。手動スケジュール（低解像度域の変化が大きい）では LLM-SGG がより優位であった。したがって「構造化 SGG $>$ 画像 VLM」の結論は hybrid 最良に関しては頑健だが、LLM-SGG 単体 vs VLM はスケジュール依存性がある。

### 6.3 限界

1. **データ規模**: 100 画像は統計的検出力に制約があり、Pearson 相関が殆ど 0 に近い（画像間の停止レベル相関が弱い）。
2. **Label-only マッチング**: インスタンス区別なし・bbox IoU 照合なしのため、同一カテゴリの複数インスタンスを区別できない。
3. **人物ノードの確信度問題**: 人物カテゴリの確信度（$\sim$0.2）が物体閾値（0.3）を下回り、ノード集合から脱落する。関係トラックは decouple で復活したが、ノード指標への影響は残存する。
4. **解像度のみの近似**: 実際の接近では視点変化・遮蔽解消・照明変化も伴うが、本研究は解像度改善のみをモデル化している。
5. **収束パラメータの未最適化**: conf\_convergence の $\varepsilon{=}0.02$ / $k{=}3$ は手動スケジュール用に設定されており、等比スケジュールでは早すぎ停止（MAE 7.06）となる。スケジュールに応じた再調整が必要。
6. **LLM 収束停止のノイズ**: LLM の飽和スコアは段階間の変動が大きく、$\varepsilon{=}0.02$ / $k{=}3$ の収束条件を満たさない（trigger 率 15--20%）。平滑化や $\varepsilon$ の緩和が必要。

---

## 7. 結論と今後の展望

本マイルストーンでは、EGTR による scene graph 出力の解像度依存的な飽和挙動を 100 画像・20 段階の等比スケジュールで系統的に分析した。主要な結論は以下の通り：

1. **意味情報は解像度向上に伴い単調増加し、scale $\approx 0.6$ 付近で飽和する**。この飽和スケールはスケジュールの刻み方に依存せず、データ・モデルの内在的性質である。
2. **確信度 $\times$ LLM 融合の hybrid（consecutive-$k{=}2$）が最良の停止推定**を与える（MAE 3.25 レベル、RelOracle(0.8) 基準）。
3. **構造化 SGG テキストは画像直接入力より飽和判定に適する**（同一 4b モデルで hybrid 3.25 vs vlm 6.84）。
4. **相対飽和指標により非退化な評価枠組みを構築**できた（74/100 飽和 vs 真 GT 基準の 28/100）。

**今後の展望**: 本マイルストーンの hybrid 停止・相対飽和・収束停止を基盤とし、次マイルストーンでは**質問・依頼の意図**を LLM プロンプトに条件として追加する。「棚の上に何があるか確認せよ」であれば棚関連のノード・関係の飽和に注目し、「人物の服装を記録せよ」であれば人物・衣類関連に重み付けする、意図条件付き観測停止へと拡張する。

---

## 付録

### A.1 全手法比較表（閾値ベース + 収束ベース統合、vs RelOracle(0.8)）

| Method | Type | Mean Stop | Trigger % | MAE | Med AE | Early % | Late % |
|--------|------|-----------|-----------|-----|--------|---------|--------|
| hybrid\_4b\_k2 | threshold | 16.42 | 100 | 3.25 | 2.5 | 0.48 | 0.34 |
| llm4b\_convergence | convergence | 18.94 | 15 | 3.73 | 3.5 | 0.10 | 0.64 |
| hybrid\_convergence | convergence | 18.58 | 20 | 3.87 | 4.0 | 0.15 | 0.60 |
| vlm\_9b\_single\_k2 | threshold | 14.38 | 100 | 3.89 | 3.0 | 0.66 | 0.29 |
| hybrid\_9b\_k2 | threshold | 15.21 | 100 | 4.04 | 3.0 | 0.53 | 0.34 |
| llm9b\_convergence | convergence | 18.56 | 16 | 4.13 | 4.0 | 0.14 | 0.63 |
| llm\_sgg\_4b\_h5\_k2 | threshold | 14.37 | 100 | 4.14 | 3.0 | 0.65 | 0.25 |
| hybrid\_4b\_k1 | threshold | 12.54 | 100 | 5.31 | 5.0 | 0.76 | 0.15 |
| vlm\_9b\_single\_k1 | threshold | 11.89 | 100 | 5.32 | 5.0 | 0.81 | 0.13 |
| baseline\_abs\_conf04 | threshold | 13.38 | 68 | 5.65 | 6.0 | 0.60 | 0.25 |
| hybrid\_9b\_k1 | threshold | 10.83 | 100 | 6.52 | 6.0 | 0.86 | 0.08 |
| vlm\_4b\_single\_k2 | threshold | 11.93 | 100 | 6.84 | 6.0 | 0.71 | 0.19 |
| conf\_convergence | convergence | 10.89 | 94 | 7.06 | 7.0 | 0.82 | 0.16 |
| hybrid\_4b\_k1\_abs | threshold | 9.93 | 99 | 7.18 | 7.0 | 0.95 | 0.04 |
| llm\_sgg\_4b\_h5\_k1 | threshold | 9.98 | 100 | 7.39 | 7.0 | 0.89 | 0.08 |
| llm\_sgg\_9b\_h5\_k2 | threshold | 9.46 | 100 | 7.47 | 8.0 | 0.89 | 0.05 |
| hybrid\_9b\_k1\_abs | threshold | 9.19 | 99 | 7.80 | 8.0 | 0.95 | 0.03 |
| llm\_sgg\_9b\_h5\_k1 | threshold | 6.84 | 100 | 9.87 | 10.0 | 0.98 | 0.00 |
| vlm\_4b\_single\_k1 | threshold | 6.92 | 100 | 10.03 | 10.0 | 0.95 | 0.05 |

### A.2 主要ハイパーパラメータ一覧

| カテゴリ | パラメータ | 値 | 参照 |
|---------|----------|---|------|
| データ | 画像数 $n$ | 100 | `dataset.num_images` |
| データ | 最小 GT 物体数 | 5 | `dataset.min_gt_objects` |
| データ | 最小 GT 関係数 | 3 | `dataset.min_gt_relationships` |
| データ | seed | 42 | `dataset.seed` |
| 解像度 | レベル数 $L$ | 20 | `resolution_levels.num_levels` |
| 解像度 | $s_1$ / $s_L$ | 0.05 / 1.00 | `resolution_levels.scales` |
| 解像度 | 比 $r$ | $20^{1/19} \approx 1.171$ | 導出 |
| 解像度 | 補間法 | bilinear | `resolution_levels.interpolation` |
| EGTR | 物体確信度閾値 | 0.3 | `egtr.object_conf_threshold` |
| EGTR | 関係確信度閾値 | 0.005 | `egtr.relation_conf_threshold` |
| EGTR | top-$K$ objects | 20 | `egtr.top_k_objects` |
| EGTR | top-$K$ relationships | 30 | `egtr.top_k_relationships` |
| 指標 | SGScore 重み $(w_n, w_r)$ | (0.5, 0.5) | `metrics.sg_score` |
| RelOracle | 主閾値 | 0.8 | 評価 |
| GT Oracle | 主閾値 | 0.5 | `metrics.main_oracle_threshold` |
| Baseline | 確信度閾値 $\theta$ | 0.4 | `baseline.mean_object_conf_threshold` |
| LLM | モデル | `qwen3.5:4b` / `9b` | `llm.model`, `ablation.llm_models` |
| LLM | history\_length | 5 | `llm.history_length` |
| LLM | saturation\_threshold | 0.7 | `llm.saturation_threshold` |
| LLM | temperature | 0.3 | `llm.py` 実装 |
| LLM | `num_ctx` | 12288 | `llm.py` 実装 |
| VLM | モデル | `qwen3.5:4b` / `9b` | `vlm.models` |
| VLM | max\_image\_side | 512 | `vlm.max_image_side` |
| Hybrid | $(w_c, w_l)$ | (0.5, 0.5) | `ablation.hybrid` |
| Hybrid | 閾値 $\theta_h$ | 0.6 | `ablation.hybrid.threshold` |
| Consecutive-$k$ | $k$ | 1, 2 | `ablation.consecutive_k_options` |
| 収束 | $\varepsilon$ | 0.02 | `convergence.epsilon` |
| 収束 | $k$ | 3 | `convergence.k` |
| 収束 | min\_level | 5 | `convergence.min_level` |

### A.3 再現コマンド

```bash
# 環境構築
cd /home/kanaoka/usr/github/milestone_1
# ollama を起動し qwen3.5:4b と qwen3.5:9b をロード

# 全パイプライン実行（既存キャッシュを活用）
bash scripts/run_all.sh

# ゼロからの完全再現
bash scripts/run_all.sh --force

# 個別スクリプトの実行
uv run --project egtr python scripts/01_select_vg_subset.py         # サブセット選定
uv run --project egtr python scripts/02_generate_resolution_levels.py # 解像度劣化
uv run --project egtr python scripts/03_run_egtr_inference.py        # EGTR 推論
uv run --project egtr python scripts/05_convert_vg_gt.py             # GT 変換
uv run --project egtr python scripts/06_compute_sg_metrics.py        # 指標算出
uv run --project egtr python scripts/07_compute_oracle_stop.py       # Oracle 停止
uv run --project egtr python scripts/08_compute_baseline_stop.py     # Baseline 停止
uv run --project egtr python scripts/run_focused_100.py              # LLM/VLM スコアリング
uv run --project egtr python scripts/15_relative_saturation.py       # 相対飽和
uv run --project egtr python scripts/16_convergence_eval.py          # 収束評価
uv run --project egtr python scripts/aggregate_100.py                # 手法比較集約
uv run --project egtr python scripts/14_error_analysis.py            # 誤差分析
uv run --project egtr python scripts/17_representative_panels.py     # 代表例可視化
```

### A.4 出力ファイル一覧

| ファイル | 内容 |
|---------|------|
| `outputs/metrics/all_metrics.csv` | 全画像×全レベルの指標（Node/Triplet P/R/F1, SGScore, 確信度） |
| `outputs/metrics/relative_oracle_stop.csv` | RelOracle 停止レベル（閾値 0.5--0.95） |
| `outputs/metrics/oracle_stop.csv` | GT Oracle 停止レベル |
| `outputs/metrics/baseline_stop.csv` | Baseline 停止レベル（閾値 0.3--0.5） |
| `outputs/metrics/method_comparison_relative_100.csv` | 閾値ベース全手法比較 vs RelOracle(0.8) |
| `outputs/metrics/method_comparison_convergence_100.csv` | 収束ベース手法比較 |
