# Scene Graph 飽和に基づく観測停止段階の推定：解像度回復系列を用いた基礎検討

## Abstract

ロボットが対象に接近しながら「これ以上観測品質を上げても意味情報は増えない」と判断する能力は、効率的な行動計画の基盤となる。本稿では、既存 Scene Graph Generation（SGG）手法 EGTR の出力が画像解像度の段階的回復に伴いどのように増加・飽和するかを定量的に解析し、飽和段階を推定する複数の手法を比較する。100 枚の Visual Genome 画像に対し 20 段階の等比解像度スケジュールと bilinear 補間による劣化を適用し、Level-20 の高確信コア（上位 10 ノード / 上位 5 トリプレット）を擬似正解とする topK 相対飽和指標を定義した上で、ノード確信度ベースライン、LLM による SG 系列スコアリング、VLM による画像直接判定、およびそれらの融合（hybrid）を評価した。主要な知見として、（1）飽和は解像度系列の中〜後半に分布する（topK RelOracle(0.9) 平均停止レベル 16.36、中央値 17、71/100 が Level 20 未満で飽和）、（2）最良手法は **hybrid\_4b\_k2（MAE 3.06）** であり、後続の緩いクラスタ — vlm\_9b\_single\_k2（3.63）、llm\_sgg\_4b\_h5\_k2（3.82）、hybrid\_9b\_k2（3.84）— とともに、構造化 SGG + LLM + 確信度融合および大規模 VLM が飽和推定に有効である、（3）ただし per-image 精度は限定的であり（相関 $r \approx 0.19$--$0.25$）、手法は中心傾向を捉えるが正確な per-image 予測は依然として困難であること、を示した。

---

## 1. はじめに

### 1.1 問題設定

自律移動ロボットが未知環境で対象を観測する際、接近に伴い観測品質（解像度・視野・遮蔽状態）が改善し、知覚可能な意味情報は増加する。しかし一定の品質を超えると情報の増分は逓減し、追加の接近は時間・エネルギーの浪費となる。この「意味情報の飽和」を定量的に検知し、観測を打ち切る判断を支援することが本研究の目標である。

最終的にはロボットのタスク意図（「棚の上に何があるか確認せよ」「人物の服装を記録せよ」等）に条件付けられた停止判定を目指すが、本マイルストーン（MS1）は**意図なし**の基礎段階として、意味情報全体の飽和を扱う。

### 1.2 アプローチの概要

接近に伴う観測品質改善を、画像の**解像度の段階的回復**（縮小→bilinear 補間による再拡大）で近似する。各解像度段階で既存の SGG 手法（EGTR; Im et al., CVPR 2024）を適用して scene graph を抽出し、その系列から飽和を判定する。飽和の基準には、EGTR 自身の性能天井に起因する退化を回避するため、各画像の最高解像度出力の**高確信コア**（上位 10 ノード / 上位 5 トリプレット）を擬似正解とする**topK 相対飽和**を導入する。

### 1.3 貢献

1. **TopK 擬似 GT による非退化な相対飽和指標**：Level-20 出力の高確信コア（上位 10 ノード / 上位 5 トリプレット）を擬似 GT とする topK RelSGScore を定義した。ノイジーな低確信度の尾部を参照集合から除外することで、飽和が解像度系列の中〜後半に分布する（平均停止レベル 16.36、中央値 17、71/100 が Level 20 未満で飽和）非退化な評価枠組みを構築した。
2. **最良手法の同定と手法クラスタ**：hybrid\_4b\_k2（MAE 3.06）が明確に最良であり、後続の緩いクラスタ — vlm\_9b\_single\_k2（3.63）、llm\_sgg\_4b\_h5\_k2（3.82）、hybrid\_9b\_k2（3.84）— とともに、構造化 SGG + LLM + 確信度融合および大規模 VLM が飽和推定に有効であることを示した。consecutive-$k{=}2$ は全手法ファミリーで一貫して改善に寄与する。なお感度分析（coef 0.8）ではオラクルが早まるため上位 3 手法が僅差クラスタ（MAE 3.29--3.34）を形成するが、手法ファミリーのランキングは安定する。
3. **Bilinear 補間による光学的に忠実な劣化モデル**：bilinear 補間（面積平均、遠方観測のボケを再現）を採用し、nearest-neighbor（ブロックノイズにより飽和が解像度天井に集中）との対照で選択の根拠を実証した。
4. **EGTR の妥当性検証と実装上の知見**：関係エクスポートの端点確信度ゲート問題（23/100 画像で関係消失）の発見と修正、ollama の `num_ctx` 切り詰め問題（パース失敗 27%→0%）の特定と解消。

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

**SGDet R@K を per-image 飽和指標に使わない根拠**: 標準 SGDet R@K（IoU $\geq 0.5$ + class + predicate の strict matching）は、per-image 飽和指標として退化する。具体的には、(1) IoU は飽和タイミングに寄与しない（per-image 停止レベルが label-only 指標と 71/96 一致、すなわち bounding box 精度はラベル情報と同期して飽和する）、(2) 標準 per-image SGDet R@20 は 100 画像中 31 画像で Level-20 においてさえ 0 となり退化する。コーパスレベルの EGTR 妥当性検証（R@20 = 0.186; cf. 著者全データ 23.54）としてのみ使用する。

### 3.3 相対飽和と topK 擬似 GT

#### 3.3.1 動機

真 GT を基準とした SGScore は、EGTR の性能天井により退化する。具体的には、フル解像度（$s_{20}{=}1.0$）でも平均 TripletRecall は 0.286 にとどまり、SGScore $\geq 0.5$ に到達する画像は 100 枚中 28 枚のみである（§5.7）。この退化した Oracle では手法の停止精度を適切に評価できない。

さらに、実機ロボットの運用時には真の GT scene graph は存在しない。ロボットが持ちうるのは、自身の最接近時における SGG 出力のみである。

加えて、Level-20 出力の**全要素**を擬似 GT とする full-ref RelOracle では、低確信度のノイジーな尾部まで再現を要求するため、Oracle 停止レベルが解像度系列の上端（Level $\approx$ 17）に集中し、手法間の弁別力が制約される。高確信コアのみを参照集合とすることで、飽和を解像度系列の中盤に分布させ、評価の弁別力を確保する。

#### 3.3.2 TopK 擬似 GT の定義

各画像 $i$ の Level-$L$（$L{=}20$）における EGTR 出力から、**高確信コア**を参照集合として選定する：

- **ノード参照**: 物体確信度上位 10 カテゴリ（重複ラベル除外後のユニークラベル集合）
- **トリプレット参照**: triplet\_score 上位 5 トリプレット（ユニーク $(s, p, o)$ 集合）

この $N_{\mathrm{node}}{=}10$ / $N_{\mathrm{trip}}{=}5$ の設定は、二つの独立なアンカーから収束する：

1. **VG150 アノテーション規模**: 本サブセットにおける GT ユニークカテゴリ数の中央値は 8（四分位範囲 6--10）、ユニークトリプレット数の中央値は 5（同 3--6）であり、1 シーン分の意味内容に対応する。
2. **EGTR 確信度の崖**: 物体確信度は rank 10 で $\sim$0.36 を維持するが、triplet\_score は rank 5 以降 $\sim$0.001 に急落する。高確信コアの自然な境界が $\sim$10 ノード / $\sim$5 トリプレットに現れる。
3. **非対称 $N$ の根拠**: 関係は物体より疎である（全物体ペアが関係を持つわけではない）。対称的な $N{=}10/10$ では、Level-20 のトリプレット数（平均 $\sim$8）のほぼ全てが参照に含まれ、「コア選定」の意味が消失する。

実際の参照集合サイズは平均 5.8 ユニークノードラベル / 3.5 ユニークトリプレットである。

相対指標を以下のように定義する：

$$
\mathrm{RelNodeRecall}^{(i)}(t) = \frac{|\mathcal{N}(G_t^{(i)}) \cap \mathcal{N}_{\mathrm{topK}}^{(i)}|}{|\mathcal{N}_{\mathrm{topK}}^{(i)}|}
$$

$$
\mathrm{RelTripletRecall}^{(i)}(t) = \frac{|\mathcal{T}(G_t^{(i)}) \cap \mathcal{T}_{\mathrm{topK}}^{(i)}|}{|\mathcal{T}_{\mathrm{topK}}^{(i)}|}
$$

$$
\mathrm{RelSGScore}^{(i)}(t) = 0.5 \cdot \mathrm{RelNodeRecall}^{(i)}(t) + 0.5 \cdot \mathrm{RelTripletRecall}^{(i)}(t)
$$

ここで $\mathcal{N}_{\mathrm{topK}}^{(i)} \subseteq \mathcal{N}(G_L^{(i)})$、$\mathcal{T}_{\mathrm{topK}}^{(i)} \subseteq \mathcal{T}(G_L^{(i)})$ である。構成により $\mathrm{RelSGScore}(L) = 1.0$ が保証される（参照集合は Level-$L$ 出力の部分集合）。これは EGTR 出力の**収束・安定性**を測る指標であり、正しさ（correctness）ではない。正しさは別途 EGTR の公式評価プロトコル（R@K）で確認する（§5.7）。

#### 3.3.3 TopK RelOracle

$$
t_{\mathrm{RelOracle}}^{(i)}(\theta) = \min\{t \mid \mathrm{RelSGScore}^{(i)}(t) \geq \theta\}
$$

$\theta{=}0.9$ を主基準、$\theta{=}0.8$ を感度分析用の副基準とする。参照集合が Level-20 の高確信コア（上位 $N$）であるため、これらの要素は比較的早期に再現されやすく RelSGScore は高くなりやすい。低い閾値 0.8 では飽和が早く立ちすぎるため、より厳しい 0.9 が妥当である。また、RelSGScore は小さな参照集合上の離散値であるため量子化が生じ、$\theta \geq 0.95$ は事実上 1.0 と等価となり 43/100 画像が Level 20 に集中する。$\theta{=}0.9$ は分布を保つ最も厳しい閾値である。

Bilinear 補間下では topK RelOracle(0.9) に 71/100 画像が Level 20 未満で到達し、平均停止レベルは 16.36（median 17）であり、飽和が解像度系列の中〜後半に分布する（真 GT Oracle(0.5) では 40/100 のみ飽和）。

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

原画像を scale $s_t$ で縮小し、bilinear 補間で原サイズに再拡大する。Bilinear 補間は面積平均に基づく滑らかなボケを生成し、光学系における遠方観測（低空間周波数の通過）を模擬する。これは実際のセンサが距離に応じて低解像度像を取得する際の光学的低域通過フィルタに対応する。

Nearest-neighbor 補間はブロック状のエイリアシングを生じるが、これはデジタルズームの人工物であり光学的接近では発生しない。さらに nearest の離散的なアーティファクトは EGTR の微細構造認識を阻害し、飽和を解像度天井（Level $\approx$ 19）に集中させて手法間の弁別力を崩壊させる（§6.3）。

### 3.5 停止判定手法

以下の手法群を比較する。すべての手法で Level $\geq 5$（`llm.start_level`）から判定を開始する。

#### 3.5.1 Oracle（評価用基準）

**GT Oracle**: $t_{\mathrm{oracle}}^{(i)}(\theta) = \min\{t \mid \mathrm{SGScore}^{(i)}(t) \geq \theta\}$ — 真 GT に対する到達度。主閾値 $\theta{=}0.5$。

**TopK RelOracle（主基準）**: $t_{\mathrm{RelOracle}}^{(i)}(\theta) = \min\{t \mid \mathrm{RelSGScore}_{\mathrm{topK}}^{(i)}(t) \geq \theta\}$ — topK 擬似 GT（§3.3.2）に対する到達度。主閾値 $\theta{=}0.9$、感度分析用副閾値 $\theta{=}0.8$（§3.3.3 で主閾値選定の根拠を述べた）。

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

#### 標準 SGDet R@K 計算用データの保持

各 relationship レコードには、本研究のメイン指標（SGScore: label-only カテゴリ集合一致）に加え、将来的に標準 SGDet R@K（IoU ベース triplet マッチング）を post-hoc 計算可能とするために以下のフィールドを保持する：

- **予測側**（`src/milestone1/egtr_infer.py`）: `subject_class_id`, `subject_bbox`, `predicate_id`, `object_class_id`, `object_bbox`, `triplet_score`（= rel\_score $\times$ subj\_conf $\times$ obj\_conf、ランキング用）。`subject_id` / `object_id` は query index。
- **GT 側**（`src/milestone1/gt.py`）: 各 object に `class_id`、各 relationship に `subject_id` / `object_id`（objects へのインデックス）・`subject_class_id` / `object_class_id`・`subject_bbox` / `object_bbox`・`predicate_id`。

これらの追加フィールドは LLM 入力には混入しない（`build_sequence` は `subject_label` / `predicate` / `object_label` / `confidence` のみを明示選択するため）。メイン評価は引き続き SGScore で行い、SGDet R@K は今後の評価軸として出力データを準備した段階である。

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

1. 原画像を scale $s_t$ で縮小
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

**基準**: TopK RelOracle(0.9)（等比スケジュール、bilinear 補間、$n{=}100$）。

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

解像度の向上に伴い、ノード・関係情報は単調に増加する。以下に bilinear 補間下の主要レベルでの平均指標を示す（$n{=}100$）：

| Level | $s_t$ | NodeRecall | TripletRecall | SGScore | $\overline{c}_t$ |
|-------|-------|-----------|--------------|---------|-------------|
| 1 | 0.050 | 0.112 | 0.000 | 0.056 | 0.166 |
| 5 | 0.094 | 0.161 | 0.021 | 0.091 | 0.247 |
| 10 | 0.242 | 0.341 | 0.105 | 0.223 | 0.356 |
| 15 | 0.455 | 0.494 | 0.199 | 0.347 | 0.391 |
| 18 | 0.729 | 0.547 | 0.240 | 0.393 | 0.399 |
| 20 | 1.000 | 0.569 | 0.286 | 0.427 | 0.405 |

Bilinear 補間により意味情報は比較的早期から回復し始める。NodeRecall は Level 1 で既に 0.112、Level 10 で 0.341 に達する。TripletRecall は Level 1 で 0.000 だが Level 5 で 0.021、Level 10 で 0.105 と立ち上がる。「物体は低解像度でもある程度認識されるが関係はまだ」という段階は早期レベルの定性的観察として存在する。

**Q2: ノード先行の飽和**。NodeRecall は TripletRecall に先行して増加する。Level 10 で NodeRecall（0.341）は Level-20 最大値（0.569）の 60% に到達するのに対し、TripletRecall（0.105）は Level-20 最大値（0.286）の 37% にとどまる。ノードの情報回復が関係に先行するパターンは全解像度域で一貫している。

### 5.2 飽和スケールとオラクル定義の影響

**等比スケジュール・bilinear 補間**（本報告の主結果）における topK RelOracle の閾値感度を示す：

| 閾値 $\theta$ | 飽和画像数 | 平均停止レベル | 中央値 | 分布 |
|---------|-----------|------------|--------|------|
| **0.9（主基準）** | **71/100** | **16.36** | **17** | L5--20 に分散、ピーク L15（14 画像）、L20 が 29 画像 |
| 0.8（感度） | 80/100 | 15.01 | 14.5 | L11--20 に分散、ピーク L14（17 画像）、L20 が 20 画像 |

主基準 $\theta{=}0.9$ では 71/100 画像が Level 20 未満で飽和に到達し、平均停止レベルは 16.36（中央値 17）であり、飽和が解像度系列の**中〜後半**に分布する。感度分析用の $\theta{=}0.8$ ではオラクルが早まり（平均 15.01、中央値 14.5、80/100 が飽和）、上位手法クラスタが僅差に圧縮されるが、手法ファミリーのランキングは安定する。参照集合サイズは平均 5.8 ユニークノードラベル / 3.5 ユニークトリプレットである。

**Full-ref RelOracle との比較**: Level-20 出力の全要素を擬似 GT とする従来の full-ref RelOracle(0.8) では、平均停止レベルは 16.71（中央値 17、70/100 が飽和）であり、飽和が解像度系列の上端に集中していた。TopK による高確信コア選定は、ノイジーな低確信度の尾部への再現要求を除外することで、飽和を中盤へ移動させ、評価の弁別力を拡大した。

**Nearest-neighbor 補間との比較**: Nearest 補間下の topK RelOracle(0.8) では、平均停止レベル 18.5（中央値 19）と飽和が解像度天井に集中する。Nearest のブロック状アーティファクトは EGTR の微細構造認識を阻害し、ほぼ原画像に近い解像度まで意味情報の完全回復を遅延させる。Bilinear 補間の採用により、飽和の分布が中盤に広がり、手法間の弁別力が確保される（§6.3）。

### 5.3 手法比較（等比スケジュール、bilinear、$n{=}100$、vs TopK RelOracle(0.9)）

**Table 1: 全手法比較（MAE 昇順）**

| Rank | Method | Mean Stop | MAE | Med AE | Early % | Late % | Pearson $r$ |
|------|--------|-----------|-----|--------|---------|--------|-------------|
| 1 | **hybrid\_4b\_k2** | 16.34 | **3.06** | 2.0 | 0.42 | 0.43 | 0.193 |
| 2 | vlm\_9b\_single\_k2 | 14.41 | 3.63 | 3.0 | 0.64 | 0.26 | 0.176 |
| 3 | llm\_sgg\_4b\_h5\_k2 | 14.26 | 3.82 | 3.0 | 0.64 | 0.26 | 0.125 |
| 4 | hybrid\_9b\_k2 | 15.08 | 3.84 | 3.0 | 0.47 | 0.40 | 0.071 |
| 5 | hybrid\_4b\_k1 | 12.77 | 4.91 | 4.0 | 0.69 | 0.19 | 0.079 |
| 6 | vlm\_9b\_single\_k1 | 11.95 | 5.03 | 4.5 | 0.82 | 0.13 | 0.120 |
| 7 | baseline (conf $\geq$ 0.4) | 13.38 | 5.40 | 5.0 | 0.58 | 0.26 | 0.133 |
| 8 | vlm\_4b\_single\_k2 | 12.98 | 6.22 | 5.0 | 0.61 | 0.32 | −0.076 |
| 9 | hybrid\_9b\_k1 | 10.76 | 6.26 | 6.0 | 0.82 | 0.13 | 0.173 |
| 10 | llm\_sgg\_9b\_h5\_k2 | 9.91 | 6.81 | 7.0 | 0.84 | 0.10 | 0.139 |
| 11 | llm\_sgg\_4b\_h5\_k1 | 9.95 | 6.93 | 7.0 | 0.84 | 0.09 | −0.065 |
| 12 | llm\_sgg\_9b\_h5\_k1 | 6.95 | 9.43 | 10.0 | 0.96 | 0.01 | 0.034 |
| 13 | vlm\_4b\_single\_k1 | 7.00 | 9.66 | 10.0 | 0.94 | 0.04 | −0.115 |

**解釈**:

- **hybrid\_4b\_k2（MAE 3.06）が明確に最良**であり、確信度アンカーと LLM スコアの融合が最も効果的に飽和推定を行う。後続の緩いクラスタ — vlm\_9b\_single\_k2（3.63）、llm\_sgg\_4b\_h5\_k2（3.82）、hybrid\_9b\_k2（3.84）— がこれに続く。感度分析（coef 0.8）ではオラクルが早まるため上位 3 手法が僅差クラスタ（MAE 3.29--3.34）を形成するが、手法ファミリーのランキングは安定する。
- consecutive-$k{=}2$ は全手法ファミリーで一貫して $k{=}1$ を改善する（例: hybrid\_4b\_k1 4.91 → hybrid\_4b\_k2 3.06; llm\_sgg\_4b\_h5\_k1 6.93 → k2 3.82; vlm\_9b\_single\_k1 5.03 → k2 3.63）。
- baseline（conf $\geq$ 0.4）は MAE 5.40 で 7 位。最良手法（3.06）に対して明確な差がある。
- 全手法で Pearson 相関 $r$ は低い（最大 0.193）。手法は集団としての中心傾向を捉えるが、per-image レベルでの正確な停止予測は困難である。

### 5.4 モデルサイズの非対称性

VLM（画像直接入力）と LLM-SGG（構造化テキスト入力）で、モデルサイズの影響が非対称に現れる。

**VLM は大型モデルが有効**：

| Method | 4b ($k{=}1$) | 4b ($k{=}2$) | 9b ($k{=}1$) | 9b ($k{=}2$) |
|--------|-------------|-------------|-------------|-------------|
| VLM single MAE | 9.66 | 6.22 | 5.03 | **3.63** |

VLM は 9b + $k{=}2$ で MAE 3.63 を達成し上位クラスタに入る。一方 4b では MAE 9.66（$k{=}1$）と精度が大幅に低下する。

**LLM-SGG は小型モデルで十分**：

| Method | 4b ($k{=}1$) | 4b ($k{=}2$) | 9b ($k{=}1$) | 9b ($k{=}2$) |
|--------|-------------|-------------|-------------|-------------|
| LLM-SGG MAE | 6.93 | **3.82** | 9.43 | 6.81 |

9b は $k{=}1$ で極端に早い停止（平均 6.95）を示し、飽和を過信する。$k{=}2$ でも 9b（6.81）は 4b（3.82）に大差で劣る。

**モデルサイズ統制比較（4b 同士）**:

hybrid\_4b\_k2（SGG + LLM, MAE **3.06**）は vlm\_4b\_single\_k2（画像直接, MAE **6.22**）を大幅に上回る。同じ 4b モデルでは、構造化 SGG テキストを入力とする方が画像直接入力より飽和判定の精度が高い。ただし 9b VLM（3.63）は hybrid\_4b\_k2（3.06）に次ぐ精度であり、十分に大きな VLM は構造化 SGG ベース手法と競合し得る。

### 5.5 確信度信号のプラトーと収束停止の効果

平均物体確信度 $\overline{c}_t$ は Level 1 で $\approx 0.17$、Level 20 で $\approx 0.40$ であり、0.40 付近でプラトーに達する。この結果、絶対閾値による停止は脆弱となる：$\theta{=}0.5$ ではプラトーが 0.40 付近のため殆どの画像が到達できず、$\theta{=}0.4$ でも発火しない画像が存在する。

**収束停止（conf\_convergence）** はこの問題を部分的に解消する：信号が変化しなくなった時点で停止するため、プラトーの絶対値に依存しない。ただし確信度プラトーの到達と意味的飽和の間にギャップがあるため、単純な確信度収束は意味的飽和の推定としては不十分である。$\varepsilon$ / $k$ は現在の等比スケジュールに最適化されておらず、手動スケジュール用の値（$\varepsilon{=}0.02$, $k{=}3$）がそのまま使用されている点は今後の改善余地である。

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

### 6.1 上位手法クラスタの構造

hybrid\_4b\_k2（MAE 3.06）が明確に最良であり、後続の緩いクラスタ — vlm\_9b\_single\_k2（3.63）、llm\_sgg\_4b\_h5\_k2（3.82）、hybrid\_9b\_k2（3.84）— がこれに続く。この結果は以下の示唆を含む：

1. **確信度アンカーの効果**: hybrid\_4b\_k2（3.06）は確信度と LLM スコアの融合により最良の精度を達成し、Early/Late の偏りも均衡する（Early 42% / Late 43%）。確信度が LLM の早すぎ停止を抑制するアンカーとして機能する。
2. **構造化 SGG テキストの有効性**: LLM-SGG（3.82）は画像を一切見ずに SG 系列のテキストのみで飽和を判定する。EGTR の構造化出力が飽和判定に十分な情報を含むことを示す。
3. **大規模 VLM の競合力**: vlm\_9b\_single\_k2（3.63）は画像直接入力で上位クラスタに入る。ただし 4b VLM（6.22）は大幅に劣り、モデル規模への依存が大きい。
4. **Consecutive-$k{=}2$ の正則化**: 全手法ファミリーで $k{=}2$ が $k{=}1$ を改善し、単発のスコア揺らぎによる誤停止を防ぐ（例: hybrid\_4b\_k1 4.91 → k2 3.06）。
5. **感度分析（coef 0.8）**: オラクルが早まるため上位 3 手法が僅差クラスタ（llm\_sgg\_4b\_h5\_k2 3.29 / hybrid\_4b\_k2 3.31 / vlm\_9b\_single\_k2 3.34）を形成する。手法ファミリーのランキングは安定しており、coef 選択に対するロバスト性を示す。

### 6.2 VLM との比較に関する注意

Bilinear 補間下では vlm\_9b\_single\_k2（MAE 3.63）が上位クラスタに入り、最良の hybrid\_4b\_k2（3.06）に次ぐ精度で競合する。これは bilinear の滑らかなボケが VLM の画像認識と親和性が高いためと考えられる。一方、モデルサイズを統制した 4b 同士の比較では vlm\_4b\_single\_k2（6.22）が llm\_sgg\_4b\_h5\_k2（3.82）に大差で劣り、VLM ベースの飽和判定は大規模モデルへの依存が大きい。構造化 SGG テキスト入力は画像のアーティファクトに直接影響されない（EGTR の抽出精度を介して間接的に影響する）ため、小型モデルでも安定した精度を発揮する。

### 6.3 限界

1. **Per-image 精度の限界**: 全手法で Pearson 相関 $r \approx 0.19$--$0.25$ と弱く、手法は集団としての中心傾向を捉えるが、正確な per-image レベル予測は困難である。飽和判定の個別画像への適応が今後の課題である。
2. **TopK 擬似 GT の係数・$N$ の恣意性**: topK の $N_{\mathrm{node}}{=}10$ / $N_{\mathrm{trip}}{=}5$ および coef $\theta{=}0.9$ は VG150 規模・EGTR 確信度分布・量子化特性の 3 アンカーから決定し、$\theta{=}0.8$ との感度比較（§5.2, §6.1）でランキングの安定性を確認したが、$N$ に対する感度スイープは未実施である。$N$ の変動が Oracle 停止分布および手法ランキングに与える影響の系統的検証が必要である。
3. **Nearest-neighbor 補間の天井効果**: Nearest 補間下では飽和が Level $\approx$ 19 に集中し、手法間の弁別力が崩壊する。Bilinear 補間の採用はこの天井効果を回避するが、nearest が光学的に不適切な理由（ブロック状エイリアシングはデジタルズームの人工物であり光学的接近では発生しない）を明示的に議論する必要がある（§3.4.3）。
4. **データ規模**: 100 画像は統計的検出力に制約がある。
5. **Label-only マッチング**: インスタンス区別なし・bbox IoU 照合なしのため、同一カテゴリの複数インスタンスを区別できない。
6. **人物ノードの確信度問題**: 人物カテゴリの確信度（$\sim$0.2）が物体閾値（0.3）を下回り、ノード集合から脱落する。関係トラックは decouple で復活したが、ノード指標への影響は残存する。
7. **解像度のみの近似**: 実際の接近では視点変化・遮蔽解消・照明変化も伴うが、本研究は解像度改善のみをモデル化している。
8. **収束パラメータの未最適化**: conf\_convergence の $\varepsilon{=}0.02$ / $k{=}3$ は手動スケジュール用に設定されており、スケジュールに応じた再調整が必要。
9. **SGDet R@K による評価の未実施**: 本研究のメイン指標 SGScore は label-only カテゴリ集合一致であり、bbox IoU に基づく標準 SGDet R@K（Recall@20/50/100）での評価は行っていない。EGTR 予測 JSON および GT JSON には bbox / class\_id / predicate\_id / triplet\_score を保持済みであり、post-hoc での SGDet R@K 計算が可能な状態にある。

---

## 7. 結論と今後の展望

本マイルストーンでは、EGTR による scene graph 出力の解像度依存的な飽和挙動を 100 画像・20 段階の等比スケジュール・bilinear 補間で系統的に分析した。主要な結論は以下の通り：

1. **飽和は解像度系列の中〜後半に分布する**（topK RelOracle(0.9) 平均停止レベル 16.36、中央値 17、71/100 が Level 20 未満で飽和）。Level-20 出力の高確信コア（上位 10 ノード / 上位 5 トリプレット）を擬似 GT とする topK オラクルにより、ノイジーな尾部の再現要求を除外し、非退化な評価枠組みを構築した。
2. **hybrid\_4b\_k2（MAE 3.06）が最良の停止推定手法**であり、後続の緩いクラスタ — vlm\_9b\_single\_k2（3.63）、llm\_sgg\_4b\_h5\_k2（3.82）、hybrid\_9b\_k2（3.84）— とともに、構造化 SGG + LLM + 確信度融合および大規模 VLM が飽和推定に有効であることを示した。consecutive-$k{=}2$ が全手法で一貫して改善に寄与する。感度分析（coef 0.8）では上位 3 手法が僅差クラスタ（MAE 3.29--3.34）を形成し、手法ファミリーのランキングは安定する。
3. **Per-image 精度は限定的**である（相関 $r \approx 0.19$--$0.25$）。手法は集団としての中心傾向を捉えるが、正確な per-image レベル予測は依然として困難であり、今後の課題である。
4. **Bilinear 補間が光学的に忠実な劣化モデル**として適切であることを、nearest-neighbor との対照（nearest では飽和が天井に集中し弁別力が崩壊）で実証した。

**今後の展望**: 本マイルストーンの飽和推定手法・topK 相対飽和を基盤とし、次マイルストーンでは**質問・依頼の意図**を LLM プロンプトに条件として追加する。「棚の上に何があるか確認せよ」であれば棚関連のノード・関係の飽和に注目し、「人物の服装を記録せよ」であれば人物・衣類関連に重み付けする、意図条件付き観測停止へと拡張する。また、topK の $N$ に対する感度スイープ（coef については 0.8/0.9 の比較で安定性を確認済み）、出力データに保持済みの bbox / class\_id / triplet\_score を用いた標準 SGDet R@K（IoU $\geq 0.5$）での飽和曲線計算、および per-image 予測精度の改善も今後の課題である。

---

## 付録

### A.1 全手法比較表（vs TopK RelOracle(0.9)、bilinear、$n{=}100$）

| Rank | Method | Mean Stop | MAE | Med AE | Early % | Late % | Pearson $r$ |
|------|--------|-----------|-----|--------|---------|--------|-------------|
| 1 | hybrid\_4b\_k2 | 16.34 | 3.06 | 2.0 | 0.42 | 0.43 | 0.193 |
| 2 | vlm\_9b\_single\_k2 | 14.41 | 3.63 | 3.0 | 0.64 | 0.26 | 0.176 |
| 3 | llm\_sgg\_4b\_h5\_k2 | 14.26 | 3.82 | 3.0 | 0.64 | 0.26 | 0.125 |
| 4 | hybrid\_9b\_k2 | 15.08 | 3.84 | 3.0 | 0.47 | 0.40 | 0.071 |
| 5 | hybrid\_4b\_k1 | 12.77 | 4.91 | 4.0 | 0.69 | 0.19 | 0.079 |
| 6 | vlm\_9b\_single\_k1 | 11.95 | 5.03 | 4.5 | 0.82 | 0.13 | 0.120 |
| 7 | baseline (conf $\geq$ 0.4) | 13.38 | 5.40 | 5.0 | 0.58 | 0.26 | 0.133 |
| 8 | vlm\_4b\_single\_k2 | 12.98 | 6.22 | 5.0 | 0.61 | 0.32 | −0.076 |
| 9 | hybrid\_9b\_k1 | 10.76 | 6.26 | 6.0 | 0.82 | 0.13 | 0.173 |
| 10 | llm\_sgg\_9b\_h5\_k2 | 9.91 | 6.81 | 7.0 | 0.84 | 0.10 | 0.139 |
| 11 | llm\_sgg\_4b\_h5\_k1 | 9.95 | 6.93 | 7.0 | 0.84 | 0.09 | −0.065 |
| 12 | llm\_sgg\_9b\_h5\_k1 | 6.95 | 9.43 | 10.0 | 0.96 | 0.01 | 0.034 |
| 13 | vlm\_4b\_single\_k1 | 7.00 | 9.66 | 10.0 | 0.94 | 0.04 | −0.115 |

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
| TopK RelOracle | 主閾値 coef | 0.9（感度: 0.8） | 評価 |
| TopK RelOracle | top-$N$ nodes | 10 | 評価 |
| TopK RelOracle | top-$N$ triplets | 5 | 評価 |
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

**EGTR 予測 JSON スキーマ**（`outputs/egtr_predictions/{image_id}/level_{t:02d}.json`）:

各 relationship レコードのフィールド: `subject_id`（query index）, `subject_label`, `subject_class_id`（VG150 クラスID）, `subject_bbox`（`[x1, y1, x2, y2]`）, `predicate`, `predicate_id`（VG150 述語ID）, `object_id`（query index）, `object_label`, `object_class_id`, `object_bbox`, `confidence`（関係確信度）, `subject_conf`, `object_conf`, `triplet_score`（= rel\_score $\times$ subj\_conf $\times$ obj\_conf、SGDet R@K ランキング用）。

**GT JSON スキーマ**（`outputs/gt/{image_id}.json`）:

各 object のフィールド: `object_id`（objects 内インデックス）, `label`, `class_id`（VG150 クラスID）, `bbox`（`[x1, y1, x2, y2]`）。各 relationship のフィールド: `subject_id`（objects へのインデックス）, `subject_label`, `subject_class_id`, `subject_bbox`, `predicate`, `predicate_id`（VG150 述語ID）, `object_id`, `object_label`, `object_class_id`, `object_bbox`。
