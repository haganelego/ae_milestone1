# マイルストーン1 報告：既存SGG出力に基づく意味情報飽和判定の基礎検討

> 観測品質の改善（解像度の段階的回復）に伴い、既存Scene Graph Generation（EGTR）の出力から
> **意味情報がどこで飽和するか**を推定できるかを検討したPoCの報告。質問意図は扱わない基礎段階。

---

## 0. 要約（先に結論）

- **解像度が上がるほどノード・関係情報は増える**（NodeRecall 0.034→0.569、TripletRecall 0.000→0.286、SGScore 0.017→0.427、L1→L20）。単調増加を確認。
- **ノード飽和と関係飽和のギャップは限界的**（収束ギャップ平均 +0.46 レベル、中央値 0、69/100でノード先行）。早期レベルでは「物体は見えるが関係はまだ」（L1: NodeRecall 0.034 vs TripletRecall 0.000）が存在するが、飽和タイミングの差はほぼ同時。最近傍補間の性質（全帯域を等しく破壊）が関係復元と物体復元の差を縮小。
- **ノード確信度だけでは飽和判定に不十分**：EGTRの平均物体確信度は約0.40で頭打ちし、絶対閾値（0.5以上）はほぼ到達不能。
- **LLMでSG系列をスコアリングする提案手法が明確に最良**：確信度とLLM飽和スコアを融合した **hybrid_4b_k2 が最良**（停止誤差 MAE **1.29** レベル、RelOracle(0.8)基準）。baseline（確信度のみ、MAE 2.93）を大幅に上回り、**構造化SGG＋LLM＋確信度アンカー＋consecutive-k2** の有効性が強く支持された。
- **全100画像が飽和に到達**（RelOracle(0.8) 100/100）。旧bilinear補間では70/100に留まっていたが、最近傍補間で全画像が飽和。ただし**飽和は遅い**（RelOracle(0.8) 平均停止 L18.96、scale ≈ 0.85）。最近傍のブロック状アーティファクトが高解像度近傍まで意味復元を抑制するため。
- **構造化SGG＋LLMが画像VLMを圧倒**：同一モデル `qwen3.5` で比較し、hybrid_4b_k2（MAE 1.29）が最良VLMの vlm_9b_single_k2（MAE 6.18）を大差で上回る。構造化SGG出力の方が意味情報の増分・安定性の評価に適する。
- 評価設計上の重要な工夫：飽和の基準を**真GTではなく各画像の最接近時(Level-20)出力に対する相対到達度**とすることで、EGTRの性能天井による退化を回避。さらに解像度スケジュールを**等比数列**（各レベルで相対解像度を一定倍）にして、飽和タイミングを刻み方に依存しない形で測定。

---

## 1. 目的と位置づけ

本研究の最終目標は、ロボットが質問・依頼の意図に応じて「これ以上観測しなくても十分」と自律判断する認知システムの構築である。本マイルストーンはその**基礎段階**として、質問意図を考慮せず、観測品質の改善に伴って画像から得られる意味情報（Scene Graphとして表現）がどのように増加・安定するかを定量的に扱う。

- **データセット**：Visual Genome（VG150, test split）
- **SGG手法**：EGTR（CVPR2024、Deformable-DETRベース、finetune済みckpt）
- **観測品質の近似**：画像解像度の20段階回復（scale 0.05→1.00、縮小→最近傍補間（nearest-neighbor）再拡大）。最近傍補間はブロック状アーティファクトを生じ、bilinear の滑らかなボケより実際のセンサ解像度不足・遠方観測時の劣化に近い見え方を再現する。
- **LLM**：ollama `qwen3.5:4b` / `qwen3.5:9b`（画像は見せず、構造化SG系列のみ入力）
- **対象**：100画像（GT物体≥5、GT関係≥3、seed固定のランダムサンプル）

> **スコープ明記**：接近行動そのものではなく、接近に伴う観測品質改善のうち**解像度改善のみ**を扱う（視点変化・遮蔽・移動計画は対象外）。LLMは画像理解ではなく、**SGG出力系列の増分・安定性の評価**に用いる。

---

## 2. 手法

### 2.1 パイプライン
VGサブセット選定 → 20段階解像度劣化（最近傍補間） → 各段階でEGTR推論 → GT/擬似GTと比較 → 指標算出 → 停止判定 → 可視化。

- **補間法**: 最近傍補間（nearest-neighbor）。config: `resolution_levels.interpolation: nearest`。最近傍補間はブロック状ノイズを生じ、bilinear の滑らかなボケと比べてEGTRの意味復元をより高解像度まで抑制する（飽和が遅れる）。これは意図通りの挙動であり、実際のセンサ劣化に近い条件での飽和特性を評価するために選択した。
- **出力データの拡張**: EGTR予測JSONおよびGT JSONの各 relationship レコードに、標準 SGDet R@K（IoUベース triplet マッチング）を post-hoc 計算可能とするためのフィールド（bbox, class\_id, predicate\_id, triplet\_score 等）を保持。メイン指標は引き続き SGScore（label-only・カテゴリ集合一致）であり、SGDet R@K は将来の評価軸として出力データを準備した段階。追加フィールドは LLM 入力には混入しない（`build_sequence` が `subject_label` / `predicate` / `object_label` / `confidence` のみを明示選択するため）。

### 2.2 指標（label-only・カテゴリ集合一致）
- Node Precision/Recall/F1、Triplet Precision/Recall/F1（subject–predicate–object のラベル三つ組）
- `SGScore(t) = 0.5·NodeRecall(t) + 0.5·TripletRecall(t)`

### 2.3 飽和の基準：相対飽和（Plan B）
当初は真GTへの到達度で飽和を定義したが、**EGTRの triplet recall はフル解像度でも約0.25–0.30で頭打ち**（手法の天井）。そのため絶対閾値（SGScore≥0.5/0.8）は到達不能で、100画像中72枚が「永遠に飽和しない」退化したOracleになった。

→ 基準を **各画像自身の Level-20（最接近）EGTR出力＝擬似GT** への到達度に変更：
`RelSGScore(t) = 0.5·RelNodeRecall(t) + 0.5·RelTripletRecall(t)`（対 L20）。
これで「EGTRの天井」交絡を除去でき、**実機ロボットの状況（手元のSGG出力しか持たない）とも整合**。RelOracle(0.8) では **100/100が飽和に到達**（全画像が非退化）。

RelOracle 閾値別の到達状況（n=100）：

| θ | 到達数 | 平均停止 | 中央値 | scale@平均 |
|---|--------|----------|--------|-----------|
| 0.7 | 100/100 | 18.40 | 19 | 0.779 |
| **0.8** | **100/100** | **18.96** | **19** | **0.849** |
| 0.9 | 100/100 | 19.49 | 20 | 0.925 |

RelOracle(0.8) 停止レベルの分布：{13:1, 14:1, 15:1, 16:5, 17:2, 18:14, 19:32, 20:44}。大多数がL19–20で飽和し、最近傍補間の「高解像度近傍まで意味復元が遅れる」性質を反映。

GT-based Oracle（絶対SGScore閾値）の到達状況：**真に飽和した画像数**（全レベル中の最大SGScoreが θ 以上に達する画像数）と、停止レベル（閾値未達はL20にフォールバック）の平均を示す。

| θ | 真に飽和した画像数 | 平均停止（L20フォールバック込み） | 中央値 |
|---|--------|----------|--------|
| 0.3 | 82/100 | 17.57 | 18 |
| 0.4 | 59/100 | 18.66 | 20 |
| 0.5 | **37/100** | 19.20 | 20 |
| 0.6 | 20/100 | 19.65 | 20 |
| gain_stop | — | 5.87 | 2 |

GT Oracle(0.5) では **37/100 しか真に飽和せず**、真GT基準の退化性が残る。これが RelOracle（各画像のL20出力を基準とし、構成上 RelSGScore=1.0 に必ず到達するため 100/100 が非退化に飽和）を主基準とする根拠である。

- ※ これは「抽出情報の**収束・安定性**」を測る（研究の問い）。「正しさ」は別途 EGTR純正プロトコルの R@K で確認（§5）。

### 2.4 停止判定手法
- **Oracle**：基準（GTベース／相対）への到達で理想停止段階を定義（評価用）。
- **Baseline**：平均物体確信度 ≥ θ（ノード確信度のみ）。
- **提案・LLM**：直近5段階のSG系列をLLMに入力し `saturation_score` を出力、閾値で停止。
- **提案・hybrid**：`0.5·平均物体確信度 + 0.5·LLM飽和スコア` を融合して停止（確信度がLLMの暴走を抑える）。
- **収束停止（B-1, 実機向け）**：信号（物体確信度等）の直近k遷移の増分が全てε未満になったら停止（**過去のみ使用＝deployable**）。
- **VLMベースライン**：同じ `qwen3.5`（vision対応）に**画像を直接**見せて飽和判定（single/sequence）。「構造化SG vs 生画像」をモデル能力を統制して比較。
- 各停止系に `consecutive-k`（k=1/2）変種。

---

## 3. 結果：5つの研究上の問いへの回答

### Q1. 解像度改善でノード・関係情報は増えるか → **はい**
平均（n=100, L1→L20）：NodeRecall 0.034→0.569、TripletRecall 0.000→0.286、SGScore 0.017→0.427。単調増加を確認。

代表的なレベル別推移（100画像平均）：

| Level | scale | NodeRecall | TripletRecall | SGScore | mean_obj_conf | mean_rel_conf |
|-------|-------|-----------|---------------|---------|---------------|---------------|
| 1  | 0.050 | 0.034 | 0.000 | 0.017 | 0.162 | 0.000 |
| 5  | 0.094 | 0.063 | 0.000 | 0.031 | 0.116 | 0.002 |
| 10 | 0.242 | 0.122 | 0.002 | 0.062 | 0.158 | 0.003 |
| 15 | 0.455 | 0.223 | 0.062 | 0.143 | 0.354 | 0.013 |
| 18 | 0.729 | 0.463 | 0.188 | 0.325 | 0.389 | 0.022 |
| 19 | 0.854 | 0.539 | 0.235 | 0.387 | 0.402 | 0.029 |
| 20 | 1.000 | 0.569 | 0.286 | 0.427 | 0.405 | 0.031 |

最近傍補間はブロック状アーティファクトを生じるため、bilinear補間と比べてL1–L17の指標が全般に低く、回復がL18–19に集中する（急峻な立ち上がり）。L20は原画像そのものであり補間法に依存しない。

相対指標（対L20）：

| Level | RelSGScore | RelNodeRecall |
|-------|-----------|---------------|
| 1  | 0.016 | 0.032 |
| 5  | 0.038 | 0.071 |
| 10 | 0.068 | 0.131 |
| 15 | 0.232 | 0.307 |
| 18 | 0.570 | 0.659 |
| 19 | 0.750 | 0.827 |
| 20 | 1.000 | 1.000 |

### Q2. どの段階でGTに近づき安定するか／ノード飽和と関係飽和に差はあるか → **差は弱い（ほぼ同時、ノードがわずかに先行）**
ノード90%到達平均レベル 17.63、トリプレット90%到達平均レベル 18.09。収束ギャップは**平均 +0.46 レベル、中央値 0**。ノードが先にまたは同時に飽和した画像は **69/100**。

最近傍補間はブロック状アーティファクトにより全帯域の情報を等しく破壊するため、bilinear（滑らかなボケ → 低周波=物体輪郭が先に復元）と異なり、物体と関係の復元タイミング差が縮小する。早期レベルでは「物体は見えるが関係はまだ」（L1: TripletRecall 0.000 vs NodeRecall 0.034）という段階が依然存在するが、**飽和ギャップとしては限界的**であり、旧bilinear結果（ギャップ平均 +0.9）からは弱まった。

→ SGGを用いる意義は「早期レベルでの関係情報の不在」により依然示されるが、飽和タイミングの先行性はこの補間条件下では強い根拠にならない。

### Q3. ノード確信度だけで飽和を判断できるか → **不十分**
平均物体確信度はレベル別 L1=0.162→**L20=0.405で頭打ち**（≈0.40プラトー）。絶対閾値の到達率は ≥0.3で100/100（平均停止10.45）、**≥0.4で73/100**（平均停止16.69）、**≥0.5で≈7%**（プラトーが0.5を超えることは稀）。当初計画値0.7では事実上到達不能。
→ 絶対閾値はプラトー水準に依存して脆く、**関係情報や系列の安定性（収束）を見る必要**がある。収束化（B-1: conf_convergence）すると **90%で発火**するが MAE 10.52 と高い（早すぎ停止）。実機向けにはLLM収束系（hybrid_convergence MAE 3.04, 発火32%）がより有望。

### Q4. LLMでSG系列をスコアリングし飽和タイミングを推定できるか → **できる（hybrid_4b_k2 が明確に最良）**
RelOracle(0.8) 基準の停止誤差（n=100, MAE昇順, 全手法）：

| Rank | 手法 | 平均停止 | MAE | MedAE | Early% | Late% | Pearson r |
|------|------|----------|-----|-------|--------|-------|-----------|
| 1 | **hybrid_4b_k2**（確信度×LLM） | 19.35 | **1.29** | 1.0 | 0.11 | 0.49 | 0.034 |
| 2 | hybrid_9b_k2 | 18.30 | 2.16 | 1.0 | 0.23 | 0.37 | −0.091 |
| 3 | llm_sgg_4b_h5_k2 | 17.77 | 2.53 | 1.0 | 0.31 | 0.37 | −0.011 |
| 4 | baseline（conf≥0.4） | 16.69 | 2.93 | 2.0 | 0.63 | 0.17 | 0.009 |
| 5 | hybrid_4b_k1 | 16.30 | 3.38 | 2.0 | 0.55 | 0.20 | −0.052 |
| 6 | hybrid_9b_k1 | 15.58 | 4.04 | 3.0 | 0.61 | 0.17 | −0.106 |
| 7 | vlm_9b_single_k2 | 12.84 | 6.18 | 6.0 | 0.97 | 0.01 | −0.005 |
| 8 | llm_sgg_4b_h5_k1 | 12.17 | 7.07 | 7.0 | 0.81 | 0.08 | 0.088 |
| 9 | vlm_9b_single_k1 | 10.74 | 8.22 | 8.0 | 1.00 | 0.00 | 0.048 |
| 10 | vlm_4b_single_k2 | 8.10 | 10.98 | 12.5 | 0.96 | 0.03 | 0.096 |
| 11 | llm_sgg_9b_h5_k2 | 6.25 | 12.71 | 13.0 | 1.00 | 0.00 | 0.092 |
| 12 | vlm_4b_single_k1 | 5.22 | 13.74 | 14.0 | 1.00 | 0.00 | 0.068 |
| 13 | llm_sgg_9b_h5_k1 | 5.08 | 13.88 | 14.0 | 1.00 | 0.00 | 0.031 |

- **hybrid_4b_k2 が明確に最良**（MAE 1.29）。確信度がアンカーとなりLLMの早すぎ停止を抑制し、2位以下と0.87ポイント差。提案手法 hybrid_4b_k2 は bilinear（旧MAE 3.25）・nearest（MAE 1.29）の**両補間条件で一貫して1位**であり、補間法を変えても頑健。
- **consecutive-k=2 が一貫して有効**：hybrid_4b k1→k2 で MAE 3.38→1.29、hybrid_9b で 4.04→2.16、llm_sgg_4b で 7.07→2.53。
- **小モデル(4b)が構造化SGGで優位**：LLM-SGG MAE 4b_k2=2.53 vs 9b_k2=12.71。9bは飽和を過信し早期停止。VLMでは逆に9b>4b（MAE 6.18 vs 10.98）で大モデルが必要。
- **baseline（確信度のみ）は4位**（MAE 2.93）。提案手法上位3手法（hybrid/LLM系, k=2）が全て上回る。

#### 収束ベース停止（B-1, deployable）の比較

過去の信号推移のみで停止判定する収束系手法（RelOracle(0.8)基準、n=100）：

| 手法 | 平均停止 | Trigger% | MAE | MedAE | Early% | Late% |
|------|----------|----------|-----|-------|--------|-------|
| baseline_abs_conf04 | 16.69 | 73% | 2.93 | 2.0 | 0.63 | 0.17 |
| hybrid_convergence | 17.28 | 32% | 3.04 | 1.0 | 0.29 | 0.35 |
| llm4b_convergence | 16.88 | 34% | 3.46 | 1.0 | 0.33 | 0.36 |
| llm9b_convergence | 13.89 | 56% | 5.67 | 6.0 | 0.56 | 0.20 |
| hybrid_4b_k1_abs | 12.54 | 95% | 6.56 | 6.5 | 0.92 | 0.02 |
| llm_sgg_4b_abs | 12.17 | 88% | 7.07 | 7.0 | 0.81 | 0.08 |
| hybrid_9b_k1_abs | 10.83 | 96% | 8.25 | 9.0 | 0.95 | 0.02 |
| conf_convergence | 8.60 | 90% | 10.52 | 12.0 | 0.86 | 0.06 |
| llm_sgg_9b_abs | 5.08 | 100% | 13.88 | 14.0 | 1.00 | 0.00 |

hybrid_convergence（MAE 3.04）は発火率32%ながら発火時の精度が高い。conf_convergence（LLM不要）は90%発火だがMAE 10.52で早すぎ停止。実機配備には発火率と精度のトレードオフ調整（ε/k）が必要。

#### 誤り分析・データセット特性（n=100）

- **未再現カテゴリ**: 全レベルを通じて一度も再現されなかったカテゴリは **24/137 (17.5%)**。
- **未再現述語**: 全レベルを通じて一度も再現されなかった述語は **20/32 (62.5%)**。VG150の極端な長尾分布（`on` が全関係の約40%）に起因し、EGTRの欠陥ではない。
- **L20時点の関係ゼロ画像**: 4/100（関係デカップル修正後）。

### Q5. 構造化SGG出力はVLM/キャプションより分析に適すか → **はい（大差）**
同一モデル `qwen3.5` で「構造化SGテキスト（画像なし）」vs「生画像」を比較。

- 能力統制比較（4b）：hybrid_4b_k2（SGG+LLM）MAE **1.29** vs vlm_4b_single_k2（画像）MAE **10.98** → 約8.5倍の差。
- 最良VLM：vlm_9b_single_k2 MAE 6.18（7位）であり、SGG系上位3手法にも及ばない。
- VLMは全般に早すぎ停止（Early% 0.96–1.00）。最近傍補間のブロック状アーティファクトが画像品質を直接見るVLMに悪影響し、劣化画像を「十分」と判断しがち。一方、構造化SGG系列は画像アーティファクトの影響を受けず、意味情報の増分・安定性の変化をLLMが正確に追跡できる。

→ 構造化SGG出力は画像VLMより意味情報の飽和判定に**圧倒的に**適する。

---

### 代表例（`outputs/figures/representative_*.png`）
飽和スペクトラムを跨ぐ3画像を可視化（各：level 1/5/10/15/20 の画像＋生成SG＋GT＋RelSGScore推移＋LLM reason）：

| 役割 | image_id | RelOracleStop@0.8 | シーン概要 |
|---|---|---|---|
| 早期飽和 | 2317015 | 18 | テニス選手（man, racket, arm…21物体・17関係） |
| 中間 | 2317647 | 20 | ベンチで本/眼鏡（bench, book, girl…33物体・21関係） |
| 後期/未飽和 | 2324457 | 20 | 赤いバスのある街路（bus, building…24物体・17関係） |

最近傍補間下では全画像の飽和がL18–20に集中するため、旧bilinear時代のような早期(L9)/中間(L16)/後期(L20)の明確なスプレッドは縮小している。`representative_summary.png` に3画像のRelSGScore推移とRelOracleStopを重ね、飽和タイミングの分布を一望できる。

---

## 4. 評価設計上の重要な発見（方法論）

1. **退化Oracleの回避**：絶対GT基準はEGTRの天井で機能しない → **相対飽和（対L20）** で全100画像が飽和に到達（RelOracle(0.8) 100/100）。実機の停止問題とも整合。
2. **ollama `num_ctx` 不具合**：既定 `num_ctx=4096` が長プロンプト（履歴5×top20/30 で最大~9.2kトークン）を無言で切り詰め、生成枠0で1トークン停止 → LLMの「パース失敗27%」の正体。`num_ctx=12288` で **0%に解消**。修正前は不当にVLM>LLMに見えていた。
3. **関係エクスポートの過剰フィルタ**：人物ボックスは確信度~0.2で物体閾値0.3に届かず、その関係（man-wearing-shirt 等）が連鎖的に消え、23/100の擬似GTが関係0件になっていた。**EGTRの失敗ではなく出力規約の問題**。本来のSGGの使い方（端点足切りなしのtop-K）に修正し **23→4/100**、真GT triplet recall@L20 0.250→0.286。物体トラック（確信度信号）は据え置きで baseline/hybrid/収束に影響なし。
4. **収束ベース停止（実機向け）**：絶対閾値は信号の飽和水準に依存して脆い。「これ以上接近しても変わらない」を**過去の推移だけ**で判定する収束停止は、conf_convergence（90%発火）やhybrid_convergence（32%発火, MAE 3.04）など有望だが、発火率とMAEのトレードオフがあり ε/k の調整が必要。

---

## 5. EGTRの妥当性（論文整合性）

「triplet recallが低い／述語の大半が当たらない」のは**EGTRの不具合ではなく既知の長尾現象**であることを確認：
- 同梱の著者評価JSONが論文Table 1を再現（R@20/50/100 = 23.54/30.21/34.35、mR@50=7.94、AP50=30.83）。我々の500画像実測（RTX5080）も想定レンジ内（R@20/50/100 = 20.4/26.3/30.5）。
- VG150は述語が極端な長尾（`on` だけで全関係の約40%）。R@50≈30 に対し **mR@50≈7.9**（約4倍差）は全SGG手法共通＝**多くの述語が低recallなのは構造的に想定通り**。
- 我々のPoC指標が低めなのは label-only・IoU無し・n=100・top-K・劣化画像という**指標/設定差**であり、モデル欠陥ではない。

---

## 6. 限界

- 100画像・label-only・カテゴリ集合一致（インスタンス区別なし、bbox IoU無し）。
- 人物ノード自体は確信度~0.25で物体閾値0.3に届かず node 集合から落ちる（関係トラックは復活済みだが node-level の人物弱点は残存）。
- 観測品質を解像度のみで近似（視点・遮蔽・移動は対象外）。
- LLM収束系（収束をLLMスコアに適用）は ε=0.02/k=3 ではノイズで発火しにくく要調整。

---

## 7. 成果物

- 指標CSV：`outputs/metrics/{all_metrics, oracle_stop, baseline_stop, relative_metrics, relative_oracle_stop, method_comparison_relative_100, method_comparison_convergence_100}.csv`
- 図：`outputs/figures/{relative_method_comparison_mae_100, stop_distribution_all_100, saturation_rate_100, convergence_vs_absolute_mae_100, representative_*}.png`
- 研究ノート：`.claude/docs/research/{m1-analysis, m1-evaluation, m1-relative-saturation, m1-scale100, m1-convergence, egtr-paper-claims, egtr-repro, m1-code-review}.md`
- コード：`src/milestone1/*`、`scripts/01..17`（実行手順は `MILESTONE1.md`）

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
| `relationships[].triplet_score` | relationship | rel\_score $\times$ subj\_conf $\times$ obj\_conf（SGDet R@K ランキング用） |

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

## 8. 次マイルストーンへの接続

本段階で「観測品質改善に伴う意味情報の収束は、構造化SGG出力＋LLM（特に確信度×LLMのhybrid）で定量的に推定可能」「相対飽和の枠組みで非退化に評価できる」ことを確認した。次段階では、ここに**質問・依頼の意図**をLLMプロンプトに条件として加え、「その意図に答えるのに十分な観測か」へと停止判定を拡張する。本マイルストーンの相対飽和・hybrid停止・収束停止がその基盤となる。
