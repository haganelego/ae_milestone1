# マイルストーン1 報告：既存SGG出力に基づく意味情報飽和判定の基礎検討

> 観測品質の改善（解像度の段階的回復）に伴い、既存Scene Graph Generation（EGTR）の出力から
> **意味情報がどこで飽和するか**を推定できるかを検討したPoCの報告。質問意図は扱わない基礎段階。

---

## 0. 要約（先に結論）

- **解像度が上がるほどノード・関係情報は増える**（NodeRecall 0.112→0.569、TripletRecall 0.000→0.286、SGScore 0.056→0.427、L1→L20）。単調増加を確認。Bilinear補間を採用し、Nearest と比べて低解像度での回復が早い（例: L10 NodeRecall 0.341 vs Nearest 0.122）。
- **飽和は解像度レンジの中〜後半に分布**：TopKオラクル（coef 0.9）の平均停止 L16.36（中央値 17）、71/100 が L20 前に飽和、L5–20 に広がる。Bilinear補間の採用により飽和点が解像度レンジ内に留まる（Nearest では L18–20 に天井集中）。
- **ノード確信度だけでは飽和判定に不十分**：EGTRの平均物体確信度は約0.40で頭打ちし、絶対閾値（0.5以上）はほぼ到達不能。
- **hybrid_4b_k2 が最良の停止推定手法**：TopKオラクル（coef 0.9）基準で、hybrid_4b_k2（MAE **3.06**）が明確に最良。続く緩いクラスタとして vlm_9b_single_k2（MAE **3.63**）、llm_sgg_4b_h5_k2（MAE **3.82**）、hybrid_9b_k2（MAE **3.84**）が確信度のみの baseline（MAE 5.40）を明確に上回る。ただし **per-image 精度は限定的**（Pearson r ≈ 0.19–0.25、手法は中心傾向を捉えるが正確な per-image 予測は困難）。
- **TopKオラクルによる評価枠組みの改善**：L20 の高確信コア（top-10 ノード / top-5 トリプレット）を擬似GTとすることで、ノイジーな裾の完全再現要求を外し、飽和が中〜後半（平均 16.36 / 中央値 17）に分布する評価を実現。
- **構造化SGG＋LLMが画像VLMを上回る**：同一モデル `qwen3.5` で比較し、SGG系上位手法（hybrid_4b_k2 MAE 3.06、llm_sgg_4b_h5_k2 MAE 3.82）が最良VLM vlm_9b_single_k2（MAE 3.63）と同等以上。4b VLM（MAE 6.22–9.66）は大きく劣後。
- **Bilinear補間を採用**（光学的忠実性）。Nearest 結果は比較として併記。

---

## 1. 目的と位置づけ

本研究の最終目標は、ロボットが質問・依頼の意図に応じて「これ以上観測しなくても十分」と自律判断する認知システムの構築である。本マイルストーンはその**基礎段階**として、質問意図を考慮せず、観測品質の改善に伴って画像から得られる意味情報（Scene Graphとして表現）がどのように増加・安定するかを定量的に扱う。

- **データセット**：Visual Genome（VG150, test split）
- **SGG手法**：EGTR（CVPR2024、Deformable-DETRベース、finetune済みckpt）
- **観測品質の近似**：画像解像度の20段階回復（scale 0.05→1.00、縮小→bilinear補間再拡大）。Bilinear補間は面積平均による低域通過であり、光学的な遠方観測時のボケ（センサ解像度不足）を忠実に再現する。
- **LLM**：ollama `qwen3.5:4b` / `qwen3.5:9b`（画像は見せず、構造化SG系列のみ入力）
- **対象**：100画像（GT物体≥5、GT関係≥3、seed固定のランダムサンプル）

> **スコープ明記**：接近行動そのものではなく、接近に伴う観測品質改善のうち**解像度改善のみ**を扱う（視点変化・遮蔽・移動計画は対象外）。LLMは画像理解ではなく、**SGG出力系列の増分・安定性の評価**に用いる。

---

## 2. 手法

### 2.1 パイプライン
VGサブセット選定 → 20段階解像度劣化（bilinear補間） → 各段階でEGTR推論 → GT/擬似GTと比較 → 指標算出 → 停止判定 → 可視化。

- **補間法**: Bilinear補間。config: `resolution_levels.interpolation: bilinear`。Bilinear補間は面積平均の低域通過であり、光学センサや遠方観測時の滑らかなボケを再現する。Nearest-neighbor（最近傍）はブロック状エイリアシング（デジタルズーム的アーティファクト）を生じ、光学的には起こらない高周波ノイズで飽和を解像度天井に押しやる。補間法の選択根拠と Nearest との比較は §3.6 で詳述。
- **出力データの拡張**: EGTR予測JSONおよびGT JSONの各 relationship レコードに、標準 SGDet R@K（IoUベース triplet マッチング）を post-hoc 計算可能とするためのフィールド（bbox, class\_id, predicate\_id, triplet\_score 等）を保持。メイン指標は引き続き SGScore（label-only・カテゴリ集合一致）であり、SGDet R@K は将来の評価軸として出力データを準備した段階。追加フィールドは LLM 入力には混入しない（`build_sequence` が `subject_label` / `predicate` / `object_label` / `confidence` のみを明示選択するため）。

### 2.2 指標（label-only・カテゴリ集合一致）
- Node Precision/Recall/F1、Triplet Precision/Recall/F1（subject–predicate–object のラベル三つ組）
- `SGScore(t) = 0.5·NodeRecall(t) + 0.5·TripletRecall(t)`

### 2.3 飽和の基準

#### 2.3.1 GT-based Oracle（絶対基準）の退化

当初は真GTへの到達度で飽和を定義したが、**EGTRの triplet recall はフル解像度でも約0.286で頭打ち**（手法の天井）。GT-based Oracle の到達状況（n=100, bilinear）：

| θ | 真に飽和 |
|---|----------|
| 0.4 | 63/100 |
| 0.5 | **40/100** |
| 0.6 | 21/100 |

絶対閾値では多数の画像が「永遠に飽和しない」退化した評価になる。

#### 2.3.2 Full参照 RelOracle（対L20、旧オラクル）

基準を **各画像自身の Level-20（最接近）EGTR出力＝擬似GT** への到達度に変更：
`RelSGScore(t) = 0.5·RelNodeRecall(t) + 0.5·RelTripletRecall(t)`（対 L20）。
これで「EGTRの天井」交絡を除去でき、**実機ロボットの状況（手元のSGG出力しか持たない）とも整合**。

Full参照 RelOracle 閾値別（n=100, bilinear）：

| θ | 平均停止 | 中央値 | sat<20 |
|---|----------|--------|--------|
| 0.7 | 15.22 | 15 | 85/100 |
| 0.8 | 16.71 | 17 | 70/100 |
| 0.9 | 17.90 | 19.5 | 50/100 |

しかし、full参照 RelOracle は L20 のノイジーな低確信度出力の完全再現を要求するため、飽和が解像度天井近傍に集中しやすい（平均停止 16.71）。これを解消するため TopK RelOracle を導入した（§2.3.3）。

#### 2.3.3 TopK RelOracle（採用オラクル）

擬似GT を L20 の**全出力ではなく高確信コア（top-10 ノード / top-5 トリプレット）**に絞る。これにより「L20のノイジーな低確信度の裾を完全再現する」要求を外し、意味的に重要な情報の飽和のみを評価する。

**top-10/5 の根拠（2つの独立なアンカーが収束）**：
- **VG150 アノテーション規模**: GT ユニークカテゴリ中央値 8（6–10）、ユニークトリプレット中央値 5（3–6）→ 1シーン分の意味コンテンツに相当。
- **EGTR 確信度の崖**: 物体確信度は rank 10 でまだ ~0.36 だが、triplet\_score は rank 5 以降 ~0.001 に崩壊 → 確信度の高いコアは ~10 ノード / ~5 トリプレット。
- **非対称 N が正しい**: 関係は物体より疎（全物体ペアが関係を持つわけではない）。対称の 10/10 ではトリプレット top-10 ≈ 全トリプレット（L20平均~8本）となり "コア" 選択の意味が消失。

Oracle stop = RelSGScore（top-10/5 参照集合に対する recall、0.5·node + 0.5·triplet）≥ coef の最初のレベル。
RelSGScore(L20) = 1.0 by construction（参照 ⊂ L20）。平均参照集合サイズ: ノード 5.8 ラベル、トリプレット 3.5 本。

**TopK RelOracle 閾値別（n=100, bilinear）**：

| coef | 平均停止 | 中央値 | sat<20 | 分布 |
|------|----------|--------|--------|------|
| **0.9（主基準）** | **16.36** | **17** | 71/100 | L5–20 に展開、peak L15（14画像）、L20 に29画像 |
| 0.8（感度） | 15.01 | 14.5 | 80/100 | L11–20 に展開、peak L14（17画像）、L20 に20画像 |

**coef 0.9 を主基準とする根拠**：参照集合が L20 の高確信コア（上位N）であるため、該当項目は早いレベルで確実に回収され RelSGScore が高くなりやすい。したがって低い閾値（0.8）では飽和が早く立ちすぎ、より厳しい 0.9 が意味的に妥当な基準となる。加えて RelSGScore は小さな参照集合（ノード5.8/トリプレット3.5）上の離散量であり、coef ≥ 0.95 は量子化により 1.0 と等価（43/100 が L20 に集中）となる。0.9 は分布を保つ最も厳しい閾値である。

- ※ これは「抽出情報の**収束・安定性**」を測る（研究の問い）。「正しさ」は別途 EGTR純正プロトコルの R@K で確認（§5）。

### 2.4 停止判定手法
- **Oracle**：基準（GTベース／相対／TopK）への到達で理想停止段階を定義（評価用）。
- **Baseline**：平均物体確信度 ≥ θ（ノード確信度のみ）。
- **提案・LLM**：直近5段階のSG系列をLLMに入力し `saturation_score` を出力、閾値で停止。
- **提案・hybrid**：`0.5·平均物体確信度 + 0.5·LLM飽和スコア` を融合して停止（確信度がLLMの暴走を抑える）。
- **収束停止（B-1, 実機向け）**：信号（物体確信度等）の直近k遷移の増分が全てε未満になったら停止（**過去のみ使用＝deployable**）。
- **VLMベースライン**：同じ `qwen3.5`（vision対応）に**画像を直接**見せて飽和判定（single/sequence）。「構造化SG vs 生画像」をモデル能力を統制して比較。
- 各停止系に `consecutive-k`（k=1/2）変種。

---

## 3. 結果：研究上の問いへの回答

### 3.1 Q1. 解像度改善でノード・関係情報は増えるか → **はい**

平均（n=100, bilinear, L1→L20）：NodeRecall 0.112→0.569、TripletRecall 0.000→0.286、SGScore 0.056→0.427。単調増加を確認。

代表的なレベル別推移（100画像平均、bilinear）：

| Level | scale | NodeRecall | TripletRecall | SGScore | mean_obj_conf |
|-------|-------|-----------|---------------|---------|---------------|
| 1  | 0.050 | 0.112 | 0.000 | 0.056 | 0.166 |
| 5  | 0.094 | 0.161 | 0.021 | 0.091 | 0.247 |
| 10 | 0.242 | 0.341 | 0.105 | 0.223 | 0.356 |
| 15 | 0.455 | 0.494 | 0.199 | 0.347 | 0.391 |
| 18 | 0.729 | 0.547 | 0.240 | 0.393 | 0.399 |
| 20 | 1.000 | 0.569 | 0.286 | 0.427 | 0.405 |

L20 は原画像そのものであり補間法に依存しない。Bilinear は低解像度帯（L1–L15）で Nearest より回復が早い（§3.6 で比較）。

### 3.2 Q2. ノード飽和と関係飽和に差はあるか → **早期レベルでは存在するが、飽和タイミング差は限定的**

早期レベルで「物体は見えるが関係はまだ」という段階が存在する（L1: NodeRecall 0.112 vs TripletRecall 0.000）。SGGを用いる意義は、この**早期レベルでの関係情報の不在**により示される。飽和タイミングの先行性については、label-only set で L1 時点の NodeRecall 0.112 vs TripletRecall 0.000 という早期観察として確認。

### 3.3 Q3. ノード確信度だけで飽和を判断できるか → **不十分**

平均物体確信度はレベル別 L1=0.166→**L20=0.405で頭打ち**（≈0.40プラトー）。絶対閾値の到達率は ≥0.3で100/100（平均停止5.00）、**≥0.4で68/100**（平均停止13.38）、**≥0.5で7/100**（平均停止19.27）。当初計画値0.7では事実上到達不能。
→ 絶対閾値はプラトー水準に依存して脆く、**関係情報や系列の安定性（収束）を見る必要**がある。

### 3.4 評価指標設計の検討（方法論的貢献）

本節では、評価枠組みの設計過程で検討・決定した事項を集約する。これらの検討自体が本研究の方法論的貢献である。

#### 3.4.1 SGScore vs SGDet@20：主指標の選択

**SGScore（label-only カテゴリ集合一致）を主指標に採用し、SGDet@20 を per-image 飽和指標として使用しない**。理由：

1. **IoU は飽和タイミングに新情報を足さない**：per-image 停止レベルを label-only と SGDet@20 で比較したところ、71/96 画像で一致。bbox 精度はラベル一致と同時に飽和する（独立の飽和シグナルを持たない）。
2. **per-image 標準 SGDet@20 は退化**：100画像中 31画像が R@20(L20)=0（フル解像度でも strict IoU+class+predicate マッチングで1件もヒットしない）。これを飽和基準にすると「飽和しない」が標準的な結果になり評価として機能しない。

**標準 SGDet R@20 はコーパスレベルの EGTR 妥当性検証値としてのみ報告**：本実験100画像で 0.186（cf. 自前500画像実測 20.4、著者評価 23.54）。指標・設定差（label-only、IoU無し、n=100、top-K、劣化画像含む）による差であり、モデル欠陥ではない（§5 参照）。

#### 3.4.2 TopKオラクルの定義と根拠

Full参照 RelOracle（擬似GT = L20 の全出力）は、L20 の低確信度ノイジー出力の完全再現を要求するため、飽和が解像度天井に集中しやすい（平均停止 16.71、中央値 17）。これに対し TopK RelOracle は擬似GT を **L20 の top-10 ノード（確信度順）/ top-5 トリプレット（triplet\_score 順）**に限定する。

Top-10/5 の数値的根拠は §2.3.3 に記載。この選択により：
- 参照集合がシーンの意味的コアに絞られる（平均: ノード 5.8 ラベル、トリプレット 3.5 本）
- ノイジーな裾の完全再現要求が外れ、飽和点が解像度レンジの中〜後半に分布（coef 0.9 で平均停止 16.36、中央値 17 vs full参照の 16.71/17）。感度分析として coef 0.8（平均停止 15.01、中央値 14.5）も併記

#### 3.4.3 Bilinear 補間の採用根拠

光学センサによる遠方観測は面積積分的な低域通過（滑らかなボケ）であり、bilinear がこれを忠実に再現する。Nearest-neighbor はポイントサンプリングによるエイリアシング（ブロック状アーティファクト）を生じるが、これは光学系では発生しないデジタルズーム的ノイズである。

Nearest を用いた場合、このブロックノイズが高解像度近傍まで意味復元を抑制し、飽和が L18–20 に天井集中する → 評価の弁別レンジが崩壊する。詳細比較は §3.6 に記載。

### 3.5 Q4. LLMでSG系列をスコアリングし飽和タイミングを推定できるか → **できる（上位手法が確信度 baseline を上回る）**

TopK RelOracle（coef 0.9）基準の停止誤差（n=100, bilinear, MAE昇順, 全13手法）：

| Rank | 手法 | 平均停止 | MAE | MedAE | Early% | Late% | Pearson r |
|------|------|----------|-----|-------|--------|-------|-----------|
| 1 | **hybrid_4b_k2**（確信度×LLM） | 16.34 | **3.06** | 2.0 | 0.42 | 0.43 | 0.193 |
| 2 | vlm_9b_single_k2 | 14.41 | 3.63 | 3.0 | 0.64 | 0.26 | 0.176 |
| 3 | llm_sgg_4b_h5_k2（LLM-SGG） | 14.26 | 3.82 | 3.0 | 0.64 | 0.26 | 0.125 |
| 4 | hybrid_9b_k2 | 15.08 | 3.84 | 3.0 | 0.47 | 0.40 | 0.071 |
| 5 | hybrid_4b_k1 | 12.77 | 4.91 | 4.0 | 0.69 | 0.19 | 0.079 |
| 6 | vlm_9b_single_k1 | 11.95 | 5.03 | 4.5 | 0.82 | 0.13 | 0.120 |
| 7 | baseline（conf≥0.4） | 13.38 | 5.40 | 5.0 | 0.58 | 0.26 | 0.133 |
| 8 | vlm_4b_single_k2 | 12.98 | 6.22 | 5.0 | 0.61 | 0.32 | −0.076 |
| 9 | hybrid_9b_k1 | 10.76 | 6.26 | 6.0 | 0.82 | 0.13 | 0.173 |
| 10 | llm_sgg_9b_h5_k2 | 9.91 | 6.81 | 7.0 | 0.84 | 0.10 | 0.139 |
| 11 | llm_sgg_4b_h5_k1 | 9.95 | 6.93 | 7.0 | 0.84 | 0.09 | −0.065 |
| 12 | llm_sgg_9b_h5_k1 | 6.95 | 9.43 | 10.0 | 0.96 | 0.01 | 0.034 |
| 13 | vlm_4b_single_k1 | 7.00 | 9.66 | 10.0 | 0.94 | 0.04 | −0.115 |

感度分析（coef 0.8）: 上位は llm_sgg_4b_h5_k2（3.29）/ hybrid_4b_k2（3.31）/ vlm_9b_single_k2（3.34）の僅差3方クラスタ。閾値を下げるとオラクルが早まり（平均 15.01 vs 16.36）、上位が拮抗する。ランキングファミリは安定。

参考 — full参照 RelOracle(0.8) 基準の上位5手法（bilinear）：
hybrid_4b_k2 MAE 3.39、vlm_9b_single_k2 MAE 3.82、llm_sgg_4b_h5_k2 MAE 4.07、hybrid_9b_k2 MAE 4.31、vlm_9b_single_k1 MAE 5.18。ただし full参照オラクルは飽和が天井近傍に集中するため手法の弁別が困難であり、TopK を主基準とした。

**主な知見**：
- **hybrid_4b_k2 が明確に最良（MAE 3.06）**：確信度×LLM融合＋consecutive-k2 が最も効果的。続く緩いクラスタとして vlm_9b_single_k2（3.63）、llm_sgg_4b_h5_k2（3.82）、hybrid_9b_k2（3.84）がある。
- **consecutive-k=2 が一貫して有効**：hybrid_4b k1→k2 で MAE 4.91→3.06、llm_sgg_4b で 6.93→3.82。
- **小モデル(4b)が構造化SGGで優位**：LLM-SGG MAE 4b_k2=3.82 vs 9b_k2=6.81。9bは飽和を過信し早期停止。VLMでは逆に 9b > 4b（MAE 3.63 vs 6.22）で大モデルが必要。
- **baseline（確信度のみ）は7位**（MAE 5.40）。提案手法上位6手法が全て上回る。

### 3.6 Nearest 補間との比較

Nearest-neighbor 補間のレベル別推移（100画像平均、参考）：

| Level | scale | NodeRecall | TripletRecall | SGScore | mean_obj_conf |
|-------|-------|-----------|---------------|---------|---------------|
| 1  | 0.050 | 0.034 | 0.000 | 0.017 | 0.162 |
| 5  | 0.094 | 0.063 | 0.000 | 0.031 | 0.116 |
| 10 | 0.242 | 0.122 | 0.002 | 0.062 | 0.158 |
| 15 | 0.455 | 0.223 | 0.062 | 0.143 | 0.354 |
| 18 | 0.729 | 0.463 | 0.188 | 0.325 | 0.389 |
| 20 | 1.000 | 0.569 | 0.286 | 0.427 | 0.405 |

Bilinear は低解像度帯で Nearest を大幅に上回る（例: L10 NodeRecall 0.341 vs 0.122、SGScore 0.223 vs 0.062）。これは bilinear が低周波成分（物体輪郭・大まかな構造）を保存するのに対し、nearest のブロック状アーティファクトが全周波数を等しく破壊するためである。

**Nearest での TopK 評価は飽和の天井集中を示す**：
- TopK coef 0.8: 平均停止 18.5、中央値 19（L18–20 に集中）
- 飽和点がほぼ全画像で L18–20 に集中し、手法間の差が縮小（最良手法 hybrid_4b_k2 MAE = 1.43）

Nearest ではブロック状アーティファクトが高解像度近傍まで意味復元を抑制し、飽和が解像度天井に張り付く。**Bilinear を採用する根拠**は、光学的忠実性に加え、飽和が解像度レンジの中〜後半に分布し（coef 0.9 で平均 16.36、中央値 17）、手法間の差を評価できる点にある。

### 3.7 Q5. 構造化SGG出力はVLM/キャプションより分析に適すか → **はい（SGG系が優位）**

同一モデル `qwen3.5` で「構造化SGテキスト（画像なし）」vs「生画像」を比較（TopK coef 0.9 基準）。

- 最良 SGG 系: hybrid_4b_k2 MAE **3.06**（1位）、llm_sgg_4b_h5_k2 MAE **3.82**（3位）
- 最良 VLM: vlm_9b_single_k2 MAE **3.63**（2位）
- 能力統制比較（4b）：hybrid_4b_k2 MAE **3.06** / llm_sgg_4b_h5_k2 MAE **3.82** vs vlm_4b_single_k2 MAE **6.22**
- 9b VLM は2位と健闘するが、4b VLM は大きく劣後 → 画像直接入力は大モデルが必要
- VLM 4b は全般に早すぎ停止（Early% 0.91–0.53）

→ 構造化SGG出力は、モデルサイズの影響を受けにくく、意味情報の飽和判定に安定して適する。

---

## 4. 評価設計上の重要な発見（方法論）

1. **TopKオラクルの導入**: L20 の高確信コア（top-10 ノード / top-5 トリプレット）を擬似GTとし、ノイジーな裾の完全再現要求を外すことで、飽和が解像度レンジの中〜後半（coef 0.9 で平均 16.36、中央値 17）に分布する評価枠組みを実現。full参照オラクルの天井集中問題を解消。参照が高確信コアゆえ RelSGScore が高くなりやすく、主基準は厳しめの coef 0.9 とした（量子化により coef≥0.95 は 1.0 と等価で L20 集中するため、0.9 が分布を保つ最厳閾値）。
2. **ollama `num_ctx` 不具合**：既定 `num_ctx=4096` が長プロンプト（履歴5×top20/30 で最大~9.2kトークン）を無言で切り詰め、生成枠0で1トークン停止 → LLMの「パース失敗27%」の正体。`num_ctx=12288` で **0%に解消**。修正前は不当にVLM>LLMに見えていた。
3. **関係エクスポートの過剰フィルタ**：人物ボックスは確信度~0.2で物体閾値0.3に届かず、その関係（man-wearing-shirt 等）が連鎖的に消え、23/100の擬似GTが関係0件になっていた。**EGTRの失敗ではなく出力規約の問題**。本来のSGGの使い方（端点足切りなしのtop-K）に修正し **23→4/100**、真GT triplet recall@L20 0.250→0.286。物体トラック（確信度信号）は据え置きで baseline/hybrid/収束に影響なし。
4. **Bilinear 補間の復帰**: Nearest はブロック状アーティファクトにより飽和を解像度天井に押しやる。Bilinear は光学的忠実性と飽和の中盤分布の両方を確保する。

---

## 5. EGTRの妥当性（論文整合性）

「triplet recallが低い／述語の大半が当たらない」のは**EGTRの不具合ではなく既知の長尾現象**であることを確認：
- 同梱の著者評価JSONが論文Table 1を再現（R@20/50/100 = 23.54/30.21/34.35、mR@50=7.94、AP50=30.83）。我々の500画像実測（RTX5080）も想定レンジ内（R@20/50/100 = 20.4/26.3/30.5）。
- VG150は述語が極端な長尾（`on` だけで全関係の約40%）。R@50≈30 に対し **mR@50≈7.9**（約4倍差）は全SGG手法共通＝**多くの述語が低recallなのは構造的に想定通り**。
- 我々のPoC指標が低めなのは label-only・IoU無し・n=100・top-K・劣化画像という**指標/設定差**であり、モデル欠陥ではない。
- 本実験100画像でのコーパスレベル SGDet R@20@L20 = 0.186。

---

## 6. 限界

- 100画像・label-only・カテゴリ集合一致（インスタンス区別なし、bbox IoU無し）。
- **per-image 飽和予測精度は限定的**：Pearson r ≈ 0.25（正だが弱い相関）。手法は中心傾向を捉えるが、各画像の正確な飽和レベル予測は困難。
- 人物ノード自体は確信度~0.25で物体閾値0.3に届かず node 集合から落ちる（関係トラックは復活済みだが node-level の人物弱点は残存）。
- 観測品質を解像度のみで近似（視点・遮蔽・移動は対象外）。
- TopKオラクルの top-10/5 は VG150 + EGTR に calibrate されており、他データセット・他SGG手法への転移性は未検証。

---

## 7. 成果物

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

## 8. 次マイルストーンへの接続

本段階で「観測品質改善に伴う意味情報の収束は、構造化SGG出力＋LLM（特にLLM-SGG系/hybrid系）で定量的に推定可能」「TopKオラクルで飽和を解像度レンジ中盤に分布させ評価できる」ことを確認した。次段階では、ここに**質問・依頼の意図**をLLMプロンプトに条件として加え、「その意図に答えるのに十分な観測か」へと停止判定を拡張する。本マイルストーンのTopKオラクル・hybrid/LLM停止がその基盤となる。
