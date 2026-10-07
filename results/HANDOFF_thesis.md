# 博論 Ch.3（FlowChain-CNF）への反映事項：引き継ぎ

作成 2026-10-07。この文書だけで内容が分かるように書いてある（数値はすべて本文中にある）。
元の論文：Neurocomputing NEUCOM-D-26-08804（受理済み、論文側は修正しない）。修正は博論にだけ反映する。
すべて既存の checkpoint で測り直したもので、学習はしていない。コードと生データは、元の PC の keepflow リポジトリのブランチ `thesis/solver-gt-fix`（未 push）にある。

---

## 1. 何が問題だったか（要点）

1. **ODE solver**：論文は DOPRI5（atol=rtol=1e-2）としているが、ETH log-prob の列（主表、更新の行、ablation の表）だけは、実装の不具合で RK4（9 ステップ）で計算されていた。Sinkhorn、計算時間、solver 比較の表は DOPRI5 で正しい。
2. **solver 比較の表の RK4・100 の ETH（0.19）**：古い別の実行の値が残っていたもので、RK4・100 の結果ではなかった。
3. **Fork/Branch の GT**：Sinkhorn の基準にした GMM の std が 0.09 で、生成過程（std 0.3。観測点にもノイズがあり、評価座標は最後の観測点が原点）と合っていなかった。生成過程の厳密な事後分布に置き換えた。
4. **Social-GAN**：再現実装が公表値を再現できない（ETH の ADE/FDE が原論文より良く出る）。密度評価の表からは行ごと外す。

**どれも結論は変わらない。**

## 2. 博論で差し替える表

### 主表（密度推定の精度と計算時間）
- Social-GAN の行は削除する。
- 計算時間と Unicycle の列は論文の値のまま。
- Fork/Branch は新しい GT、ETH log-prob は DOPRI5（FlowChain-CNF は 3 seed の平均 ± 標準偏差）。
- 太字は Pred at t の行の中での最良。

| Method | Time [ms] | Fork ↓ | Branch ↓ | Unicycle ↓ | ETH log-prob ↑ |
|---|---|---|---|---|---|
| Trajectron++ | 62 | 1.97 | 1.70 | 0.758 | −1.660 |
| MID | 23534 | 1.72 | **0.89** | **0.678** | −0.955 |
| SingularTrajectory | 6966 | 5.34 | 4.57 | 6.266 | −1.359 |
| MoFlow | 39596 | 3.80 | 3.16 | 1.733 | −3.946 |
| FlowChain-NF (conference) Pred at t | **44** | 1.72 | 1.53 | 0.903 | — |
| FlowChain-CNF Pred at t | 195 | **1.41** | 1.27 | 0.815 | **−0.770 ± 0.099** |
| 　Update from t+1 | 0.6 | 0.94 | 1.13 | 0.847 | −0.711 ± 0.167 |
| 　Update from t+3 | 0.6 | 0.59 | 1.06 | 0.844 | −1.178 ± 0.215 |
| 　Update from t+5 | 0.5 | 0.63 | 1.09 | 1.125 | −2.338 ± 0.569 |

参考（論文の値）：CNF の Fork/Branch は 1.40/1.19、更新 1.03/0.62/0.61（Fork）、1.43/1.07/1.17（Branch）。ETH log-prob は −0.741 ± 0.104、更新 −0.709/−1.141/−2.292。

### ablation の表（3 seed の平均 ± 標本標準偏差）
ADE/FDE は論文の値のまま。

| Method | Fork ↓ | Branch ↓ | Unicycle ↓ | ETH log-prob ↑ |
|---|---|---|---|---|
| FlowChain-CNF (full) | 1.375 ± 0.090 | **1.268 ± 0.036** | 0.830 ± 0.034 | **−0.770 ± 0.099** |
| w/o consistent rectified flow | 1.469 ± 0.145 | 1.396 ± 0.030 | **0.816 ± 0.016** | −0.808 ± 0.229 |
| w/o CIF | **1.242 ± 0.068** | 1.371 ± 0.018 | 0.827 ± 0.017 | −0.847 ± 0.278 |
| w/o CIFM | Failed | Failed | Failed | Failed |

### solver 比較の表
計算時間と ETH log-prob は論文の値のまま。ただし RK4・100 の ETH は 0.19 → 0.95。Fork は新しい GT。

| Solver | #step | Time | Fork | ETH log-prob |
|---|---|---|---|---|
| Dormand–Prince 5 | – | 263.1 | 1.41 | 0.93 |
| Tsitouras 5 | – | 505.6 | 1.41 | 0.97 |
| Euler | 1 | 45.5 | 1.93 | 0.81 |
| | 10 | 383.7 | 1.49 | 0.91 |
| | 100 | 3770.4 | 1.42 | 0.95 |
| Runge–Kutta 4 | 1 | 146.02 | 1.62 | 0.63 |
| | 10 | 1408.1 | 1.33 | 0.97 |
| | 100 | 15153.7 | 1.38 | 0.95 |

注意：この表の ETH log-prob はグリッドを使った評価で、評価コードが論文以降に少し変わっているため、測り直すと ±0.1 程度ずれる（DOPRI5 0.95、RK4・10 0.88）。RK4・100 の 0.95 もこの誤差を含む。

## 3. 結論（変わらないことの確認）
- Fork は CNF が最良。Branch は MID が最良で CNF が 2 位（論文と同じ）。ETH log-prob は CNF が最良。
- GT を替えても、手法の順位は Fork・Branch とも変わらない。
- **更新の効果は、新しい GT の方がはっきりする**。論文では Branch の t+1 が Pred at t より悪かったが、新しい GT では t+1・t+3・t+5 のすべてで Pred at t より良い。
- ablation：Fork は w/o CIF、Branch と ETH は full が最良（論文と同じ）。
- solver：DOPRI5 が精度と速度の釣り合いで最良、という結論は同じ。

## 4. 本文の修正
- 「MID の最良予算を 0.075 nats 上回る」→ **0.046 nats**（優位は保たれる）。
- 主表のキャプションに追加（Social-GAN を外した理由）：
  > Social-GAN is omitted from this table because our re-implementation did not reproduce its published results; its trajectory accuracy is quoted from the original paper.
- 付録（評価プロトコル）の GT と solver の記述を差し替え：
  > For the Fork and Branch subsets, the ground-truth density is the exact posterior of the generating process given the observed window: a Gaussian mixture over the branches whose covariance accounts for the noise shared through the last observed position. For the update rows it is further conditioned on the revealed future positions. Log-probabilities of FlowChain-CNF are evaluated with the Dormand–Prince 5 solver.
- 付録の「基底分布の分散 σ² は log 空間で最適化される」は、FlowChain-CNF については誤り。flow matching の目的関数が σ に勾配を流さないため、初期値（実データ 0.1、unicycle 0.03）のまま固定される。博論では「固定」と書く。
- SimTrajDensity の概要図は描き直さない（色のスケールを示していない図なので、分かりやすさを優先する）。

## 5. 新規内容：予測密度の calibration（HPD coverage）

### 方法
- 各テストケースで予測分布から N=1000 サンプルを引き、サンプルと GT の log p を同じ関数で評価する。u = （GT より密度が高いサンプルの割合）＋ 0.5 ×（同値の割合）。calibrated なら u ~ U(0,1)。
- coverage(α) = P(u ≤ α)、α = 0.05, 0.10, …, 0.95。ECE = mean_α |coverage − α|、符号付き誤差 = mean_α (coverage − α)。**負なら過信、正なら過小信頼（予測が広すぎる）**。
- 95% CI はテストケースの bootstrap（B=2000）。ECE は |·| のため bootstrap で上に偏るので、比較には符号付き誤差の CI を使う。
- FlowChain-CNF：各ステップの marginal p(x_{t+n}|o_t)（論文の log-prob と同じ関数）、DOPRI5。CNF の joint は、軌跡全体が (z₀, w) の決定的な関数なので 24 次元の密度を持たず、定義できない。
- FlowChain-NF：joint（1 ステップ条件付き密度の積）で、その密度に対応する祖先サンプリングでサンプルを引く。参考として NF 自身のサンプラーの結果も出している（この密度とは別の分布をサンプルする）。
- Trajectron++：各ステップの GMM から厳密にサンプル。
- テストケース：実データは 1 エージェントにつき 1 snapshot（ETH 44、Hotel 122、Univ 722、Zara1 142、Zara2 189）。SDD は等間隔に 500。合成データは各 1000。seed 1 のみ。NF と Trajectron++ は SDD の checkpoint がない。
- 実装の検証：真の事後分布からサンプルし、真の密度で評価すると ECE 0.002〜0.009 で、対角線に乗る。

### 主評価（ECE / 符号付き誤差）

| データ | FlowChain-CNF | FlowChain-NF（joint） | Trajectron++ |
|---|---|---|---|
| ETH | 0.016 / +0.013 | 0.362 / −0.362 | 0.199 / −0.199 |
| Hotel | 0.020 / +0.013 | 0.212 / −0.160 | 0.065 / −0.060 |
| Univ | 0.028 / +0.003 | 0.213 / +0.162 | 0.209 / −0.208 |
| Zara1 | 0.067 / +0.066 | 0.412 / +0.412 | 0.123 / −0.123 |
| Zara2 | 0.059 / +0.057 | 0.371 / +0.371 | 0.132 / −0.132 |
| SDD | 0.066 / +0.060 | — | — |
| Unicycle | 0.093 / −0.093 | 0.335 / +0.335 | 0.304 / −0.304 |
| Fork | 0.076 / −0.076 | 0.033 / +0.033 | 0.200 / +0.200 |
| Branch | 0.121 / −0.121 | 0.048 / +0.048 | 0.189 / +0.189 |

CNF の符号付き誤差の 95% CI：ETH [−0.030, +0.050]、Hotel [−0.024, +0.049]、Univ [−0.011, +0.017] は 0 を含む。Zara1・Zara2・SDD は正側で 0 を含まない。

### 更新経路（ETH、150 snapshot × k=1..6）
論文の stratum 表と同じ snapshot を使う。更新後の密度はキャッシュした粒子の上にしかないので、100×100 のグリッドにラスタ化し、GT のセルまでの累積質量を u とした。

| | oracle（フル再計算） | 更新前 | 更新後 |
|---|---|---|---|
| ESS/N ≥ 0.05 | 0.055 / +0.008 | 0.030 / +0.002 | 0.068 / +0.027（CI [−0.011, +0.066]） |
| ESS/N < 0.05 | 0.068 / −0.008 | 0.368 / −0.368 | 0.154 / −0.133（CI [−0.199, −0.068]） |

- ESS/N < 0.05 の更新後は、31% のケースで u ≥ 0.99（GT がほぼ台の外）。
- 図の α ≈ 0.3〜0.5 の段差は、ラスタ化で密度に上限をかけることによる同値（質量の平均 52%）の影響なので、その部分の形は解釈しない。
- strata 別：dense interaction の過信は更新前からのもの（更新前 −0.098、更新後 −0.065）。sharp acceleration は 4 snapshot しかなく判断できない。

### 言える主張
1. FlowChain-CNF の各ステップの予測密度は、実データで過信を示さない（ECE 0.016〜0.067、ずれは保守側のみ）。
2. 誤差は初期ステップに集中する（n=1 の ECE 0.14〜0.28、n とともに下がる）。向きは固定された基底 σ と整合する（実データでは σ=0.1 が広すぎて n=1 が過小信頼、合成データでは狭すぎて過信）。ただし σ を変えた再評価はしていないので、因果としては未検証。
3. FlowChain-CNF は、両方を評価した 8 データセットすべてで Trajectron++ より calibrated。
4. ESS/N ≥ 0.05 では、更新後も oracle と同程度に calibrated（更新前もすでに calibrated なので、calibration を改善するわけではない）。
5. ESS/N < 0.05 では更新後に過信に転じる（−0.13）。oracle は calibrated なので、ESS/N = 0.05 を再計算に戻す基準にする根拠になる。
6. FlowChain-NF の joint は calibrated ではなく（0.21〜0.41、シーンにより向きが違う）、NF 自身のサンプラーとも食い違う。論文が NF に log-prob を載せない理由を実測で裏付ける。

### 言えないこと
- CNF と NF の calibration の優劣（2 次元の marginal と 24 次元の joint で、検定の難しさが違う）。
- SDD での手法比較、seed に対する頑健性（seed 1 のみ）、strata 別の主張。

### 要旨の文案
> FlowChain-CNF の予測密度は実データで過信を示さず（ECE 0.016–0.067）、全データセットで Trajectron++ より calibrated だった。誤差は初期ステップに集中し、その向きは固定された基底分散と整合する。高速更新は ESS/N ≥ 0.05 では oracle と同程度に calibrated で、ESS/N < 0.05 では過信に転じる（符号付き誤差 −0.13）。

## 6. 図（元の PC の `results/calib_dopri5/` にある）
- `reliability.pdf`：データセットごとの reliability diagram（CNF、NF、Trajectron++、検証用 GT）
- `cnf_step_ece.pdf`：CNF のステップごとの ECE と符号付き誤差
- `update_reliability_ess.pdf` / `update_reliability_strata.pdf`：更新経路の ESS 別、strata 別
- `pit_hist.pdf`：u のヒストグラム
