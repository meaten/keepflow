# 博論用の再計算：solver 設定の修正、fork/branch の GT 修正、測り直し（完了報告）

作成 2026-10-06。keepflow リポジトリ、作業ブランチ `thesis/solver-gt-fix`。パスはリポジトリのルートからの相対パス。
論文（NEUCOM-D-26-08804）は受理済み。ここでの修正は**博論（Ch.3）に反映するためのもの**。学習はしておらず、既存の checkpoint だけを使った。既存の結果ファイルは上書きしていない（新しい結果はすべて新しいタグで保存）。

---

## 0. 結論（博論で変えるもの）

| 項目 | 博論での扱い | 結論への影響 |
|---|---|---|
| ETH log-prob（主表と ablation の表） | 論文の値は RK4 で計算されていた。**dopri5 の値に差し替える** | なし（CNF が最良、ablation の順序も同じ） |
| solver 比較の表の RK4・100 の ETH | 0.19 → **0.95** | なし |
| Fork/Branch の Sinkhorn（主表・ablation の表・solver 比較の表） | **正しい GT（生成過程の厳密な事後分布）で測り直した値に差し替える** | 順位は旧 GT と同じ |
| Social-GAN | **密度評価の表から行ごと外す**（再現実装が公表値を再現できないため。ADE/FDE の表は引用値のまま） | — |
| 本文の「MID の最良予算を 0.075 nats 上回る」 | **0.046 nats** に | 優位は保たれる |

## 1. 調査結果

### solver 設定が効かなくなった経緯
- 論文の時期、solver の切り替え（`CNF_SOLVER`・`NUM_STEP`、dopri5 は atol=rtol=1e-2）は**未コミットの作業ツリーにしかなかった**。
- 2026-08-09 の e559dc7 は `git add` でファイルを個別に指定しており、`continuousflowchain_onestep.py` が漏れていた。
- 2026-08-10 06:15 JST、旧セッションが「検証していない変更を戻す」として `git checkout keepflow/models/traj/continuousflowchain_onestep.py` を実行し、そこで消えた。以後は常に RK4（区間あたり 10 点 = 9 ステップ）で動いていた。
- 根拠は旧セッションの記録（`~/.claude/projects/-home-maeda-univ-keepflow/a489ffc1-….jsonl`）と、8/9 にコミットされた `202604_Neurocomputing_FlowChain-CNF/results_E7_repro_details.md`。

### 論文の数値がどの solver か
- **dopri5**：Sinkhorn、計算時間、solver 比較の表（2 月の実行）。
- **RK4**：ETH log-prob の列（主表の −0.741 ± 0.104、更新の 3 行、ablation の ETH 列。8/10 08:47 以降の実行）。
- RK4 だったことは、今のコード（RK4 のみ）で seed 1 を計算し直し、当時の値が小数 4 桁まで一致したことで確認した（−0.7677 / −0.6338 / −1.338 / −2.9083）。

### solver 比較の表の RK4・100 の ETH（0.19）
- `~/.bash_history` によると、最初の実行（solver ごとのディレクトリで学習と評価）のあと checkpoint を差し替えて評価をやり直している。
- しかし `CIF_spline_rk4_100/eth/metrics.json` だけは 2/8 付のまま上書きされていなかった。表の値は、最初の実行の結果が残ったもの。実行時間 255 ms から見て、中身は dopri5。

### fork/branch の GT
- `process_data.py` で std が二重に掛かり、保存された GT の GMM は std 0.09 になっていた（生成器のノイズは 0.3）。さらに、評価座標がノイズの乗った最後の観測点を原点にしていることも反映されていなかった。
- 保存データ（`sim{fork,branch}_test.pkl`、2025-07-16 作成）は論文のすべての実行より前なので、論文の Sinkhorn は最初からこの GT に対する値。
- 正しい事後分布：平均 μ_k + Σd/L、共分散 s²I + (s²/L)·11ᵀ の GMM（L = 8、d は観測点の直線からのずれ）。更新ステップでは、観測済みの未来区間で各モード内の正規分布を条件付け、モードの重みもその尤度で更新する。実装は `keepflow/data/traj/sim_gt.py`。

### 評価コードの不具合（論文の値には影響なし）
- 8/9 の e559dc7 で、ラスタ化の関数が粒子のない格子点を NaN にするよう変わり、それ以降の Sinkhorn は壊れていた（全ステップ 0.026 などの無意味な値）。論文の Sinkhorn はすべてそれより前の実行。
- 今回、NaN にするのを図を描くときだけに限定して直した。
- 2 月から 7 月の間に評価側のラスタ化（RBF の平滑化 1 → 10 が有力）も変わっており、今の Sinkhorn は 7 月の値（`metrics_uniform`）と完全に一致し、2 月の値とは 0.03〜0.05 ずれる。予測そのもの（ADE/FDE）は 2 月と 8 桁まで一致する。

## 2. 再現確認（solver の読み込みを復元したあと）

| | 論文の元の値 | 今回 |
|---|---|---|
| ADE/FDE（simfork、dopri5） | 0.38324383 / 0.23876925 | 完全一致 |
| ADE/FDE（simfork、RK4・10） | 0.38343886 / 0.24113354 | 完全一致 |
| Sinkhorn Fork / Branch（dopri5、旧 GT） | 1.396 / 1.193（2 月） | 1.450 / 1.239（7 月の値と完全一致） |
| ablation の表の Sinkhorn（旧 GT、3 seed） | 1.387 ± 0.084 など | **全 6 欄で完全一致** |
| ETH log-prob（RK4） | −0.7677（seed 1） | 完全一致 |

## 3. GT の妥当性の確認（`scripts/validate_exact_gt.py`、`results/exact_gt/`）

生成器から 40 万本の軌跡を作り、各ステップの位置のヒストグラムと各 GT の距離を測った（12 ステップ平均）。

| | fork | branch |
|---|---|---|
| 厳密な EMD：同分布の別サンプル（ばらつきの目安） | 0.005 | 0.006 |
| 厳密な EMD：**新 GT** | **0.007** | **0.008** |
| 厳密な EMD：旧 GT | 0.465 | 0.460 |
| 偏り補正済み Sinkhorn divergence：新 GT / 旧 GT | 0.0004 / 0.134 | 0.0001 / 0.135 |

条件付けの確認：テスト GT の HPD の u は、観測済み区間 k = 0・1・3・5 のどれでも一様（ECE 0.002〜0.009、各ステップと joint の両方）。

注意：論文の Sinkhorn（reg 1.0）だけは、狭すぎる旧 GT の方が近く出る（fork 0.606 対 0.731）。正則化が強く、幅の狭い分布が有利になるという指標そのものの性質。上の偏りのない 2 つの指標では、新 GT が正しいことは明確。

## 4. 博論用の表（そのまま貼れる形）

### 主表（Social-GAN を除く）
計算時間と Unicycle は論文の値のまま。Fork/Branch は新 GT、ETH log-prob は dopri5（3 seed の平均 ± 標準偏差）。

| Method | Comp. Time [ms] | Fork | Branch | Unicycle | ETH log-prob |
|---|---|---|---|---|---|
| Trajectron++ | 62 | 1.97 | 1.70 | 0.758 | −1.660 |
| MID | 23534 | 1.72 | **0.89** | **0.678** | −0.955 |
| SingularTrajectory | 6966 | 5.34 | 4.57 | 6.266 | −1.359 |
| MoFlow | 39596 | 3.80 | 3.16 | 1.733 | −3.946 |
| FlowChain-NF（会議版）Pred at t | **44** | 1.72 | 1.53 | 0.903 | — |
| FlowChain-CNF Pred at t | 195 | **1.41** | 1.27 | 0.815 | **−0.770 ± 0.099** |
| 　Update from t+1 | 0.6 | 0.94 | 1.13 | 0.847 | −0.711 ± 0.167 |
| 　Update from t+3 | 0.6 | 0.59 | 1.06 | 0.844 | −1.178 ± 0.215 |
| 　Update from t+5 | 0.5 | 0.63 | 1.09 | 1.125 | −2.338 ± 0.569 |

旧 GT の値との対照（論文の値 → 新 GT）：Trajectron++ 2.02/1.67 → 1.97/1.70、MID 1.80/0.97 → 1.72/0.89、SingularTrajectory 5.52/5.04 → 5.34/4.57、MoFlow 3.65/3.08 → 3.80/3.16、NF 1.62/1.44 → 1.72/1.53、CNF 1.40/1.19 → 1.41/1.27、更新 1.03/0.62/0.61 → 0.94/0.59/0.63（Fork）、1.43/1.07/1.17 → 1.13/1.06/1.09（Branch）。

### ablation の表（3 seed の平均 ± 標準偏差、標本標準偏差）

| Arm | Fork | Branch | Unicycle | ETH log-prob（dopri5） |
|---|---|---|---|---|
| full | 1.375 ± 0.090 | **1.268 ± 0.036** | 0.830 ± 0.034 | **−0.770 ± 0.099** |
| w/o CRF | 1.469 ± 0.145 | 1.396 ± 0.030 | **0.816 ± 0.016** | −0.808 ± 0.229 |
| w/o CIF | **1.242 ± 0.068** | 1.371 ± 0.018 | 0.827 ± 0.017 | −0.847 ± 0.278 |

論文の値（旧 GT と RK4）：Fork 1.387 / 1.464 / 1.253、Branch 1.279 / 1.333 / 1.344、ETH −0.741 / −0.799 / −0.836。Unicycle、ADE/FDE は変わらない。

### solver 比較の表
計算時間は論文の値のまま。ETH log-prob の列は RK4・100 の欄だけ差し替える。

| Solver | #step | Fork（新 GT） | 論文の Fork | ETH log-prob |
|---|---|---|---|---|
| Dormand–Prince 5 | – | 1.41 | 1.40 | 0.93 |
| Tsitouras 5 | – | 1.41 | 1.44 | 0.97 |
| Euler | 1 | 1.93 | 2.00 | 0.81 |
| | 10 | 1.49 | 1.51 | 0.91 |
| | 100 | 1.42 | 1.43 | 0.95 |
| Runge–Kutta 4 | 1 | 1.62 | 1.77 | 0.63 |
| | 10 | 1.33 | 1.35 | 0.97 |
| | 100 | 1.38 | 1.40 | **0.95**（論文 0.19） |

補足：同じ条件で測り直すと DOPRI5 は 0.95（論文 0.93）、RK4・10 は 0.88（論文 0.97）で、評価コードの変更による ±0.1 程度のずれがある。RK4・100 の 0.95 もこの誤差を含む。

## 5. 順位
- 新 GT と旧 GT（今の値）で、手法の順位は Fork・Branch とも同じ。
- Social-GAN を除いた 6 手法では、Fork は CNF が最良、Branch は MID が最良で CNF が 2 位（論文と同じ）。
- 更新の効果は新 GT の方がはっきりする。Branch の t+1 は論文では Pred at t より悪かった（1.43 対 1.19）が、新 GT では t+1・t+3・t+5 のすべてで Pred at t（1.27）より良い。

## 6. 本文の修正案

付録（Evaluation Protocol）：
> For the Fork and Branch subsets, the ground-truth density is the exact posterior of the generating process given the observed window: a Gaussian mixture over the branches whose covariance accounts for the noise shared through the last observed position. For the update rows it is further conditioned on the revealed future positions. Log-probabilities of FlowChain-CNF are evaluated with the Dormand–Prince 5 solver.

主表のキャプション（Social-GAN を外した理由）：
> Social-GAN is omitted from this table because our re-implementation did not reproduce its published results; its trajectory accuracy is quoted from the original paper in Tables X and Y.

本文：「MID の最良予算を 0.075 nats 上回る」→ 0.046 nats。

## 7. calibration の評価（dopri5 で再実行済み、`results/calib_dopri5/`）

前回（`results/calib/`、CNF は RK4 の 9 ステップ）と同じテストケース・seed・N=1000 で、CNF を dopri5 で計算し直した。NF、Trajectron++、検証用の GT は solver に依存しないので、前回の結果をコピーして図に重ねている。更新経路のグリッド評価は、今回直したラスタ化（論文当時の挙動）を使っているので、solver の違いに加えてこの分も前回からずれる。

### 主評価（CNF、全ステップまとめ。ECE / 符号付き誤差、負＝過信）

| データ | RK4（前回） | dopri5 | dopri5 の符号付き誤差の 95% CI |
|---|---|---|---|
| ETH (44) | 0.015 / +0.010 | 0.016 / +0.013 | [−0.030, +0.050] |
| Hotel (122) | 0.016 / +0.004 | 0.020 / +0.013 | [−0.024, +0.049] |
| Univ (722) | 0.028 / +0.003 | 0.028 / +0.003 | [−0.011, +0.017] |
| Zara1 (142) | 0.060 / +0.058 | 0.067 / +0.066 | [+0.035, +0.095] |
| Zara2 (189) | 0.053 / +0.050 | 0.059 / +0.057 | [+0.028, +0.084] |
| SDD (500) | 0.062 / +0.056 | 0.066 / +0.060 | [+0.039, +0.081] |
| Unicycle | 0.084 / −0.084 | 0.093 / −0.093 | [−0.102, −0.084] |
| Fork | 0.128 / −0.128 | **0.076 / −0.076** | [−0.084, −0.069] |
| Branch | 0.137 / −0.137 | 0.121 / −0.121 | [−0.130, −0.112] |

- n=1 の ECE は 0.14〜0.28 で、RK4 のときとほぼ同じ。誤差が初期ステップに集中する傾向も変わらない。
- Trajectron++（ECE 0.065〜0.304）より CNF の ECE が小さいことは、両方を評価した 8 データセットすべてで変わらない。

### 更新経路（ETH、150 snapshot × k=1..6）

| | oracle | 更新前 | 更新後 |
|---|---|---|---|
| ESS/N ≥ 0.05 | 0.055 / +0.008 | 0.030 / +0.002 | 0.068 / +0.027（CI [−0.011, +0.066]） |
| ESS/N < 0.05 | 0.068 / −0.008 | 0.368 / −0.368 | **0.154 / −0.133**（CI [−0.199, −0.068]） |

- ESS/N < 0.05 の更新後は、RK4 のとき（−0.216）より過信が小さいが、CI は 0 を含まず、過信であることは変わらない。31% のケースで u ≥ 0.99（前回 45%）。
- このとき、更新後の分布の質量の平均 52% がラスタの上限で同値になる。図で α ≈ 0.3〜0.5 にある段差はこの影響なので、曲線のその部分の形は解釈しない。
- strata 別：dense interaction の過信は更新前からのもの（更新前 −0.098、更新後 −0.065、CI は 0 を含む）。sharp acceleration は 4 snapshot しかなく、CI が広すぎて判断できない。

### 主張への影響
`results/calib/HANDOFF.md` §4 の主張 1〜6 は、dopri5 でもすべて成り立つ。数値は上の表の値に差し替える。

| 主張 | dopri5 での裏付け |
|---|---|
| 1. 実データで過信を示さない | ECE 0.016〜0.067、ずれは保守側のみ |
| 2. 誤差は初期ステップに集中 | n=1 で 0.14〜0.28 |
| 3. Trajectron++ より calibrated | 8 データセットすべて |
| 4. ESS/N ≥ 0.05 で更新後も calibrated | +0.027、CI は 0 を含む |
| 5. ESS/N < 0.05 で過信 | −0.133、CI は 0 を含まない |
| 6. NF の joint は calibrated でない | solver に依存せず前回のまま |

一つだけ数値の記述が変わる。博論の要旨の「ESS/N < 0.05 では過信に転じる（符号付き誤差 −0.22）」は **−0.13** になる。

## 8. ファイル
- コード：`keepflow/models/traj/continuousflowchain_onestep.py`（solver 設定の読み込みを復元）、`keepflow/utils/default_params.py`（`CNF_ATOL`・`CNF_RTOL`）、`keepflow/data/traj/sim_gt.py`（新 GT）、`keepflow/visualization/traj/traj_visualizer.py`（新 GT の格子と NaN の修正）、`keepflow/metrics/traj/traj_metrics.py`（`emd_exact`）
- スクリプト：`scripts/run_thesis_remeasure.py`（測り直し 51 本のキュー）、`scripts/summarize_thesis_remeasure.py`（表の集計）、`scripts/validate_exact_gt.py`（GT の妥当性）
- 結果：`results/thesis_remeasure.md`（集計表）、`results/exact_gt/validate_{simfork,simbranch}.json`、各 `experiments/…/metrics_thesis_*.json`
- ログ：`logs/thesis_rerun/`

## 9. 未実施のこと
- `process_data.py` の `scale` のバグ修正：このファイルには以前からの未コミットの変更があり、データを作り直すとテストの軌跡自体が変わるため、手を付けていない。評価は `sim_gt.py` で行うので、結果には影響しない。
- 概要図 `maeda4.jpg` は描き直さない（ユーザーの判断。色のスケールを示していない図なので、分かりやすさを優先して今のままにする）。
