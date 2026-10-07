# 引き継ぎ：FlowChain の予測密度の calibration 評価（HPD coverage）

作成 2026-10-05。結果の数値と主張はこの文書だけで分かるように書いてある。パスは keepflow リポジトリのルートからの相対パス（元の環境では `/home/maeda/univ/keepflow`）。
リポジトリ：keepflow リポジトリ（branch `revision/neucom-d-26-08804`）。成果物とスクリプトは **未コミット**。

> **2026-10-07 追記**：ここの CNF の数値は RK4（9 ステップ）で計算したもの。dopri5 で計算し直した結果は `results/calib_dopri5/` と `results/HANDOFF_rerun.md` §7 にあり、博論ではそちらを使う。主張はすべて変わらない。

---

## 1. 何をしたか

博論 Ch.3（= Neurocomputing 論文 NEUCOM-D-26-08804、`202604_Neurocomputing_FlowChain-CNF/main.tex`）の評価設定（T_o=8、T_f=12、既存 checkpoint、seed 1）を使い、FlowChain-CNF と FlowChain-NF の予測密度が calibrated かを HPD coverage で測った。KDE や新しい密度推定器は使っていない。

### 定義
- テストケースごとに予測分布から N=1000 サンプルを引き、サンプルと GT の log p を **1 回のバッチ呼び出しで同じ関数** で評価する。こうすると CIF の潜在 w のサンプリングや Hutchinson のトレース推定ノイズが、サンプルと GT で同じ扱いになる。
- u = mean_i 1[log p(x_i) > log p(x*)] + 0.5·（同値の割合）。calibrated なら u ~ U(0,1)。
- coverage(α) = mean 1[u ≤ α]、α ∈ {0.05, …, 0.95}。
- ECE = mean_α |coverage − α|。符号付き誤差 = mean_α (coverage − α)。**負なら過信、正なら過小信頼（予測が広すぎる）**。
- CI はテストケースを単位にした bootstrap（B=2000）の 95% 区間。全ステップをまとめた行は、1 ケースの全ステップをまとめて復元抽出している。
- 注意：ECE は |·| を含むので bootstrap 分布が上に偏り、点推定が CI の下端付近に来ることがある。比較には符号付き誤差の CI を使うこと。

### 各手法の評価対象
| 手法 | config | 密度 | サンプル |
|---|---|---|---|
| `cnf` | `configs/traj/FlowMatchingChain_onestep/CIF_spline` | 各ステップの marginal p(x_{t+n}\|o_t)。論文表の log-prob と同じ経路（`analytic_log_prob_chain`、k=0）。n=1..12 ごとに u を出す | モデル自身のサンプラー |
| `nf` | `configs/traj/FlowChain/CIF_separate_cond_v` | joint：Σ_n log p(x_n\|x_{<n},o)（`log_prob_separate`）。u は 1 個（n=−1）。速度空間で評価するが、−2log Δt は u に影響しない | 祖先サンプリング（毎ステップ基底ノイズを引き直す）。この joint 密度に対応する分布 |
| `nf-native` | 同上 | 同上 | NF 自身のサンプラー（ノイズは最初の 1 回だけ）。**この密度とは別の分布**なので HPD 検定ではなく、ただの順位検定 |
| `trajectron` | `configs/traj/Trajectron` | 各ステップの位置空間 GMM（表と同じ `y_dist`）。GT の積分軸は修正済み（§5-2） | GMM から厳密にサンプル |
| `gt` | — | 真の事後分布（検証用） | 真の事後分布 |

- CNF の joint は出していない。軌跡全体が (z₀, w) の決定的な関数なので 24 次元の密度が存在しない。
- テストケース：実データは 1 エージェントにつき 1 snapshot（ETH 44、Hotel 122、Univ 722、Zara1 142、Zara2 189）。SDD は 1836 エージェントから等間隔に 500 件。SimTrajDensity は全 1000 件。
- **欠けているもの**：NF と Trajectron++ は SDD の checkpoint がない。seed 2・3 は未実行（checkpoint は `configs_seed/s2`, `s3` にある）。

## 2. 実装の検証（合格）
真の事後分布からサンプルし、真の密度で評価した ECE（各 1000 件）：
- unicycle（各ステップ、closed form）0.009
- fork：各ステップ 0.002、joint 0.009
- branch：各ステップ 0.003、joint 0.005

fork/branch では、保存されている GT GMM が誤っていたため（§5-1）、生成過程から導いた厳密な事後分布を使った。

## 3. 結果（ECE / 符号付き誤差。全文は `results/calib/summary.csv`）

| データ | CNF（全ステップまとめ） | NF（joint、祖先サンプリング） | NF-native | Trajectron++ |
|---|---|---|---|---|
| ETH | 0.015 / +0.010 | 0.362 / −0.362 | 0.482 / −0.482 | 0.199 / −0.199 |
| Hotel | 0.016 / +0.004 | 0.212 / −0.160 | 0.370 / −0.368 | 0.065 / −0.060 |
| Univ | 0.027 / +0.003 | 0.213 / +0.162 | 0.419 / −0.419 | 0.209 / −0.208 |
| Zara1 | 0.060 / +0.058 | 0.412 / +0.412 | 0.310 / −0.310 | 0.123 / −0.123 |
| Zara2 | 0.053 / +0.050 | 0.371 / +0.371 | 0.271 / −0.271 | 0.132 / −0.132 |
| SDD | 0.062 / +0.055 | — | — | — |
| Unicycle | 0.084 / −0.084 | 0.335 / +0.335 | 0.389 / −0.389 | 0.304 / −0.304 |
| Fork | 0.128 / −0.128 | 0.033 / +0.033 | 0.045 / −0.045 | 0.200 / +0.200 |
| Branch | 0.137 / −0.137 | 0.048 / +0.048 | 0.008 / −0.008 | 0.189 / +0.189 |

CNF の符号付き誤差の CI：ETH・Hotel・Univ は 0 を含む。Zara1・Zara2・SDD は 0 を含まない（正側）。

**CNF のステップ依存**（`cnf_step_ece.pdf`）：n=1 の ECE は 0.14–0.27 で、n とともにほぼ単調に下がる（実データは n ≥ 4 で概ね 0.02–0.07、合成データは n=12 でも 0.065–0.085）。符号は n=1 で、実データが正（+0.14〜+0.27）、合成データが負（−0.20 前後）。

**基底 σ の確認**：CNF の基底 σ は学習後も初期値のまま（ETH・Zara1・fork 0.1、unicycle 0.03）。flow matching の目的関数が σ に勾配を流さないため。付録の「σ² は log 空間で最適化」という記述は CNF については誤り。

### 更新経路（SNIS 更新、ETH）
- 論文の strata 表と同じ 150 snapshot × k=1..6（ゲートなし）。100×100 の共通グリッドにラスタ化し、GT セルまでの累積質量を u とした。
- 7650 行。5 snapshot は k の一部が欠ける。

| | oracle（フル再計算） | 更新前 | 更新後 |
|---|---|---|---|
| ESS/N ≥ 0.05 | 0.056 / +0.001 | 0.030 / −0.018 | 0.067 / +0.011 |
| ESS/N < 0.05 | 0.066 / −0.018 | 0.381 / −0.381 | 0.216 / −0.216 |

- ESS/N < 0.05 の更新後：45% のケースで u ≥ 0.99。
- strata 別（更新後の符号付き誤差）：dense interaction −0.093（更新前も −0.125 なので、更新前から引き継いだもの）。sharp acceleration −0.190（4 snapshot、CI が広すぎる）。

## 4. 言える主張と言えない主張

**言える**
1. CNF の各ステップの予測密度は、実データで過信を示さない（ECE 0.015–0.062。ずれは保守側＝広すぎる側に限られる）。
2. CNF の誤差は初期ステップに集中し、その向きは固定された基底 σ と整合する（実データでは σ=0.1 が広すぎ、合成データでは狭すぎ）。**因果としては未検証**（σ を変えた再評価はしていない）。
3. CNF は、両方を評価した全 8 データセットで Trajectron++ より ECE が小さい。
4. ESS/N ≥ 0.05 では、更新後も oracle と同程度に calibrated。論文の「log-prob の正の差は sharpness」という記述を、「calibration を保ったまま鋭くなる」と強められる。ただし更新前もすでに calibrated なので、calibration を改善しているわけではない。
5. ESS/N < 0.05 では過信に転じる（−0.216、oracle は −0.018）。ESS/N = 0.05 で再計算に戻す基準の根拠になる。
6. NF の joint 密度は calibrated ではなく（0.21–0.41、方向はシーンで異なる）、NF 自身のサンプラーとも食い違う。論文が NF に log-prob を載せない理由を実測で裏付ける。

**言えない**
- CNF と NF の calibration の優劣（2 次元の marginal と 24 次元の joint を比べることになり、同じ土俵にない）。
- SDD での手法比較、seed に対する頑健性（seed 1 のみ）、strata 別の主張。
- 更新経路でグリッド解像度より細かいこと：既存のラスタは粒子の 95 パーセンタイルで密度に上限をかけるため、低 ESS 帯の更新後の分布では質量の平均 38% がこの上限で同値になる。曲線の α ≈ 0.35–0.5 の段差はこの影響。

**論文・博論用の要旨（案）**
> FlowChain-CNF の予測密度は実データで過信を示さず（ECE 0.015–0.062）、全データセットで Trajectron++ より calibrated だった。誤差は初期ステップに集中し、その向きは固定された基底分散と整合する。高速更新は ESS/N ≥ 0.05 では oracle と同程度に calibrated で、ESS/N < 0.05 では過信に転じる（符号付き誤差 −0.22）。

## 5. 見つかった既存コードの問題（共有コードは未修正）
1. **fork/branch の GT GMM の std が 0.09。** `process_data.py` で、すでに std に等しい `scale` に std を掛けている。生成過程では観測点を含む全 20 点に std 0.3 のノイズが乗り、評価座標はノイズの乗った最後の観測点を原点にしている。厳密な事後分布は、平均 μ_k + Σd/8、共分散 s²I + (s²/8)·11ᵀ の GMM（d = 観測点の直線からのずれ）。`scripts/calib_hpd.py` の `GroundTruth` に実装済み。**論文の fork/branch の Sinkhorn 距離（GT として使用）に影響する。**
2. **Trajectron++ の GT 積分軸。** `integrate_samples` は第 2 軸で cumsum を取るため [S,bs,T,2] を想定しているが、`keepflow/models/traj/trajectron.py` の `predict` は [bs,T,2] を渡すので、時間ではなく x/y 方向に積分される。修正した積分での ETH 平均 GT log-prob は −1.632 で、論文の −1.660 とほぼ同じ。論文の数値は別の経路から出たとみられ、**表への影響は確認できていない**。
3. **`cfg.MODEL.FLOW.CNF_SOLVER` がモデルのコードから一度も読まれていない。** 常に rk4・10 ステップで解いており、付録（dopri5, atol=rtol=1e-2）と食い違う。`test.py --cnf_solver` も効いていない。NFE や solver 比較の表に影響する可能性がある。
4. **NF のサンプラーと密度の不一致。** NF 自身のサンプラーは x_n = f_n(x_{n−1}) で、ノイズは 1 回しか引かない。一方、密度は teacher-forced な 1 ステップ条件付きの積である。

## 6. ファイルと再現方法
- スクリプト（`scripts/`）：
  - `calib_hpd.py`：各ケースの u を計算。`--method {cnf,nf,nf-native,trajectron,gt} --dataset X --device cuda:N [--num_shards K --shard i] [--max_cases M]`
  - `calib_update_hpd.py`：更新経路の u
  - `calib_summary.py`：shard の結合、`summary.csv`、図の出力
- 結果（`results/calib/`）：
  - `u_{method}_{dataset}.csv`：列は case_id, scene, agent, t, n（joint は −1）, u, ess_frac, logp_gt, tie_frac, nonfinite_frac
  - `u_cnf-update_eth.csv`：source = oracle / pre / post、k、stratum、tie_mass、gt_outside を含む
  - 図：`reliability.pdf`、`cnf_step_ece.pdf`、`update_reliability_ess.pdf`、`update_reliability_strata.pdf`、`pit_hist.pdf`
- ログ：`logs/calib/`
- 実行環境：`pyenv exec python`（env `keepflow`）、TITAN RTX ×4。CNF は約 6–8 s/ケース、ほかは約 0.2 s/ケース。
- 祖先サンプリングで発散した NF のサンプル（log p = −∞ として計上）は最大 1.9%（fork の 1 ケース）で、平均はほぼ 0。

## 7. 未実施の作業
- seed 2・3 での再実行（CI と頑健性の確認のため）
- 基底 σ を変えた再評価（主張 2 の因果の検証）
- 上記の既存バグの修正、および論文の該当箇所（Sinkhorn の GT、付録の solver・σ の記述）の改訂
- NF の SDD checkpoint の学習（必要な場合のみ）
