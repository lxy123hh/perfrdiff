# Metric Calculation Spec (PerFRDiff Project)

## 1) Are These Formulas "Universal"?

Short answer: partially.

- FRRea (FID) is a widely used metric in generative modeling.
- FRCorr, FRdist, FRDiv, FRDvs, FRVar, FRSyn, and PMAFRG are task-specific metrics used in MAFRG/REACT-style evaluation.
- Therefore, the names are not globally universal like classification accuracy, and the exact implementation details (aggregation, scaling, neighbor protocol) must follow this project.

## 2) Model Outputs Used for Evaluation

In `evaluate_rewrite_weight_all.py`, the model generates:

- `evaluate_rewrite_weight_all.py` supports `--protocol {mafrg,pmafrg,both}`
  - `mafrg`: generic test split
  - `pmafrg`: person-specific test split
  - `both`: runs both protocols sequentially after one model load
- When `--compute_frrea` is enabled with `--protocol both`, rendered FID assets are stored under protocol-specific subdirectories of `frrea_dir`
- Emotion prediction tensor: `prediction_emotion`, shape `[N, K, T, D]`
  - `N`: number of test samples
  - `K`: number of stochastic generations per sample (default `K=10`)
  - `T`: sequence length (default `T=750`)
  - `D`: emotion feature dimension (default `D=25`)

For realism (FRRea/FID):

- `prediction_emotion` -> decode to 3DMM (`[N, K, T, 58]`) -> render fake frames
- Real frames are sampled from ground-truth listener video
- FID is computed between `fid/fake` and `fid/real`

## 3) Notation

- `\hat{y}_{n,k}`: predicted emotion sequence of sample `n`, candidate `k`, shape `[T, D]`
- `y_j`: ground-truth emotion sequence of sample `j`, shape `[T, D]`
- `A(n)`: neighbor index set for sample `n` from person-specific neighbor matrix
- `CCC(.,.)`: concordance correlation coefficient
- `DTW(.,.)`: dynamic time warping distance

## 4) Metric Formulas (Project Implementation)

### 4.1 PMAFRG-FRCorr (FRCorr*)

For each `(n, k)`, compute CCC against all neighbors and keep the maximum:

`max_{j in A(n)} CCC(\hat{y}_{n,k}, y_j)`

Project aggregation (as implemented) is:

`FRCorr* = (1/N) * sum_{n=1..N} sum_{k=1..K} max_{j in A(n)} CCC(\hat{y}_{n,k}, y_j)`

Note: this implementation sums over `K` (not dividing by `K` again).

### 4.2 PMAFRG-FRdist

Weighted DTW over feature groups:

`d(\hat{y}, y) = (1/15)*DTW(AU[0:15]) + 1*DTW(VA[15:17]) + (1/8)*DTW(EXP[17:25])`

For each `(n, k)`, keep minimum neighbor distance:

`min_{j in A(n)} d(\hat{y}_{n,k}, y_j)`

Aggregation:

`FRdist = (1/N) * sum_{n=1..N} sum_{k=1..K} min_{j in A(n)} d(\hat{y}_{n,k}, y_j)`

### 4.3 FRDiv (implemented by `compute_s_mse`)

Within each sample `n`, flatten each candidate sequence to length `D' = T*D`, then compute pairwise squared Euclidean distance among `K` candidates:

`FRDiv_n = [ sum_{a,b} ||z_{n,a} - z_{n,b}||^2 ] / [ K*(K-1)*D' ]`

`FRDiv = (1/N) * sum_n FRDiv_n`

### 4.4 FRDvs

Fix candidate index `k`, compare different samples:

`FRDvs = [ sum_{k=1..K} sum_{n,m} ||z_{n,k} - z_{m,k}||^2 ] / [ N*(N-1)*K*D' ]`

### 4.5 FRVar

Temporal variance over frame dimension, then global mean:

`FRVar = mean_{n,k,d}( Var_t(\hat{y}_{n,k,t,d}) )`

### 4.6 FRSyn (TLCC-based)

For each pair `(\hat{y}_{n,k}, speaker_n)`, compute cross-correlation over lags in `[-(2*fps-1), ..., +(2*fps-1)]`.
Take lag at peak correlation and use absolute offset in frames.

Project aggregation returns mean absolute lag offset (lower is better):

`FRSyn = mean( |offset| )`

### 4.7 FRRea (FID)

`FRRea = FID( Dis(fake_frames), Dis(real_frames) )`

In code, this is computed by `pytorch-fid` on directories:

- `result_dir/fid/fake`
- `result_dir/fid/real`

## 5) Display Scaling Used in Table-style Output

In project print output for table-style values:

- `FRDiv(table) = FRDiv * 100`
- `FRDvs(table) = FRDvs * 100`
- `FRVar(table) = FRVar * 100`

This corresponds to reporting these three metrics under `x10^-2` display convention.

## 6) File Pointers (Current Project)

- `evaluate_rewrite_weight_all.py`
- `metric/FRC.py`
- `metric/FRD.py`
- `metric/S_MSE.py`
- `metric/FRDvs.py`
- `metric/FRVar.py`
- `metric/TLCC.py`
- `metric/FRRea.py`
- `utils/render.py`
