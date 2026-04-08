# Time-Domain Adaptive Filters

Sample-by-sample (or block-based) adaptive filters operating directly on the discrete-time signal. All filters share the same goal: given a reference signal $x(n)$ and a desired signal $d(n)$, identify the unknown system $\mathbf{w}^*$ such that $e(n) = d(n) - \hat{d}(n) \to 0$.

**Common state:**
$$\mathbf{u}(n) = [x(n),\, x(n-1),\, \ldots,\, x(n-N+1)]^T \in \mathbb{R}^N$$
$$e(n) = d(n) - \mathbf{w}^T(n)\,\mathbf{u}(n)$$

---

## `lms.py` — Least Mean Squares (LMS)

**Update rule:**
$$\mathbf{w}(n+1) = \mathbf{w}(n) + \mu\, e(n)\, \mathbf{u}(n)$$

Stochastic gradient descent on the instantaneous MSE $\mathcal{L} = e^2(n)$.

| **Pros** | **Cons** |
|---|---|
| $O(N)$ per sample; trivial to implement | Convergence rate tied to input power; must have $0 < \mu < 2/\lambda_\max$ |
| No matrix operations | Slow on correlated (coloured) inputs |
| Unbiased in stationary conditions | No forgetting mechanism |

---

## `nlms.py` — Normalized LMS (NLMS)

**Update rule:**
$$\mathbf{w}(n+1) = \mathbf{w}(n) + \frac{\mu}{\|\mathbf{u}(n)\|^2 + \epsilon}\, e(n)\, \mathbf{u}(n)$$

Normalises the step size by input power, making the effective step size $\tilde{\mu} = \mu / \|\mathbf{u}\|^2$ approximately independent of input level.

| **Pros** | **Cons** |
|---|---|
| Robust to input power variations | Still sensitive to eigenvalue spread |
| Stability condition simplifies to $0 < \mu < 2$ | Slightly higher cost than LMS due to norm |
| Drop-in replacement for LMS | |

---

## `blms.py` — Block LMS (BLMS)

Processes $L$ samples per iteration. The input block is arranged into a Hankel convolution matrix $\mathbf{A} \in \mathbb{R}^{L \times N}$:
$$\mathbf{A}_{i,j} = x(nL + i - j), \quad i=0,\ldots,L-1,\; j=0,\ldots,N-1$$

**Block error and update:**
$$\mathbf{e}(n) = \mathbf{d}(n) - \mathbf{A}\,\mathbf{w}, \qquad \mathbf{w} \leftarrow \mathbf{w} + \frac{\mu}{L}\,\mathbf{A}^T\,\mathbf{e}(n)$$

Equivalent to averaging $L$ instantaneous LMS gradients before taking a step.

| **Pros** | **Cons** |
|---|---|
| Naturally vectorisable; BLAS-friendly | Increased latency of $L$ samples |
| Smoother gradient estimates than LMS | No normalisation by default |
| Enables frequency-domain implementation | Same eigenvalue-spread sensitivity as LMS |

---

## `bnlms.py` — Block NLMS (BNLMS)

Extends BLMS with row-wise exponential moving average (EMA) normalisation:
$$\boldsymbol{\phi}(n) = \beta\,\boldsymbol{\phi}(n-1) + (1-\beta)\,\text{diag}\!\left(\mathbf{A}\mathbf{A}^T\right) \in \mathbb{R}^L$$

**Update:**
$$\mathbf{h} \leftarrow \mathbf{h} + \frac{\mu}{L}\,\mathbf{A}^T \operatorname{diag}\!\left(\boldsymbol{\phi}(n)+\epsilon\right)^{-1}\mathbf{e}(n)$$

Each row of $\mathbf{A}$ is individually normalised by a smoothed estimate of its squared norm.

| **Pros** | **Cons** |
|---|---|
| More stable than BLMS on non-stationary inputs | EMA introduces a tuning parameter $\beta$ |
| Row-wise normalisation adapts to local power | Slightly higher memory than BLMS |

---

## `apa.py` — Affine Projection Algorithm (APA)

Maintains the $P$ most recent input vectors as columns of $\mathbf{A} \in \mathbb{R}^{N \times P}$ and the corresponding desired outputs $\mathbf{D} \in \mathbb{R}^P$.

**Error vector and projection step:**
$$\mathbf{e} = \mathbf{D} - \mathbf{A}^T\mathbf{w}, \qquad \boldsymbol{\delta} = \left(\mathbf{A}^T\mathbf{A} + \alpha\mathbf{I}\right)^{-1}\mathbf{e}$$

**Update:**
$$\mathbf{w} \leftarrow \mathbf{w} + \mu\,\mathbf{A}\,\boldsymbol{\delta}$$

Projects the weight update onto the affine subspace defined by the last $P$ input vectors. Reduces to NLMS for $P=1$.

| **Pros** | **Cons** |
|---|---|
| Much faster convergence than NLMS on correlated inputs | $O(P^3)$ matrix inversion per sample |
| $P$ is a direct convergence–complexity trade-off | Regularisation $\alpha$ requires tuning |
| Well-understood convergence theory | Memory grows as $O(NP)$ |

---

## `rls.py` — Recursive Least Squares (RLS)

Minimises the exponentially weighted least-squares cost:
$$\mathcal{L}(n) = \sum_{i=0}^{n} \lambda^{n-i}\, e^2(i)$$

**Matrix inversion lemma (time-recursive form):**
$$\mathbf{r}(n) = \mathbf{P}(n-1)\,\mathbf{u}(n)$$
$$\mathbf{g}(n) = \frac{\mathbf{r}(n)}{\lambda + \mathbf{u}^T(n)\,\mathbf{r}(n)}$$
$$\mathbf{w}(n) = \mathbf{w}(n-1) + e(n)\,\mathbf{g}(n)$$
$$\mathbf{P}(n) = \lambda^{-1}\!\left[\mathbf{P}(n-1) - \mathbf{g}(n)\,\mathbf{u}^T(n)\,\mathbf{P}(n-1)\right]$$

$\mathbf{P}(n) \approx \left[\sum_i \lambda^{n-i}\mathbf{u}(i)\mathbf{u}^T(i)\right]^{-1}$ is the inverse input correlation matrix.

| **Pros** | **Cons** |
|---|---|
| Optimal convergence for Gaussian inputs | $O(N^2)$ per sample |
| Forgetting factor $\lambda$ handles non-stationarity | $\mathbf{P}$ can become ill-conditioned (divergence risk) |
| No step-size tuning required | Numerical stability requires careful initialisation ($\delta\mathbf{I}$) |

---

## `kalman.py` — Time-Domain Kalman Filter

State-space formulation with a random-walk prior on $\mathbf{w}$:
$$\mathbf{w}(n) = \mathbf{w}(n-1) + \mathbf{q}(n), \quad \mathbf{q} \sim \mathcal{N}(\mathbf{0},\,\mathbf{Q}), \quad \mathbf{Q} = \sigma^2_v\,\mathbf{I}$$
$$d(n) = \mathbf{u}^T(n)\,\mathbf{w}(n) + v(n), \quad v(n) \sim \mathcal{N}(0,\,R(n))$$

**Measurement noise adaptation** (instantaneous estimate):
$$R(n) = e^2(n) + \epsilon$$

**Predict–update cycle:**
$$\mathbf{P}^-(n) = \mathbf{P}(n-1) + \mathbf{Q}$$
$$\mathbf{K}(n) = \frac{\mathbf{P}^-(n)\,\mathbf{u}(n)}{\mathbf{u}^T(n)\,\mathbf{P}^-(n)\,\mathbf{u}(n) + R(n)}$$
$$\mathbf{w}(n) = \mathbf{w}(n-1) + \mathbf{K}(n)\,e(n)$$
$$\mathbf{P}(n) = \left(\mathbf{I} - \mathbf{K}(n)\,\mathbf{u}^T(n)\right)\mathbf{P}^-(n)$$

| **Pros** | **Cons** |
|---|---|
| Principled probabilistic framework; optimal for linear-Gaussian models | $O(N^2)$ per sample (same as RLS) |
| Adaptive $R(n)$ allows tracking time-varying systems | Using $e_n^2$ as $R(n)$ is a noisy (non-causal-optimal) approximation |
| Process noise $\mathbf{Q}$ provides regularisation | Two covariance parameters require tuning |
