# Frequency-Domain Adaptive Filters

All filters here operate on blocks of $M$ samples using the **overlap-save** method. By moving to the frequency domain, $O(N^2)$ convolution becomes $O(N \log N)$ FFT operations, which is decisive for long room impulse responses (RIRs) common in acoustic echo cancellation.

**Notation:**
- $M$ — block size (= partition size)
- $N$ — number of partitions (PFDAF/PFDKF only)
- $\mathcal{F}\{\cdot\}$ — real FFT of length $2M$, producing $M+1$ complex bins
- $\odot$ — element-wise (Hadamard) product
- $(\cdot)^*$ — complex conjugate

**Overlap-save output extraction** (all filters):
$$\mathbf{x}_n = [x_\text{old},\, x_n] \in \mathbb{R}^{2M}, \qquad X_n = \mathcal{F}\{\mathbf{x}_n\} \in \mathbb{C}^{M+1}$$
$$y_n = \mathcal{F}^{-1}\{H \odot X_n\}[M:] \quad\text{(take second half)}$$

**Zero-tail constraint** enforces a causal FIR of length $\leq M$:
$$h = \mathcal{F}^{-1}\{H\}, \quad h[M:] \leftarrow 0, \quad H \leftarrow \mathcal{F}\{h\}$$

---

## `fdaf.py` — Frequency-Domain Adaptive Filter (FDAF)

Block-frequency-domain equivalent of NLMS with an EMA power normaliser.

**Error and gradient computation:**
$$e_n = d_n - y_n, \qquad E_n = \mathcal{F}\!\left\{[\mathbf{0}_M,\; e_n \odot w_\text{Hann}]\right\}$$

**Power normalisation (EMA):**
$$\boldsymbol{\phi}(n) = \beta\,\boldsymbol{\phi}(n-1) + (1-\beta)\,|X_n|^2, \qquad \beta = 0.9$$

**Filter update:**
$$G = \frac{\mu\, E_n}{\boldsymbol{\phi}(n) + \epsilon}, \qquad H \leftarrow H + X_n^* \odot G$$

Followed by zero-tail constraint.

The Hann window on $e_n$ tapers the gradient to reduce spectral leakage when computing the cross-correlation $X_n^* \odot E_n$.

| **Pros** | **Cons** |
|---|---|
| $O(M \log M)$ per block vs. $O(M^2)$ for time-domain block LMS | Total filter length fixed at $M$; long RIRs require large $M$ (high latency) |
| EMA normaliser is robust to non-stationary input power | Hann window on error introduces a bias in the gradient |
| Simple; single-partition special case of PFDAF | Zero-tail constraint adds one extra FFT/IFFT pair per block |

---

## `pfdaf.py` — Partitioned-Block FDAF (PFDAF)

Decomposes a long filter $H \in \mathbb{C}^{N(M+1)}$ into $N$ partitions, each of length $M$, allowing a block latency of $M$ samples regardless of total filter length $NM$.

**State:**
$$\mathbf{H},\,\mathbf{X} \in \mathbb{C}^{N \times (M+1)}, \quad \mathbf{X}_p = X_{n-p} \quad p=0,\ldots,N-1$$

**Filtered output:**
$$Y = \sum_{p=0}^{N-1} H_p \odot X_p, \qquad y = \mathcal{F}^{-1}\{Y\}[M:]$$

**Gradient:**
$$X_2 = \sum_{p=0}^{N-1} |X_p|^2, \qquad G = \frac{\mu\, E}{X_2 + \epsilon}$$

**Update (all partitions simultaneously):**
$$H_p \leftarrow H_p + X_p^* \odot G, \quad p = 0,\ldots,N-1$$

**Zero-tail constraint — two modes:**

- *Partial* (`partial_constrain=True`): constrain only partition $p \bmod N$ per block. Reduces cost by factor $N$ at the price of a slower effective constraint.
- *Full* (`partial_constrain=False`): constrain all $N$ partitions every block. Stricter, but $N\times$ more costly.

| **Pros** | **Cons** |
|---|---|
| Decouples block latency ($M$) from filter length ($NM$) | Requires maintaining $N$ FFT frames of history |
| Partial constraint halves per-block cost asymptotically | Partial constraint may introduce residual aliasing in early iterations |
| Scales efficiently to RIR lengths of thousands of taps | More complex implementation than FDAF |

---

## `fdkf.py` — Frequency-Domain Kalman Filter (FDKF)

Applies an independent scalar Kalman filter **per frequency bin** under the assumption that frequency-domain channel coefficients are uncorrelated across bins.

**Per-bin state-space model (scalar complex-valued):**
$$H_k(n) = H_k(n-1) + q_k, \quad q_k \sim \mathcal{N}(0, Q)$$
$$E_k(n) = X_k(n)\,H_k(n) + v_k, \quad v_k \sim \mathcal{N}(0, R_k)$$

**Adaptive measurement noise (EMA of squared error spectrum):**
$$R_k(n) = \beta\,R_k(n-1) + (1-\beta)\,|E_k(n)|^2$$

**Process noise prediction:**
$$P_k^-(n) = P_k(n-1) + Q\,|H_k(n-1)|$$

**Kalman gain and update:**
$$K_k(n) = \frac{P_k^-(n)\,X_k^*(n)}{X_k(n)\,P_k^-(n)\,X_k^*(n) + R_k(n)}$$
$$H_k(n) = H_k(n-1) + K_k(n)\,E_k(n)$$
$$P_k(n) = \left(1 - K_k(n)\,X_k(n)\right)P_k^-(n)$$

Followed by zero-tail constraint.

| **Pros** | **Cons** |
|---|---|
| Per-bin adaptation rates; bins with more signal energy converge faster | Assumes independence across bins (only exact for circular convolution) |
| Adaptive $R_k$ tracks non-stationary noise without manual tuning | $|H_k|$-dependent process noise is a heuristic, not Bayesian-optimal |
| $O(M)$ per-bin scalar ops; no matrix inversions | EMA lag in $R_k$ can cause sluggish adaptation during rapid changes |

---

## `pfdkf.py` — Partitioned-Block FDKF (PFDKF)

Combines the $N$-partition structure of PFDAF with the Kalman framework of FDKF. Each partition $p$ and frequency bin $k$ has its own variance $P_{p,k}$.

**State:** $\mathbf{P} \in \mathbb{R}^{N \times (M+1)}$, $\mathbf{H},\mathbf{X} \in \mathbb{C}^{N \times (M+1)}$

**Aggregate power:**
$$X_2 = \sum_{p=0}^{N-1} |X_p|^2$$

**Innovation variance estimate:**
$$P_e = \frac{1}{2}\,\mathbf{P} \odot X_2 + \frac{|E|^2}{N}$$

**Per-partition gain:**
$$\mu_{p,k} = \frac{P_{p,k}}{(P_e)_k + \epsilon}$$

**Variance update (state-dependent process noise):**
$$\mathbf{P} \leftarrow A^2\!\left(1 - \frac{1}{2}\,\mu \odot X_2\right)\mathbf{P} + (1-A^2)\,|\mathbf{H}|^2$$

**Filter update:**
$$H_p \leftarrow H_p + \mu_p \odot X_p^* \odot E, \quad p=0,\ldots,N-1$$

Zero-tail constraint applied in partial or full mode (same as PFDAF).

The variance update blends a forgetting-factor term ($A^2 \approx 1$, slow drift) with an energy-normalised term $|(1-A^2)|H|^2|$ that acts as a data-driven floor.

| **Pros** | **Cons** |
|---|---|
| Handles very long RIRs with principled per-bin gain control | Most complex filter in this library; many interacting hyperparameters |
| Partial constraint preserves low latency | Approximations in $P_e$ and the variance update are heuristic |
| $A$ provides a clean forgetting-factor interpretation | Sensitive to initialisation of $P_\text{initial}$; too large → slow, too small → unstable |
