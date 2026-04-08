# Nonlinear Adaptive Filters

Linear adaptive filters assume the acoustic path is LTI. In practice, loudspeaker drivers, power amplifiers, and ADC/DAC clipping introduce nonlinearities. The filters in this directory expand the input into a richer feature space so that a *linear* weight vector can approximate the resulting nonlinear input-output map.

All filters use NLMS-style normalised gradient steps and share the same error definition:
$$e(n) = d(n) - \hat{d}(n)$$

---

## `volterra.py` — Second-Order Volterra Filter (SVF)

Truncates the Volterra series at the second order. The output is:
$$\hat{d}(n) = \sum_{k=0}^{M-1} w_k\, u_k(n) + \sum_{i=0}^{L-1}\sum_{j \geq i}^{L-1} h_{ij}\, u_i(n)\, u_j(n)$$

where $\mathbf{u}(n) = [x(n), \ldots, x(n-M+1)]^T$, and the second sum runs over the upper-triangular part of $\mathbf{u}_{:L}\mathbf{u}_{:L}^T$.

**Implementation detail:** The $L(L+1)/2$ unique second-order products are extracted once per sample and stored in a sliding buffer $\mathbf{U}_2 \in \mathbb{R}^{M \times L(L+1)/2}$.

$$x_2(n) = \mathbf{U}_2(n)\,\mathbf{h}_2, \qquad \mathbf{g}(n) = \mathbf{u}(n) + x_2(n)$$

The combined feature vector $\mathbf{g}$ is fed into the linear weight $\mathbf{w}$:

$$\hat{d}(n) = \mathbf{w}^T \mathbf{g}(n)$$

**Updates (two NLMS steps):**
$$\mathbf{w} \leftarrow \mathbf{w} + \frac{\mu_1\, e(n)\, \mathbf{g}(n)}{\|\mathbf{g}(n)\|^2 + \epsilon}$$
$$\nabla_2 = \mathbf{U}_2^T\,\mathbf{w}, \qquad \mathbf{h}_2 \leftarrow \mathbf{h}_2 + \frac{\mu_2\, e(n)\, \nabla_2}{\|\nabla_2\|^2 + \epsilon}$$

| **Pros** | **Cons** |
|---|---|
| Grounded in Volterra series theory; interpretable kernels | Quadratic parameter count $O(L^2)$ grows rapidly with nonlinearity memory $L$ |
| Two independent step sizes for linear/nonlinear components | Third- and higher-order distortion unmodelled |
| NLMS normalisation stabilises both update stages | Coupled updates ($\mathbf{w}$ affects $\nabla_2$) can slow convergence |

---

## `flaf.py` — Functional Link Adaptive Filter (FLAF)

Expands each tap $u_k$ into $2P+1$ features using trigonometric basis functions:
$$\phi_k = \left[u_k,\; \sin(0),\; \cos(0),\; \sin(\pi u_k),\; \cos(\pi u_k),\; \ldots,\; \sin\!\left((P{-}1)\pi u_k\right),\; \cos\!\left((P{-}1)\pi u_k\right)\right]$$

The full feature vector $\mathbf{g}(n) \in \mathbb{R}^{(2P+1)M}$ concatenates $\phi_k$ for all $M$ taps. Note: $\sin(0)=0$ so the $p=0$ sine term is trivially zero.

$$\hat{d}(n) = \mathbf{w}^T\mathbf{g}(n)$$

**Update:**
$$\mathbf{w} \leftarrow \mathbf{w} + \frac{2\mu\, e(n)\, \mathbf{g}(n)}{\|\mathbf{g}(n)\|^2 + \epsilon}$$

| **Pros** | **Cons** |
|---|---|
| $O((2P+1)M)$ parameters; moderate expansion | Feature vector is $(2P+1)\times$ larger than linear filter — higher memory and compute |
| Universal approximation for band-limited nonlinearities | Trigonometric basis fixed; may not match the actual nonlinearity efficiently |
| Single weight vector; straightforward convergence analysis | $p=0$ sine features are always zero (wasted capacity) |

---

## `aeflaf.py` — Adaptive Exponential FLAF (AEFLAF)

Modifies FLAF by multiplying each trigonometric basis function by an adaptive exponential envelope parameterised by a scalar $a \geq 0$:
$$g_{\sin,k,j}(n) = e^{-a|u_k|}\sin(j\pi u_k), \qquad g_{\cos,k,j}(n) = e^{-a|u_k|}\cos(j\pi u_k)$$

The scalar $a$ controls how quickly the basis functions decay for large input amplitudes, effectively tuning the nonlinearity operating point.

**Gradient of $y$ w.r.t. $a$:**
$$\frac{\partial g_{\sin,k,j}}{\partial a} = -|u_k|\, g_{\sin,k,j}, \qquad \frac{\partial g_{\cos,k,j}}{\partial a} = -|u_k|\, g_{\cos,k,j}$$

$$\nabla_a = \mathbf{z}^T\mathbf{w}, \quad z_{\sin,k,j} = -|u_k|\,g_{\sin,k,j}, \quad z_{\cos,k,j} = -|u_k|\,g_{\cos,k,j}$$

(passthrough elements $z_k = 0$).

**Joint updates:**
$$\mathbf{w} \leftarrow \mathbf{w} + \frac{\mu\, e(n)\, \mathbf{g}(n)}{\|\mathbf{g}(n)\|^2 + \epsilon}$$
$$a \leftarrow a + \frac{\mu_a\, e(n)\, \nabla_a}{\nabla_a^2 + \epsilon}$$

| **Pros** | **Cons** |
|---|---|
| Exponential envelope adapts to the amplitude regime of the nonlinearity | Joint adaptation of $\mathbf{w}$ and $a$ creates a non-convex objective |
| Single extra scalar $a$ with negligible memory overhead | $a$ must stay non-negative for the envelope to be valid; no constraint enforced |
| Can outperform FLAF when nonlinearity is strongly amplitude-dependent | Gradient w.r.t. $a$ is coupled to the current $\mathbf{w}$; convergence less predictable |

---

## `sflaf.py` — Split FLAF (SFLAF)

Explicitly separates the linear and nonlinear processing paths:
$$\hat{d}(n) = \underbrace{\mathbf{w}_L^T\,\mathbf{u}(n)}_{y_L} + \underbrace{\mathbf{w}_{FL}^T\,\mathbf{g}(n)}_{y_{FL}}$$

The feature vector $\mathbf{g}(n) \in \mathbb{R}^{2PM}$ contains only trigonometric terms (no passthrough), interleaved as $[\sin(0), \cos(0), \sin(\pi u_k), \cos(\pi u_k), \ldots]$ per tap.

**Independent NLMS updates:**
$$\mathbf{w}_L \leftarrow \mathbf{w}_L + \frac{\mu_L\, e(n)\, \mathbf{u}(n)}{\|\mathbf{u}(n)\|^2 + \epsilon}$$
$$\mathbf{w}_{FL} \leftarrow \mathbf{w}_{FL} + \frac{\mu_{FL}\, e(n)\, \mathbf{g}(n)}{\|\mathbf{g}(n)\|^2 + \epsilon}$$

| **Pros** | **Cons** |
|---|---|
| Independent step sizes allow the linear path to converge fast and stabilise before the nonlinear path | Linear and nonlinear paths share the same error signal; gradient interference possible |
| Interpretable: $\mathbf{w}_L$ approximates the linear component of the echo path | No mechanism to weight the relative contribution of each path |
| Slightly smaller feature vector than FLAF (no per-tap passthrough in $\mathbf{g}$) | Fixed equal weighting may be suboptimal when nonlinearity is weak |

---

## `cflaf.py` — Collaborative FLAF (CFLAF)

Extends SFLAF with a **learned sigmoid gate** $\lambda_n \in (0,1)$ that controls how much the nonlinear branch contributes:
$$\lambda_n = \sigma(\alpha_n) = \frac{1}{1+e^{-\alpha_n}}, \qquad \hat{d}(n) = y_L(n) + \lambda_n\, y_{FL}(n)$$

**Two-stage update:**

1. Update $\mathbf{w}_{FL}$ using the full un-gated error $e_{FL} = d - (y_L + y_{FL})$:
$$\mathbf{w}_{FL} \leftarrow \mathbf{w}_{FL} + \frac{\mu_{FL}\, e_{FL}\, \mathbf{g}}{\|\mathbf{g}\|^2 + \epsilon}$$

2. Compute gated error $e_n = d - (y_L + \lambda_n y_{FL})$ and update gate and linear path:
$$\gamma \leftarrow \beta\,\gamma + (1-\beta)\,y_{FL}^2 \quad\text{(EMA power of FL output)}$$
$$\alpha \leftarrow \alpha + \frac{\mu_a\, e_n\, y_{FL}\, \lambda_n(1-\lambda_n)}{\gamma}, \qquad \alpha \leftarrow \operatorname{clip}(\alpha,\,-4,\,4)$$
$$\mathbf{w}_L \leftarrow \mathbf{w}_L + \frac{\mu_L\, e_n\, \mathbf{u}}{\|\mathbf{u}\|^2 + \epsilon}$$

The gradient $\partial\hat{d}/\partial\alpha = y_{FL}\,\lambda_n(1-\lambda_n)$ is the standard logistic derivative. Normalisation by $\gamma$ stabilises the $\alpha$ update across varying signal levels.

| **Pros** | **Cons** |
|---|---|
| Gate automatically suppresses nonlinear branch when the system is approximately linear (reduces misadjustment) | Three-step update with two error signals is harder to analyse theoretically |
| Clipping $\alpha \in [-4,4]$ prevents saturation of the sigmoid | EMA parameter $\beta$ and $\mu_a$ add tuning burden |
| Smoothly interpolates between purely linear and fully nonlinear operation | Performance depends on whether $y_{FL}^2$ is a reliable proxy for nonlinear contribution |
