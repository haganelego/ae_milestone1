# Bayesian Methods for Semantic-Saturation Stopping — Explainer

> Tutorial-style walkthrough of the candidate stopping methods (equations +
> concrete numeric examples) and how each maps onto the Milestone-1 problem:
> detecting when a scene-graph sequence saturates under progressive resolution
> recovery (20 levels, EGTR on Visual Genome).
>
> Audience note: written for a reader not deeply familiar with Bayesian methods.
> Date: 2026-08-14.

## 0. Bayes in one equation

Bayes = "update a prior belief with data into a posterior belief":

```
p(θ | data)  =  p(data | θ) · p(θ)  /  p(data)
 posterior       likelihood    prior     (normalizer, ignorable)
```

- **θ**: unknown we care about (here: parameters of the saturation curve).
- **prior p(θ)**: belief before seeing data ("probably around this").
- **likelihood p(data|θ)**: if θ had this value, how likely is the observed data.
- **posterior p(θ|data)**: updated belief. Comes out as a *distribution*, not a
  point — this is the whole point of Bayes (uncertainty is retained).

Mini example: coin with heads-probability θ. Prior "≈0.5". Observe 2 heads / 3
tosses. Posterior = "slightly above 0.5, still with spread." More data → sharper
posterior. Our task swaps "coin probability" for "shape of a saturation curve."

### Running example series

Observed semantic-attainment y_t at level t (adjacent-level similarity
saturating toward 1):

```
t   : 1     2     3     4     5     6     7     8    ...  20
y_t : 0.30  0.46  0.58  0.69  0.78  0.85  0.90  0.93 ...  ~0.99
```

Question each method answers: "At level 5, when does it saturate / may we stop?"

---

## 1. Bayesian saturating-curve fitting (PRIMARY)

Assume a smooth curve under the series:

```
y_t = f(t; θ) + noise,     f(t; θ) = s∞ − (s∞ − s₀)·exp(−λt)
θ = (s∞, λ, s₀, σ)
```

- **s∞**: final attainment level (semantic ceiling of this image)
- **λ**: saturation speed, **s₀**: initial value, **σ**: observation noise

### Flow (with numbers)

**Prior** (from past images):
```
s∞ ~ Normal(0.95, 0.05)    "most images saturate around 0.95"
λ  ~ Normal(0.35, 0.15)    "typical saturation speed"
```

**Likelihood** (levels 1..5 observed): for a candidate θ, measure how well the
curve passes through the points:
```
p(y_1:5 | θ) = ∏_{t=1}^{5} Normal(y_t | f(t;θ), σ²)
```

**Posterior** = prior × likelihood. Only 4 params → grid approximation (brute
force over a grid) is enough; no MCMC needed, milliseconds. E.g. at level 5:
```
s∞ posterior ≈ 0.97 ± 0.03    (still wide this early)
λ  posterior ≈ 0.33 ± 0.06
```

**Posterior predictive**: use the whole θ posterior to predict future y and the
residual gain Δ_t = s∞ − f(t), as distributions.

**Stopping rule**:
```
stop at first t where  P(Δ_t < ε | y_1:t) ≥ δ
e.g. ε=0.02, δ=0.9: "stop once P(residual gain < 0.02) exceeds 90%"
```
On the running example, the condition triggers around level 7–8, and reports
"saturation at level 8 ± 1."

### What this buys us
- **Prediction, not just detection**: at level 5 it forecasts "saturation ≈
  level 8" → directly usable for transmission scheduling.
- **ε, δ are interpretable**: ε = tolerated semantic loss, δ = confidence.
- **Non-saturating images handled naturally**: if s∞ posterior never converges,
  stop-probability never rises → the method says "undecidable" (a threshold rule
  cannot). Connects to the EGTR detector-ceiling analysis.

---

## 2. No-prior variant: maximum-likelihood curve fit (frequentist)

Same curve, no prior; pick the single best-fitting point:
```
θ̂ = argmax_θ  p(y_1:t | θ)     (= nonlinear least squares)
```
No posterior spread; point estimate + asymptotic CI. At level 5 the data has not
yet bent, so **s∞ fits anywhere from 0.9 to 1.2** (unidentifiable). Without the
prior "support," early extrapolation is unstable. Good as a contrast baseline
showing the prior's value.

---

## 3. SPRT (Sequential Probability Ratio Test) — test, don't fit

Drop the curve; look at the increment g_t = 1 − sim(G_t, G_{t−1}) (info gained at
that level). Two hypotheses:
```
H₀: "still gaining" (mean increment μ = μ₁, large)
H₁: "saturated"     (mean increment μ = μ₀ ≈ 0)
```
Accumulate the log-likelihood ratio each level:
```
Λ_t = Σ_{i=1}^{t} log [ p(g_i | H₁) / p(g_i | H₀) ]
stop (declare saturated) when  Λ_t ≥ log((1−β)/α)
α = false-alarm rate, β = miss rate
```
Example: g_t = 0.16, 0.12, 0.11, 0.09, 0.07, 0.05, 0.03, 0.02...; as g shrinks,
Λ_t accumulates and crosses the threshold at level 7–8 → stop.

- **Pros**: light, few assumptions, explicit α/β. Calibrate μ₀, μ₁ on other
  images (that is where the "other-image info" enters).
- **Con**: detection only, **no prediction** ("how many more levels" is unknown).

Use as the opposite pole to the Bayesian curve: predicts-but-heavy vs
light-but-no-prediction.

---

## 4. BOCPD (Bayesian Online Changepoint Detection) — catch the regime shift

Treat the series as an "improving phase" then a "plateau phase"; detect *when*
it switched. Hidden variable = run length r_t (steps since last changepoint);
update its posterior each step:
```
p(r_t | y_1:t)   computed recursively
r_t collapsing to 0  ⇔  a changepoint (improving → plateau)
```
Example: after the large-increment region, y flattens (0.03→0.02→0.01); the
"new regime" probability spikes near level 7 → saturation onset.

- Assumes less about curve shape; more robust to non-monotone wiggle.
- **Caveat**: 20 points is short for changepoint detection → unstable; keep as
  ablation / auxiliary to §1, not primary.

---

## 5. Conformal prediction — distribution-free guarantee layer

Not a standalone method; a **wrapper** that turns "90% confident" into "actually
covers 90%," with no distributional assumption. Collect prediction errors
(nonconformity scores) on a calibration set; take the 90% quantile q; new-image
interval = prediction ± q → coverage ≥ 90% guaranteed.
```
calib errors {e_1..e_n} → 90% quantile q
new interval = [pred − q, pred + q]   ⇒  coverage ≥ 90%
```
Lets us attach a guarantee to "semantic loss at stop ≤ ε" regardless of model
correctness. SGG×conformal is brand new (CVPR 2025). Needs a calibration set
(again, other-image info).

---

## 6. Gaussian process — "don't even assume the curve shape" (reference)

§1 fixed the functional form (exp saturation). A GP puts a prior on the function
itself, assuming only smoothness ("nearby levels → nearby y"). Flexible but with
20 points the shape prior is too weak → unstable extrapolation. §1 (strong shape
prior) is primary; GP is a comparison point.

---

## Map to our problem

| Method | Predict? | Assumption / info source | Role |
|---|---|---|---|
| §1 Bayesian curve | ✅ forecast | curve form + prior (other images) | **PRIMARY**: stop + saturation-point forecast |
| §2 ML fit | weak | curve form only | contrast showing prior's value |
| §3 SPRT | ❌ detect only | calibration data | lightweight, error-rate-controlled pole |
| §4 BOCPD | changepoint | weak | ablation / auxiliary |
| §5 Conformal | adds guarantee | calibration data | guarantee layer on any method |
| §6 GP | ✅ but unstable | smoothness only | reference comparison |

**Shared essence**: judging the future from a few early points always requires
supplying missing information — via a *prior*, a *curve assumption*, or
*calibration data*. Bayes (§1) supplies it as an explicit, interpretable prior
and, uniquely, also yields *prediction* — best fit for the transmission
application.

---

## Next step

Implement §1 on the existing 100-image score series; compare stopping MAE vs the
LLM-threshold method. Turn the math into an implementation design (grid
approximation vs numpyro; how to build the prior) — candidate for Codex handoff.
