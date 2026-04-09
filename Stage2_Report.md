# Stage 2 Report — Deep Learning Mechanism Fixes

**Course:** COMP5329 Deep Learning, University of Sydney, Semester 1 2026  
**Assignment:** Assignment 1 — QANet for Extractive Question Answering  
**Stage:** Stage II — Fix Deep Learning Mechanism Errors  
**Dataset:** SQuAD v1.1 (mini subset)  
**Model:** QANet (Yu et al., 2018)

---

## 1. Background & Task Description

### 1.1 Overview

Stage II focuses on correctness of individual deep learning mechanisms. Unlike Stage I (crash bugs), Stage II bugs allow the pipeline to run but produce mathematically wrong behavior — distorted training dynamics, wrong gradient signals, or incorrect algorithmic implementations. These errors map directly to lecture topics: optimisation algorithms, attention mechanisms, and loss computation.

A total of **8 bugs** were identified and fixed across 4 files. All are **LOGIC** category — they do not cause Python exceptions but silently produce incorrect computations.

---

## 2. Stage 2 Bug Fixes

---

### Bug S2-1 — `Optimizers/adam.py`, Line 53

**Category:** LOGIC (Anti-regularization)

**Buggy code:**
```python
# Weight decay
if wd != 0.0:
    grad = grad.add(p, alpha=-wd)
```

**Fixed code:**
```python
# Weight decay
if wd != 0.0:
    grad = grad.add(p, alpha=wd)
```

**Explanation:**  
L2 regularisation adds the penalty $\frac{\lambda}{2}\|\theta\|^2$ to the loss. Its gradient with respect to parameter $p$ is $\lambda p$ (positive). The effective gradient for the update should be:

$$\tilde{g} = g + \lambda p$$

`tensor.add(other, alpha=scalar)` computes `tensor + scalar * other`. Using `alpha=-wd` gives $\tilde{g} = g - \lambda p$, which **subtracts** the regularisation gradient. This causes the optimiser to push parameters **away from zero** on every step — the exact opposite of regularisation. With `alpha=wd` (positive), $\tilde{g} = g + \lambda p$, producing the correct weight-decay update $p \leftarrow (1 - \eta\lambda)p - \eta g$ that shrinks parameter magnitude.

---

### Bug S2-2 — `Optimizers/adam.py`, Line 69

**Category:** LOGIC (Wrong second moment — breaks adaptive learning rate)

**Buggy code:**
```python
# Update biased moment estimates
m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
v.mul_(beta2).add_(grad, alpha=1.0 - beta2)
```

**Fixed code:**
```python
# Update biased moment estimates
m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
```

**Explanation:**  
The Adam algorithm (Kingma & Ba, 2015) maintains two moment estimates:
- **First moment** $m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$ — exponential moving average (EMA) of gradients
- **Second moment** $v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$ — EMA of **squared** gradients

The buggy code uses `add_(grad, ...)` for both, meaning both `m` and `v` track the EMA of the raw gradient $g$. Without $g^2$ in the second moment, $v$ carries no information about gradient variance; $\sqrt{\hat{v}_t}$ becomes nearly identical to $|\hat{m}_t|$, so the denominator no longer provides per-parameter learning rate adaptation — the core feature of Adam.

`addcmul_(grad, grad, value=scalar)` computes `v += scalar * grad * grad`, correctly accumulating $g_t^2$.

---

### Bug S2-3 — `Optimizers/adam.py`, Lines 72–73

**Category:** LOGIC (Bias correction formula broken — sign flips and divergence)

**Buggy code:**
```python
# Bias correction
bias_correction1 = 1.0 - beta1 * t
bias_correction2 = 1.0 - beta2 * t
```

**Fixed code:**
```python
# Bias correction
bias_correction1 = 1.0 - beta1 ** t
bias_correction2 = 1.0 - beta2 ** t
```

**Explanation:**  
The Adam paper introduces bias-corrected estimates to account for initialisation at zero:

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

The denominator is $1 - \beta^t$ where $t$ is the **exponent** (power). With $\beta_1 = 0.9$ at step $t = 2$:
- **Correct:** $1 - 0.9^2 = 1 - 0.81 = 0.19$ — scales moments up to compensate for the slow ramp-up
- **Buggy:** $1 - 0.9 \times 2 = -0.8$ — **negative**, which flips the sign of $\hat{m}$, causing parameter updates in the wrong direction

Beyond step $t > 1/\beta$, the buggy correction becomes negative for all future steps, permanently inverting the update direction and diverging. Changing `*` to `**` restores exponentiation.

---

### Bug S2-4 — `Optimizers/sgd.py`, Line 39

**Category:** LOGIC (Anti-regularization — same error as S2-1)

**Buggy code:**
```python
# Weight decay: equivalent to L2 regularisation
if wd != 0.0:
    grad = grad.add(p, alpha=-wd)
```

**Fixed code:**
```python
# Weight decay: equivalent to L2 regularisation
if wd != 0.0:
    grad = grad.add(p, alpha=wd)
```

**Explanation:**  
Same sign error as in `adam.py`. For vanilla SGD, the L2-regularised parameter update is:

$$p \leftarrow p - \eta(\nabla\mathcal{L} + \lambda p) = p(1 - \eta\lambda) - \eta\nabla\mathcal{L}$$

With `alpha=-wd`, the effective update is $p \leftarrow p(1 + \eta\lambda) - \eta\nabla\mathcal{L}$, which **grows** parameter magnitude on every step (anti-weight-decay / inverse L2). PyTorch's reference `torch.optim.SGD` implementation uses positive alpha: `d_p = d_p.add(p, alpha=weight_decay)`.

---

### Bug S2-5 — `Losses/loss.py`, Line 13

**Category:** LOGIC (Loss scale 2× too large)

**Buggy code:**
```python
def qa_ce_loss(p1, p2, y1, y2):
    """QA span loss using cross-entropy."""
    return F.cross_entropy(p1, y1) + F.cross_entropy(p2, y2)
```

**Fixed code:**
```python
def qa_ce_loss(p1, p2, y1, y2):
    """QA span loss using cross-entropy."""
    return 0.5 * (F.cross_entropy(p1, y1) + F.cross_entropy(p2, y2))
```

**Explanation:**  
The QA span loss is the **average** of the start-position loss and end-position loss:

$$\mathcal{L} = \frac{1}{2}\left[\text{CE}(p_1, y_1) + \text{CE}(p_2, y_2)\right]$$

The companion function `qa_nll_loss` on line 7 correctly includes `0.5 *`. Without it, `qa_ce_loss` is twice as large as `qa_nll_loss` for the same predictions, causing gradients and effective learning rate to be 2× larger when this loss is selected. This breaks the comparability between the two loss functions and produces incorrect training dynamics under `qa_ce`.

---

### Bug S2-6 — `Models/encoder.py`, Line 78

**Category:** LOGIC (Unscaled attention — gradient vanishing and incorrect attention distribution)

**Buggy code:**
```python
self.scale = 1.0 / math.sqrt(self.d_k)   # computed but never used
...
attn = torch.bmm(q, k.transpose(1, 2))
attn = mask_logits(attn, attn_mask)
attn = F.softmax(attn, dim=2)
```

**Fixed code:**
```python
attn = torch.bmm(q, k.transpose(1, 2)) * self.scale
attn = mask_logits(attn, attn_mask)
attn = F.softmax(attn, dim=2)
```

**Explanation:**  
Scaled dot-product attention (Vaswani et al., 2017) computes:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

The $\frac{1}{\sqrt{d_k}}$ scaling is essential. The dot products $QK^\top$ have variance $d_k$ when $q$ and $k$ components are drawn from a unit-variance distribution. Without scaling, for larger $d_k$ (e.g., $d_k = 96/8 = 12$ in QANet), the logits grow in magnitude, pushing softmax into regions of near-zero gradient — the distribution collapses to near-one-hot, and the attention module can only attend to a single position per query. The attribute `self.scale` is correctly initialised on line 53 but never applied to the attention logits.

---

### Bug S2-7 — `Models/encoder.py`, Line 85

**Category:** LOGIC (Wrong dimension permutation — cross-sample contamination in multi-head attention)

**Buggy code:**
```python
out = out.view(batch_size, self.num_heads, length, self.d_k)
out = out.permute(1, 2, 0, 3).contiguous().view(batch_size, length, self.d_model)
```

**Fixed code:**
```python
out = out.view(batch_size, self.num_heads, length, self.d_k)
out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, length, self.d_model)
```

**Explanation:**  
After `view`, `out` has shape `[B, num_heads, L, d_k]`. The goal is to concatenate heads per position per sample, yielding `[B, L, d_model]` where `d_model = num_heads × d_k`.

Dimension labelling: `(0=B, 1=H, 2=L, 3=d_k)`.

| Permute | Resulting shape | Semantic |
|---------|----------------|---------|
| `(1,2,0,3)` | `[H, L, B, d_k]` | ✗ Batch is third dim — subsequent `view(B, L, d_model)` concatenates across wrong axes, mixing representations from different samples |
| `(0,2,1,3)` | `[B, L, H, d_k]` | ✓ Batch first, then position, then heads — `view(B, L, d_model)` correctly concatenates all heads for each `(batch, position)` pair |

With the buggy permute, the contiguous memory layout interleaves data from different batch elements when viewed as `[B, L, d_model]`. Each sample's output representation is computed from other samples' attention outputs, completely breaking the independence of samples within a batch.

---

### Bug S2-8 — `Models/encoder.py`, Lines 70–76

**Category:** LOGIC (Head flattening order inconsistent with mask/output reshape)

**Buggy code:**
```python
q = q.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, length, self.d_k)
k = k.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, length, self.d_k)
v = v.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, length, self.d_k)
...
attn_mask = mask.unsqueeze(1).expand(-1, length, -1).repeat(self.num_heads, 1, 1)
```

**Fixed code:**
```python
q = q.permute(0, 2, 1, 3).contiguous().view(batch_size * self.num_heads, length, self.d_k)
k = k.permute(0, 2, 1, 3).contiguous().view(batch_size * self.num_heads, length, self.d_k)
v = v.permute(0, 2, 1, 3).contiguous().view(batch_size * self.num_heads, length, self.d_k)
...
attn_mask = mask.unsqueeze(1).expand(-1, length, -1).repeat_interleave(self.num_heads, dim=0)
```

**Explanation:**  
The attention block flattens the `(batch, head)` axes before `torch.bmm`, so the flattening order must match both the attention mask and the later reconstruction step `out.view(batch_size, self.num_heads, length, self.d_k)`.

With `permute(2, 0, 1, 3)`, the flattened rows are **head-major**:

$$(h_0,b_0), (h_0,b_1), \ldots, (h_1,b_0), (h_1,b_1), \ldots$$

But `view(batch_size, num_heads, ...)` interprets them as **batch-major**:

$$(b_0,h_0), (b_0,h_1), \ldots, (b_1,h_0), (b_1,h_1), \ldots$$

This means the attention output for one sample/head pair is reassigned to a different sample/head pair when the tensor is reshaped back. Changing the flattening order to `permute(0, 2, 1, 3)` makes the layout batch-major and consistent with the later `view(batch_size, num_heads, ...)`. The mask duplication must also switch from `repeat(...)` to `repeat_interleave(..., dim=0)` so that each sample's padding mask is copied once per head in the same batch-major order.

---

## 3. Summary Table

| # | File | Line(s) | Root Cause | Fix |
|---|------|---------|------------|-----|
| S2-1 | `Optimizers/adam.py` | 53 | Weight decay `alpha=-wd` → anti-regularization | Change to `alpha=wd` |
| S2-2 | `Optimizers/adam.py` | 69 | 2nd moment EMA uses `grad` instead of `grad²` | `addcmul_(grad, grad, value=1-β₂)` |
| S2-3 | `Optimizers/adam.py` | 72–73 | Bias correction: `β·t` instead of `β^t` | Change `*` to `**` |
| S2-4 | `Optimizers/sgd.py` | 39 | Weight decay `alpha=-wd` → anti-regularization | Change to `alpha=wd` |
| S2-5 | `Losses/loss.py` | 13 | `qa_ce_loss` sums losses without 0.5 average | Add `0.5 *` |
| S2-6 | `Models/encoder.py` | 78 | `self.scale` computed but never applied to attention | Multiply `bmm` output by `self.scale` |
| S2-7 | `Models/encoder.py` | 85 | Permute `(1,2,0,3)` mixes batch and head dims | Change to `(0,2,1,3)` |
| S2-8 | `Models/encoder.py` | 70–76 | Head-major flattening conflicts with batch-major restore/mask order | Flatten as `(B, H, L, d_k)` and use `repeat_interleave` for masks |

**Total: 8 LOGIC bugs across 4 files**

---

## 4. Files Modified

```
data/5329_Assignment1_2026/
├── Optimizers/
│   ├── adam.py        ← S2-1, S2-2, S2-3
│   └── sgd.py         ← S2-4
├── Losses/
│   └── loss.py        ← S2-5
└── Models/
    └── encoder.py     ← S2-6, S2-7, S2-8
```

---

## 5. Expected Impact

| Bug | Training Effect Without Fix |
|-----|-----------------------------|
| S2-1 (Adam weight decay) | Regulariser grows parameters → reduces effective weight decay benefit |
| S2-2 (Adam 2nd moment) | No adaptive learning rate → Adam degenerates; large gradient parameters under-updated |
| S2-3 (Adam bias correction) | Negative correction after step ~1 → sign flip → divergence |
| S2-4 (SGD weight decay) | Same anti-regularisation; SGD model overfits more easily |
| S2-5 (CE loss scale) | 2× gradient magnitude → 2× effective LR → potentially unstable training |
| S2-6 (Attention scale) | Soft attention collapses to hard attention; vanishing attention gradients |
| S2-7 (Attention permute) | Representations contaminated across batch samples; attention output wrong for every sample |
| S2-8 (Head flatten order) | Attention rows and masks no longer align with their original `(batch, head)` pairs; reconstructed multi-head outputs mix examples across the batch |
