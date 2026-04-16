# Geometric Shuffle Control for Belief State Regression

A control experiment to test whether transformers represent belief state structure *beyond* local next-token prediction.

## Motivation

The main result of Shai et al. (2024) is that transformers trained on HMM-generated data linearly represent the **belief state geometry** in their residual stream. But a natural question arises: is this representation merely sufficient for next-token prediction, or does it capture richer structure about the entire future?

Many distinct belief states can share the same next-token distribution while differing in their distributions over the *entire* future. If transformers only needed to represent next-token information, these states could be collapsed. The geometric shuffle control tests whether the transformer distinguishes them.

**Core idea:** Decompose belief states into a component that determines next-token prediction and an orthogonal component. Shuffle the orthogonal component while preserving the next-token component. If regression quality degrades, the transformer must be encoding the orthogonal structure.

---

## Mathematical Framework

### Setup (following paper notation)

Let the data-generating process be an edge-emitting HMM with:
- Hidden states $\mathcal{S} = \{s_1, \ldots, s_{|\mathcal{S}|}\}$
- Token vocabulary $\mathcal{X} = \{x_1, \ldots, x_{|\mathcal{X}|}\}$  
- Token-labeled transition matrices $\{T^{(x)}\}_{x \in \mathcal{X}}$ where $T^{(x)}_{i,j} = \Pr(x, s_j | s_i)$

A **belief state** $\eta \in \Delta^{|\mathcal{S}|-1}$ is a probability distribution over hidden states. Given belief state $\eta$ and observed token $x$, the updated belief state is (Eq. 1 from paper):

$$\eta' = \frac{\eta T^{(x)}}{\eta T^{(x)} \mathbf{1}}$$

where $\mathbf{1}$ is a column vector of ones ensuring normalization.

### Emission Matrix

Define the **emission matrix** $E \in \mathbb{R}^{|\mathcal{S}| \times |\mathcal{X}|}$ where:

$$E_{s,x} = \Pr(x | s) = \sum_{s'} T^{(x)}_{s, s'} = \left( T^{(x)} \mathbf{1} \right)_s$$

The **next-token distribution** given belief state $\eta$ is:

$$p_{\text{next}}(\eta) = \eta E \in \Delta^{|\mathcal{X}|-1}$$

This is a linear map from belief states to next-token distributions.

### Next-Token Equivalence

Two belief states $\eta, \eta'$ are **next-token equivalent** if and only if:

$$\eta E = \eta' E$$

The set of belief states equivalent to $\eta$ forms an affine subspace:

$$[\eta] = \{\eta' : (\eta - \eta') \in \ker(E^\top)\}$$

where $\ker(E^\top) = \{v \in \mathbb{R}^{|\mathcal{S}|} : E^\top v = \mathbf{0}\}$ is the left null space of $E$. 

Note: For row vectors $v$, the condition $vE = 0$ is equivalent to $E^\top v^\top = 0$, so the "directions that don't affect next-token prediction" live in $\ker(E^\top)$.

### Orthogonal Decomposition

For any belief state $b \in \mathbb{R}^{|\mathcal{S}|}$, we decompose:

$$b = b^{\parallel} + b^{\perp}$$

where:
- $b^{\parallel} \in \text{col}(E)$ determines the next-token distribution
- $b^{\perp} \in \ker(E^\top)$ is orthogonal to next-token prediction (changes here don't affect $bE$)

The **projection onto $\text{col}(E)$** is:

$$P_E = E (E^\top E)^{-1} E^\top$$

The **projection onto $\ker(E^\top)$** is:

$$P_E^{\perp} = I - P_E$$

Note: Use the Moore-Penrose pseudoinverse if $E^\top E$ is singular.

**Notation clarification:** Following standard convention, $\ker(E^\top) = \{v : E^\top v = 0\}$ where $v$ is a column vector. This equals the left null space of $E$: for any $v \in \ker(E^\top)$, treating $v^\top$ as a row vector gives $v^\top E = (E^\top v)^\top = 0$. Thus $\ker(E^\top)$ contains exactly the directions that don't affect next-token prediction—if we perturb a belief state $b$ by adding any $v^\top$ where $v \in \ker(E^\top)$, the next-token distribution $(b + v^\top)E = bE$ is unchanged.

### Working in Centered Coordinates

To preserve the simplex constraint $\mathbf{1}^\top b = 1$, we work in centered coordinates.

Given dataset of belief states $\{b_i\}_{i=1}^N$, compute the empirical mean:

$$\bar{b} = \frac{1}{N} \sum_{i=1}^N b_i$$

Define centered beliefs:

$$\Delta b_i = b_i - \bar{b}$$

These satisfy $\mathbf{1}^\top \Delta b_i = 0$ (they live in the simplex tangent space).

Decompose:

$$\Delta b_i = \Delta b_i^{\parallel} + \Delta b_i^{\perp}$$

where:

$$\Delta b_i^{\parallel} = P_E \, \Delta b_i, \qquad \Delta b_i^{\perp} = P_E^{\perp} \, \Delta b_i$$

---

## The Geometric Shuffle

### Construction

Given:
- Activation-belief pairs $\{(a_i, b_i)\}_{i=1}^N$ where $a_i \in \mathbb{R}^{d_{\text{resid}}}$
- Projection matrices $P_E$, $P_E^{\perp}$
- Random permutation $\sigma: \{1, \ldots, N\} \to \{1, \ldots, N\}$

Construct **shuffled belief targets**:

$$\tilde{b}_i = \bar{b} + \Delta b_i^{\parallel} + \Delta b_{\sigma(i)}^{\perp}$$

### Properties

The shuffled targets preserve:

1. **Simplex constraint:** $\mathbf{1}^\top \tilde{b}_i = 1$ (since $\mathbf{1}^\top \bar{b} = 1$ and $\mathbf{1}^\top \Delta b = 0$)

2. **Marginal distribution:** The set $\{\tilde{b}_i\}$ has approximately the same empirical distribution as $\{b_i\}$

3. **Next-token pairing:** $\tilde{b}_i E = b_i E$ (the activation $a_i$ is still paired with the correct next-token distribution)

The shuffled targets destroy:

4. **Orthogonal pairing:** The correspondence between $a_i$ and $\Delta b_i^{\perp}$ is randomized

---

## Hypothesis Test

### Regression Setup

Following Section 2.3 of the paper, find an affine map from activations to belief states:

$$\hat{b} = W a + c$$

where $W \in \mathbb{R}^{|\mathcal{S}| \times d_{\text{resid}}}$ and $c \in \mathbb{R}^{|\mathcal{S}|}$ minimize mean squared error.

### Test Statistic

Compute:
- $\text{MSE}_{\text{original}}$: regression error with original targets $\{b_i\}$
- $\text{MSE}_{\text{control}}$: regression error with shuffled targets $\{\tilde{b}_i\}$

### Interpretation

| Result | Interpretation |
|--------|----------------|
| $\text{MSE}_{\text{control}} \approx \text{MSE}_{\text{original}}$ | Transformer only encodes next-token-relevant information ($b^{\parallel}$) |
| $\text{MSE}_{\text{control}} \gg \text{MSE}_{\text{original}}$ | Transformer encodes full belief geometry including beyond-next-token structure ($b^{\perp}$) |

For statistical significance, repeat the shuffle $K$ times with different random permutations and compare the distribution of control MSEs to the original MSE.

---

## Pseudocode

### 1. Compute Emission Matrix

```
function compute_emission_matrix(T_matrices, num_states):
    """
    Construct E where E[s,x] = Pr(x|s) = sum_j T^(x)[s,j]
    
    Args:
        T_matrices: dict mapping token x -> T^(x), each shape (|S|, |S|)
        num_states: |S|
    
    Returns:
        E: shape (|S|, |X|)
    """
    tokens = sorted(T_matrices.keys())
    E = zeros(num_states, len(tokens))
    
    for x_idx, x in enumerate(tokens):
        for s in range(num_states):
            E[s, x_idx] = sum(T_matrices[x][s, :])  # sum over destination states
    
    return E
```

### 2. Compute Projection Matrices

```
function compute_projections(E):
    """
    Compute projectors onto col(E) and ker(E^T).
    
    P_E projects onto col(E), the subspace relevant for next-token prediction.
    P_E_perp projects onto ker(E^T), the orthogonal subspace (beyond next-token).
    
    Args:
        E: emission matrix, shape (|S|, |X|)
    
    Returns:
        P_E: shape (|S|, |S|}), projection onto col(E)
        P_E_perp: shape (|S|, |S|), projection onto ker(E^T)
    """
    # P_E = E (E^T E)^{-1} E^T  (standard projection onto column space)
    ETE = E.T @ E                    # shape (|X|, |X|)
    ETE_inv = pseudoinverse(ETE)     # use pinv for numerical stability
    P_E = E @ ETE_inv @ E.T          # shape (|S|, |S|)
    
    # P_E_perp = I - P_E projects onto ker(E^T) = col(E)^⊥
    P_E_perp = eye(num_states) - P_E
    
    return P_E, P_E_perp
```

### 3. Decompose Belief States

```
function decompose_beliefs(belief_states, P_E, P_E_perp):
    """
    Decompose belief states into components in col(E) and ker(E^T).
    
    Args:
        belief_states: array of shape (N, |S|), each row is a belief state
        P_E: projection onto col(E)
        P_E_perp: projection onto ker(E^T)
    
    Returns:
        b_mean: shape (|S|,)
        delta_b_parallel: shape (N, |S|), components in col(E)
        delta_b_perp: shape (N, |S|), components in ker(E^T)
    """
    N = len(belief_states)
    
    # Compute empirical mean
    b_mean = mean(belief_states, axis=0)
    
    # Center the data
    delta_b = belief_states - b_mean  # shape (N, |S|)
    
    # Project onto col(E) and ker(E^T)
    # For row vectors: delta_b @ P, since P is symmetric
    delta_b_parallel = delta_b @ P_E       # component that determines next-token
    delta_b_perp = delta_b @ P_E_perp      # component orthogonal to next-token
    
    return b_mean, delta_b_parallel, delta_b_perp
```

### 4. Construct Shuffled Targets

```
function geometric_shuffle(belief_states, P_E, P_E_perp, rng):
    """
    Create shuffled targets: b̃_i = b̄ + Δb_i^∥ + Δb_{σ(i)}^⊥
    
    Args:
        belief_states: array of shape (N, |S|)
        P_E, P_E_perp: projection matrices
        rng: random number generator
    
    Returns:
        shuffled_beliefs: array of shape (N, |S|)
    """
    # Decompose
    b_mean, delta_b_parallel, delta_b_perp = decompose_beliefs(
        belief_states, P_E, P_E_perp
    )
    
    # Random permutation
    N = len(belief_states)
    sigma = rng.permutation(N)
    
    # Shuffle perpendicular components
    delta_b_perp_shuffled = delta_b_perp[sigma]
    
    # Reconstruct: b̃_i = b̄ + Δb_i^∥ + Δb_{σ(i)}^⊥
    shuffled_beliefs = b_mean + delta_b_parallel + delta_b_perp_shuffled
    
    return shuffled_beliefs
```

### 5. Linear Regression

```
function fit_affine_regression(activations, targets):
    """
    Find W, c minimizing ||targets - (activations @ W^T + c)||^2
    
    Args:
        activations: shape (N, d_resid)
        targets: shape (N, |S|)
    
    Returns:
        W: shape (|S|, d_resid)
        c: shape (|S|,)
    """
    N = len(activations)
    
    # Augment with ones for bias term
    A_aug = hstack([activations, ones((N, 1))])  # shape (N, d_resid + 1)
    
    # Solve least squares
    params = lstsq(A_aug, targets)  # shape (d_resid + 1, |S|)
    
    W = params[:-1, :].T  # shape (|S|, d_resid)
    c = params[-1, :]     # shape (|S|,)
    
    return W, c


function compute_mse(activations, targets, W, c):
    """
    Compute mean squared error.
    """
    predictions = activations @ W.T + c
    mse = mean(sum((targets - predictions)^2, axis=1))
    return mse
```

### 6. Full Control Experiment

```
function run_control_experiment(activations, belief_states, T_matrices, 
                                 num_shuffles=1000, seed=42):
    """
    Test whether transformer encodes beyond-next-token structure.
    
    Args:
        activations: residual stream activations, shape (N, d_resid)
        belief_states: ground truth beliefs, shape (N, |S|)
        T_matrices: token-labeled transition matrices
        num_shuffles: number of random shuffles for control distribution
        seed: random seed
    
    Returns:
        results: dict with MSE values and diagnostics
    """
    num_states = belief_states.shape[1]
    rng = RandomGenerator(seed)
    
    # Step 1: Build emission matrix and projections
    E = compute_emission_matrix(T_matrices, num_states)
    P_E, P_E_perp = compute_projections(E)
    
    # Step 2: Original regression
    W_orig, c_orig = fit_affine_regression(activations, belief_states)
    mse_original = compute_mse(activations, belief_states, W_orig, c_orig)
    
    # Step 3: Control shuffles
    mse_controls = []
    for _ in range(num_shuffles):
        shuffled = geometric_shuffle(belief_states, P_E, P_E_perp, rng)
        W_ctrl, c_ctrl = fit_affine_regression(activations, shuffled)
        mse_ctrl = compute_mse(activations, shuffled, W_ctrl, c_ctrl)
        mse_controls.append(mse_ctrl)
    
    # Step 4: Compute statistics
    mse_control_mean = mean(mse_controls)
    mse_control_std = std(mse_controls)
    
    # Effect size: how much worse is control?
    effect = (mse_control_mean - mse_original) / mse_original
    
    # Diagnostics
    b_mean, delta_parallel, delta_perp = decompose_beliefs(
        belief_states, P_E, P_E_perp
    )
    var_parallel = var(delta_parallel)
    var_perp = var(delta_perp)
    
    return {
        'mse_original': mse_original,
        'mse_control_mean': mse_control_mean,
        'mse_control_std': mse_control_std,
        'effect_size': effect,
        'variance_parallel': var_parallel,
        'variance_perp': var_perp,
        'dim_kernel_E_T': num_states - matrix_rank(E),  # dim(ker(E^T))
        'p_value': mean(mse_controls <= mse_original)  # one-sided
    }
```

---

## Diagnostics

### Variance Decomposition

Report the fraction of belief state variance in each subspace:

$$r = \frac{\text{Var}(\Delta b^{\perp})}{\text{Var}(\Delta b^{\parallel}) + \text{Var}(\Delta b^{\perp})}$$

If $r \approx 0$, there is little beyond-next-token structure to test. If $r$ is substantial, the test has power.

### Dimensionality

Report $\dim(\ker(E^\top)) = |\mathcal{S}| - \text{rank}(E)$, the dimensionality of the beyond-next-token subspace.

For RRXOR: $|\mathcal{S}| = 5$, $|\mathcal{X}| = 2$, so $\text{rank}(E) \leq 2$ and $\dim(\ker(E^\top)) \geq 3$.

For Mess3: $|\mathcal{S}| = 3$, $|\mathcal{X}| = 3$, so $\text{rank}(E) \leq 3$ and $\dim(\ker(E^\top)) \geq 0$.

### Direct Perpendicular Regression

As an additional test, regress activations directly onto $\Delta b^{\perp}$:

```
W_perp, c_perp = fit_affine_regression(activations, delta_b_perp)
mse_perp = compute_mse(activations, delta_b_perp, W_perp, c_perp)

# Compare to shuffled baseline
mse_perp_shuffled = compute_mse(activations, delta_b_perp[shuffle], W_perp, c_perp)
```

If $\text{MSE}_{\perp} \ll \text{MSE}_{\perp,\text{shuffled}}$, activations linearly encode the orthogonal component.

---

## Expected Results

### RRXOR Process

- Many belief states share identical next-token distributions
- Large $\dim(\ker(E^\top))$
- **Expected:** Significant MSE increase under shuffle ($\text{MSE}_{\text{control}} \gg \text{MSE}_{\text{original}}$)

### Mess3 Process

- Depends on specific transition parameters
- If most belief states have unique next-token distributions, smaller effect expected
- Check $\text{Var}(\Delta b^{\perp})$ to assess test power

---

## References

Shai, A. S., Marzen, S. E., Teixeira, L., Gietelink Oldenziel, A., & Riechers, P. M. (2024). Transformers Represent Belief State Geometry in their Residual Stream. *NeurIPS 2024*. arXiv:2405.15943