"""
This module contains functions for creating transition matrices for various processes.
"""
import numpy as np

def get_matrix_from_args(name: str, **kwargs):
    process_functions = {
        "post_quantum": post_quantum,
        "tom_quantum": tom_quantum,
        "fanizza": fanizza,
        "rrxor": rrxor,
        "quantum_rrxor": quantum_rrxor,
        "mess3": mess3,
        "days_of_week": days_of_week,
        "zero_one_random": zero_one_random
    }
    
    if name in process_functions:
        return process_functions[name](**kwargs)
    else:
        raise ValueError(f"Invalid process name: {name}")
    
def zero_one_random(p=0.5):
    """
    Creates a transition matrix for the Zero-One Random Process.
    Which has 3 states: "0", "1", and "R"
    """
    T = np.zeros((2, 3, 3))
    state_names = {"0": 0, "1": 1, "R": 2}
    T[0, state_names["0"], state_names["1"]] = 1.0
    T[1, state_names["1"], state_names["R"]] = 1.0
    T[0, state_names["R"], state_names["0"]] = p
    T[1, state_names["R"], state_names["0"]] = 1 - p

    return T

def post_quantum(alpha=np.exp(1), beta=0.5):
    """
    Creates a transition matrix for the Post Quantum Process.
    """
    # Validate conditions for alpha and beta
    if not (alpha > 1 > beta > 0):
        raise ValueError("Condition alpha > 1 > beta > 0 not satisfied")
    if alpha + beta == 2:
        raise ValueError("Condition alpha + beta ≠ 2 not satisfied")
    if np.isclose(np.log(alpha) / np.log(beta), np.round(np.log(alpha) / np.log(beta))):
        raise ValueError("Condition ln(alpha) / ln(beta) ∉ ℚ not satisfied")

    T = np.zeros((3, 3, 3))
    m0 = np.array([[1], [1], [0]])  # Column vector
    mu0 = np.array([[1, -1, -1]])  # Row vector
    
    def _intermediate_matrix(val):
        return np.array([
            [val, 0, 0],
            [0, 1, 0],
            [0, np.log(val), 1]
        ])
    
    T[0] = np.outer(m0, mu0)
    T[1] = _intermediate_matrix(alpha)
    T[2] = _intermediate_matrix(beta)

    # Normalize T such that T[0] + T[1] + T[2] has largest abs eigenvalue = 1
    T_sum = T.sum(axis=0)
    T_sum_max_eigval = np.abs(np.linalg.eigvals(T_sum)).max()
    T /= T_sum_max_eigval

    # Verify that T[0] + T[1] + T[2] has largest abs eigenvalue = 1
    T_sum_normalized = T.sum(axis=0)
    T_sum_max_eigval = np.abs(np.linalg.eigvals(T_sum_normalized)).max()
    np.testing.assert_almost_equal(T_sum_max_eigval, 1, decimal=10, 
                                   err_msg="Largest absolute eigenvalue is not 1")

    return T
def days_of_week():
    """
    Creates a transition matrix for the Days of the Week Process.
    """
    T = np.zeros((11, 7, 7)) # emission, from, to

    d = {"M": 0, "Tu": 1, "W": 2, "Th": 3, "F": 4, "Sa": 5, "Su": 6, "Tmrw": 7, "Yest": 8, "Wknd": 9, "Wkdy": 10}
    all_days = ["M", "Tu", "W", "Th", "F", "Sa", "Su"]
    wkdy_days = [d["M"], d["Tu"], d["W"], d["Th"], d["F"]]
    wknd_days = [d["Sa"], d["Su"]]
    for day in all_days:
        T[d[day], :, d[day]] = 1.0
    
    for wkdy in wkdy_days:
        T[d["Wkdy"], :, wkdy] = 1/len(wkdy_days)
    for wknd in wknd_days:
        T[d["Wknd"], :, wknd] = 1/len(wknd_days)

    for i, day in enumerate(all_days):
        next_day = all_days[(i + 1) % len(all_days)]
        prev_day = all_days[i - 1]
        T[d["Tmrw"], d[day], d[next_day]] = 1.0
        T[d[next_day], d[day], d[next_day]] = 5.0
        T[d["Yest"], d[day], d[prev_day]] = 1.0

    # normalize so that  T.sum(axis=0) is row stochastic
    T_sum = T.sum(axis=0)
    T_row_sums = T_sum.sum(axis=0)
    T /= T_row_sums

    return T

def tom_quantum(alpha: float, beta: float):
    """
    Creates a transition matrix for the Tom Quantum Process.
    """
    # Create a 4x3x3 array filled with zeros
    T= np.zeros((4, 3, 3))
    
    # Common elements
    gamma = 1/(2*np.sqrt(alpha**2+beta**2))
    common_diag = 1/4
    middle_diag = (alpha**2 - beta**2) * gamma**2
    off_diag = 2 * alpha * beta * gamma**2
    
    # G^(0)
    T[0] = np.array([
        [common_diag, 0, off_diag],
        [0, middle_diag, 0],
        [off_diag, 0, common_diag]
    ])
    
    # G^(1)
    T[1] = np.array([
        [common_diag, 0, -off_diag],
        [0, middle_diag, 0],
        [-off_diag, 0, common_diag]
    ])
    
    # G^(2)
    T[2] = np.array([
        [common_diag, off_diag, 0],
        [off_diag, common_diag, 0],
        [0, 0, middle_diag]
    ])
    
    # G^(3)
    T[3] = np.array([
        [common_diag, -off_diag, 0],
        [-off_diag, common_diag, 0],
        [0, 0, middle_diag]
    ])
    
    return T


def fanizza(alpha: float, lamb: float):
    """
    Creates a transition matrix for the Faniza Process.
    """
    # Calculate intermediate values
    a_la = (1 - lamb * np.cos(alpha) + lamb * np.sin(alpha)) / (1 - 2 * lamb * np.cos(alpha) + lamb**2)
    b_la = (1 - lamb * np.cos(alpha) - lamb * np.sin(alpha)) / (1 - 2 * lamb * np.cos(alpha) + lamb**2)

    # Define tau
    tau = np.ones(4)

    # Define the reset distribution pi0
    pi0 = np.array([1 - (2 / (1 - lamb) - a_la - b_la) / 4, 1 / (2 * (1 - lamb)), -a_la / 4, -b_la / 4])

    # Define w
    w = np.array([1, 1 - lamb, 1 + lamb * (np.sin(alpha) - np.cos(alpha)), 1 - lamb * (np.sin(alpha) + np.cos(alpha))])

    # Create Da
    Da = np.outer(w, pi0)

    # Create Db (with the sine sign error fixed)
    Db = lamb * np.array([
        [0, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, np.cos(alpha), -np.sin(alpha)],
        [0, 0, np.sin(alpha), np.cos(alpha)]
    ])

    # Create the transition matrix T
    T = np.zeros((2, 4, 4))
    T[0] = Da
    T[1] = Db

    # Verify that T @ tau = tau (stochasticity condition)
    assert np.allclose(T[0] @ tau + T[1] @ tau, tau), "Stochasticity condition not met"

    return T

def rrxor(pR1=0.5, pR2=0.5):
    """
    Creates a transition matrix for the RRXOR Process.
    """
    T = np.zeros((2, 5, 5))
    s = {"S": 0, "0": 1, "1": 2, "T": 3, "F": 4}
    T[0, s["S"], s["0"]] = pR1
    T[1, s["S"], s["1"]] = 1 - pR1
    T[0, s["0"], s["F"]] = pR2
    T[1, s["0"], s["T"]] = 1 - pR2
    T[0, s["1"], s["T"]] = pR2
    T[1, s["1"], s["F"]] = 1 - pR2
    T[1, s["T"], s["S"]] = 1.0
    T[0, s["F"], s["S"]] = 1.0

    return T

def quantum_rrxor(phi: float, theta: float, epsilon: float = 0.0):
    """
    Creates transition matrices for the Quantum RRXOR process.

    A quantum generalization of the RRXOR process defined by a repeating
    quantum circuit with a persistent memory qubit and a measured ancilla:

        |ψ⟩_mem ──R_y(φ)──●─────────── |ψ'⟩_mem   (persistent)
                           │
        |0⟩_anc ──────────⊕──R_x(θ)──M── x_t      (measured)

    The Kraus operators are K_x = D_x · R_y(φ) where
        D_0 = diag(cos(θ/2), -i·sin(θ/2))
        D_1 = diag(-i·sin(θ/2), cos(θ/2))

    The GHMM lives in the 4-dimensional generalized Bloch representation
    {I/2, σ_x/2, σ_y/2, σ_z/2}. Belief states are 4-vectors (1, b_x, b_y, b_z)
    where (b_x, b_y, b_z) is the Bloch vector of the memory qubit's density
    matrix. The belief geometry is generically a 3-dimensional fractal in the
    Bloch ball.

    Parameters
    ----------
    phi : float
        Memory rotation angle (radians). R_y(φ) rotates the memory qubit
        before entangling with the ancilla. When φ/π is irrational, the
        process has no finite HMM.
    theta : float
        Readout angle (radians). R_x(θ) rotates the ancilla before
        measurement. θ → 0 is uninformative, θ → π is the classical limit.
    epsilon : float, optional
        Leakage parameter in [0, 1]. Mixes each transition with a reset to
        the stationary (fully mixed) state. Default 0.0 (no leakage).

    Returns
    -------
    T : ndarray, shape (2, 4, 4)
        GHMM transition matrices T[x] for tokens x ∈ {0, 1}.

    References
    ----------
    Riechers & Crutchfield (2021), Phys. Rev. Research 3, 013170.
    Riechers, Elliott & Shai (2025), arXiv:2507.07432.
    """
    if not (0 <= epsilon <= 1):
        raise ValueError(f"epsilon must be in [0, 1], got {epsilon}")

    # --- Kraus operators: K_x = D_x @ R_y(phi) ---
    c_t, s_t = np.cos(theta / 2), np.sin(theta / 2)
    c_p, s_p = np.cos(phi / 2), np.sin(phi / 2)

    R_y = np.array([[c_p, -s_p],
                    [s_p,  c_p]], dtype=complex)

    D = [np.diag([c_t, -1j * s_t]),        # D_0
         np.diag([-1j * s_t, c_t])]        # D_1

    K = [D[x] @ R_y for x in range(2)]

    # Verify Kraus completeness: Σ_x K_x†K_x = I
    completeness = sum(k.conj().T @ k for k in K)
    assert np.allclose(completeness, np.eye(2)), \
        f"Kraus completeness violated: {completeness}"

    # --- Pauli basis: {I/2, σ_x/2, σ_y/2, σ_z/2} ---
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
    B = [np.eye(2, dtype=complex) / 2, sigma_x / 2, sigma_y / 2, sigma_z / 2]

    # --- GHMM transition matrices via Bloch representation ---
    # G^(x)_{mn} = 2 tr(B_m K_x B_n K_x†)  maps column Bloch vectors
    # T^(x) = G^(x)^T                        maps row predictive vectors
    # So T^(x)_{ij} = 2 tr(B_j K_x B_i K_x†)
    T = np.zeros((2, 4, 4))
    for x in range(2):
        for i in range(4):
            for j in range(4):
                T[x, i, j] = 2 * np.trace(
                    B[j] @ K[x] @ B[i] @ K[x].conj().T
                ).real

    # --- Verify GHMM properties ---
    T_sum = T[0] + T[1]
    e0 = np.array([1, 0, 0, 0])

    # Right eigenvector: T_sum @ e_0 = e_0  (trace preservation)
    assert np.allclose(T_sum @ e0, e0), \
        f"Right eigenvector check failed: T_sum @ e0 = {T_sum @ e0}"

    # Left eigenvector: e_0 @ T_sum = e_0  (unitality → fully mixed is stationary)
    assert np.allclose(e0 @ T_sum, e0), \
        f"Left eigenvector check failed: e0 @ T_sum = {e0 @ T_sum}"

    # Non-trivial eigenvalues should have magnitude < 1 (ergodicity)
    eigvals = np.linalg.eigvals(T_sum[1:, 1:])
    assert all(np.abs(eigvals) < 1 + 1e-10), \
        f"Non-trivial eigenvalues not contractive: {eigvals}"

    # --- Apply leakage ---
    if epsilon > 0:
        # T^(x)_ε = (1 - ε) T^(x) + (ε / |X|) |1⟩⟩⟨⟨π|
        # where |1⟩⟩ = e_0 (col) and ⟨⟨π| = e_0 (row)
        reset = np.outer(e0, e0)
        for x in range(2):
            T[x] = (1 - epsilon) * T[x] + (epsilon / 2) * reset

    return T


def mess3(x=0.15, a=0.6):
    """
    Creates a transition matrix for the Mess3 Process.
    """
    T = np.zeros((3, 3, 3))
    b = (1 - a) / 2
    y = 1 - 2 * x  

    ay = a * y
    bx = b * x
    by = b * y
    ax = a * x

    T[0, :, :] = [[ay, bx, bx], [ax, by, bx], [ax, bx, by]]
    T[1, :, :] = [[by, ax, bx], [bx, ay, bx], [bx, ax, by]]
    T[2, :, :] = [[by, bx, ax], [bx, by, ax], [bx, bx, ay]]

    return T
