# Copyright (C) 2024 Matthias Deiml, Daniel Peterseim - All rights reserved

import qiskit as qk
from qiskit.circuit.library import QFT, MCXGate, RYGate, StatePreparation
from qiskit_aer import AerSimulator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import SamplerV2 as Sampler, QiskitRuntimeService, Session

import numpy as np
import scipy.sparse as sp

from qsp import qsp, Wx_to_R
from block_encoding import CPiNot, BlockEncoding, _mcx_nonsymmetric, _run_circuit


def CInc(N):
    """
    Circuit implementing a increment operation of the second N-qubit register
    controlled by the first quibt register.

    Optimized variants are implemented for N = 1, 2
    """
    x = qk.QuantumRegister(1, name="x")
    y = qk.QuantumRegister(N, name="y")

    qc = qk.QuantumCircuit(x, y, name="c_inc")

    if N == 1:
        qc.cx(x, y)
    elif N == 2:
        qc.ccx(x, y[0], y[1])
        qc.cx(x, y[0])
    elif N == 4:
        a = qk.QuantumRegister(2, name="a")

        qc = qk.QuantumCircuit(x, y, a, name="c_inc")
        qc.rccx(y[0], x, a[0])
        qc.rccx(y[1], y[2], a[1])
        qc.ccx(a[0], a[1], y[3])
        qc.rccx(y[1], y[2], a[1])
        qc.ccx(a[0], y[1], y[2])
        qc.cx(a[0], y[1])
        qc.rccx(y[0], x, a[0])
        qc.cx(x, y[0])
    else:
        qc.append(QFT(N, do_swaps=False).to_gate(), y[:])
        for j in range(N):
            lam = np.pi / (2**j)
            qc.cp(lam, x, y[j])
        qc.append(QFT(N, do_swaps=False).inverse().to_gate(), y[:])

    return qc


def CRShift(LOG_N, DIM, uncompute_control=False):
    """
    Circuit implementing a bitwise right shift of the first register by the
    second register.

    The first register is interpreted as consisting of DIM registers with N
    qubits each, all of which are shifted.
    """

    N = 2**LOG_N
    x = qk.QuantumRegister(N * DIM, name="x")
    y = qk.QuantumRegister(LOG_N, name="y")

    qc = qk.QuantumCircuit(x, y, name="shift")
    if LOG_N == 1:
        for d in range(DIM):
            qc.cswap(y[0], x[2*d], x[2*d + 1])
    else:
        for d in range(DIM):
            for i in range(0, N, 2):
                qc.cswap(y[0], x[N*d + i], x[N*d + i+1])
        next = CRShift(LOG_N-1, DIM)
        qc.append(next, x[::2] + y[1:])
        qc.append(CInc(LOG_N-1), y[:])

        qc.append(next, x[1::2] + y[1:])
        if uncompute_control:
            qc.append(CInc(LOG_N-1).inverse(), y[:])

    return qc.decompose(["c_inc", "shift"])


def C(LOG_L, DIM):
    """
    The part of the block encoding of C_F consisting of the matrices C_l.

    The matrices for all l are combined for a more efficient implementation.
    """

    L = 2 ** LOG_L
    LOG_DIM = int(np.ceil(np.log2(DIM)))

    assert 2 ** LOG_DIM == DIM

    s = qk.QuantumRegister(LOG_DIM, name="s")
    k = qk.QuantumRegister(DIM, name="k")
    j = qk.QuantumRegister(L * DIM, name="j")
    l = qk.QuantumRegister(LOG_L, name="l")
    g = qk.QuantumRegister(DIM, name="g")
    a = qk.AncillaRegister(2, name="a")

    if DIM == 1:
        qc = qk.QuantumCircuit(j, k, l, a, name="C")
    else:
        qc = qk.QuantumCircuit(j, k, s, l, g, a, name="C")
        qc.h(s)

    qc.h(k)

    specialize = DIM == 1 and LOG_L == 2

    if not specialize:
        qc.append(CRShift(LOG_L, DIM), j[:] + l[:])
    for d in range(DIM):
        qc.append(CInc(L), [k[d]] + j[d*L:(d+1)*L] + a[:])
    qc.append(CRShift(LOG_L, DIM, uncompute_control=specialize).inverse(), j[:] + l[:])

    for d in range(DIM):
        if DIM == 1:
            # Specialization for DIM == 1
            qc.h(k[d])
            qc.z(k[d])
            qc.x(k[d])
        else:
            qc.append(MCXGate(LOG_DIM, ctrl_state=d), s[:] + [a[0]])
            qc.cz(k[d], a[0])
            qc.h(k[d])
            qc.append(
                RYGate(2*np.arccos(np.sqrt(1/3))).control(2, ctrl_state="01"),
                [k[d], a[0], g[d]])
            qc.ccx(k[d], a[0], g[d])
            qc.append(MCXGate(LOG_DIM, ctrl_state=d), s[:] + [a[0]])

    return qc


def T(LOG_L, DIM):
    """
    The part of the block encoding of C_F consisting of the matrices T_{l,L}.
    """

    # Implementation of T_{l,l+1,1D}
    T1 = qk.QuantumCircuit(2, name="T1")
    T1.x(1)
    T1.swap(0, 1)
    T1.cry(np.arccos(0.5)*2+np.pi, 1, 0)
    T1.h(1)
    T1.cz(0, 1)

    L = 2 ** LOG_L

    k = qk.QuantumRegister(DIM, name="k")
    j = qk.QuantumRegister(L * DIM, name="j")
    l = qk.QuantumRegister(LOG_L, name="l")
    a = qk.AncillaRegister(1, name="a")

    if DIM == 1:
        qc = qk.QuantumCircuit(j, l, a, name="T")
    else:
        qc = qk.QuantumCircuit(j, k, l, a, name="T")

    if DIM == 1 and LOG_L == 2:
        # Specialization for most optimized case
        qc.cry(-np.pi / 2, l[1], j[0])
        qc.cry(-np.pi / 2, l[1], j[1])
        qc.cswap(l[1], j[0], j[2])
        qc.cry(-np.pi / 2, l[0], j[0])
        qc.cswap(l[1], j[0], j[2])

        return qc

    for j1 in reversed(range(1, L)):
        qc.append(MCXGate(LOG_L, ctrl_state=j1), l[:] + [a[0]])
        if DIM == 1:
            # Specialization for DIM == 1
            qc.cry(-np.pi / 2, a[0], j[j1-1])
        else:
            for d in range(DIM):
                qc.append(T1.control(1), [a[0], k[d], j[j1 + d * L-1]])
    qc.append(MCXGate(LOG_L, ctrl_state=0), l[:] + [a[0]])
    qc.x(a[0])

    return qc


def C_F(LOG_L, DIM):
    """
    Block encoding of C_F
    """

    L = 2 ** LOG_L
    LOG_DIM = int(np.ceil(np.log2(DIM)))

    assert 2 ** LOG_DIM == DIM

    j = qk.QuantumRegister(L * DIM, name="j")
    k = qk.QuantumRegister(DIM, name="k")
    s = qk.QuantumRegister(LOG_DIM, name="s")
    l = qk.QuantumRegister(LOG_L, name="l")
    g = qk.QuantumRegister(DIM, name="g")
    a = qk.AncillaRegister(2, name="a")

    if DIM == 1:
        qc = qk.QuantumCircuit(j, k, l, a, name="C_F")
        qc.append(C(LOG_L, DIM), j[:] + k[:] + l[:] + a[:])
        qc.append(T(LOG_L, DIM), j[:] + l[:] + [a[0]])
    else:
        qc = qk.QuantumCircuit(j, k, s, l, g, a, name="C_F")
        qc.append(C(LOG_L, DIM), j[:] + k[:] + s[:] + l[:] + g[:] + a[:])
        qc.append(T(LOG_L, DIM), j[:] + k[:] + l[:] + [a[0]])

    qc.h(l)

    # Most optimized case
    specialize = LOG_L == 2 and DIM == 1

    if DIM == 1:
        embedding_size = 1 + L + LOG_L
        indices_k = [i + L * DIM for i in range(DIM)]
        indices_l = [i + (1 + L) * DIM for i in range(LOG_L)]
        indices_g = indices_k

        zero_bits = [] if specialize else indices_k

        needed_ancilla = 5 if specialize else (LOG_L + 1)
    else:
        embedding_size = (1 + L) * DIM + LOG_L + LOG_DIM + DIM
        indices_k = [i + L * DIM for i in range(DIM)]
        indices_s = [i + (1 + L) * DIM for i in range(LOG_DIM)]
        indices_l = [i + (1 + L) * DIM + LOG_DIM for i in range(LOG_L)]
        indices_g = [i + (1 + L) * DIM + LOG_DIM + LOG_L for i in range(DIM)]

        zero_bits = indices_k + indices_s + indices_g

        needed_ancilla = DIM + LOG_L + 1

    a_extra = qk.AncillaRegister(max(0, needed_ancilla - len(zero_bits)))
    ae = zero_bits + a_extra[:]

    b = qk.QuantumRegister(1, name="b")

    # Construction of C_Pi_NOT operations
    if specialize:
        CX_dom = qk.QuantumCircuit(j, k, l, b, a_extra, name="Pi_j")
        CX_dom.rccx(j[0], j[1], ae[0])
        CX_dom.rccx(j[2], j[3], ae[1])
        CX_dom.cswap(l[1], j[1], j[3])

        CX_dom.x(j[3])
        CX_dom.rccx(l[0], j[3], ae[2])

        CX_dom.rccx(ae[0], ae[1], ae[3])
        CX_dom.x(ae[1])
        CX_dom.rccx(l[1], ae[1], ae[3])

        CX_dom.x(ae[2])
        CX_dom.x(ae[3])
        CX_dom.rccx(ae[2], ae[3], ae[4])

        CX_dom.x(k[0])
        CX_dom.rccx(k[0], ae[4], b[0])
    else:

        a1 = qk.AncillaRegister(1)

        if DIM == 1:
            CX_dom = qk.QuantumCircuit(j, k, l, b, a_extra, a1, name="Pi_j")
        else:
            CX_dom = qk.QuantumCircuit(j, k, s, l, g, b, a_extra, a1, name="Pi_j")

        CX_dom.append(MCXGate(len(zero_bits), ctrl_state=0), zero_bits[:] + [a1[0]])
        for i in reversed(range(LOG_L)):
            CX_dom.cx(l[i], ae[i], ctrl_state=0)
            bits = [j[i1 + L * d] for i1 in range(2 ** i) for d in range(DIM)]
            CX_dom.append(_mcx_nonsymmetric(len(bits) + 1, False), bits + [l[i], ae[i]])
            if i != 0:
                for i1 in range(2 ** (i-1)):
                    for d in range(DIM):
                        CX_dom.cswap(l[i], j[i1 + d*L], j[i1 + 2 ** i + d*L])
        if DIM == 1:
            CX_dom.append(_mcx_nonsymmetric(L, False), j[:] + [ae[0]])
        else:
            for d in range(DIM):
                CX_dom.append(_mcx_nonsymmetric(L, False), j[d*L:(d+1)*L] + [ae[d + LOG_L]])
                CX_dom.x(ae[d + LOG_L])
        CX_dom.append(_mcx_nonsymmetric(needed_ancilla-1, False), ae[:needed_ancilla])
        CX_dom.rccx(a1[0], ae[needed_ancilla-1], b)

    CX_dom = CPiNot(embedding_size, CX_dom)

    return BlockEncoding(qc, CX_dom=CX_dom, CX_img=CPiNot(embedding_size, indices_l + indices_g))


def ref_Q(L, L_MAX):
    """
    The matrix reperesentation of V_L -> V_{L_MAX}
    """
    N = 2 ** L - 1
    M = 2 ** L_MAX - 1

    step = 2 ** (L_MAX - L)

    j = np.arange(1, step)
    stencil = np.zeros(2 * step - 1)
    stencil[step - 1] = 1
    stencil[step - 1 - j] = (step - j) / step
    stencil[step - 1 + j] = (step - j) / step

    j, i = np.meshgrid(np.arange(len(stencil)), np.arange(N))
    indices = (j + i * step).flatten()
    indptr = np.arange(N+1) * (2 * step - 1)
    data = np.repeat([stencil], N, axis=0).flatten()

    return sp.csr_matrix((data, indices, indptr), shape=(N,M)).toarray()


def ref_P(L, DIM):
    """
    The rectangular preconditioner (F in the paper)
    """
    P = []
    for j in reversed(range(1, L+1)):
        Q_mat = ref_Q(j, L)

        X = np.array([[1]])
        for d in range(DIM):
            X = np.kron(X, Q_mat)

        P.append(2 ** (-j * (2 - DIM)/2) * X)
    P = np.vstack(P)

    return P.T


def ref_S_1d(L):
    """
    1d stiffness matrix
    """
    N = 2 ** L - 1
    return 2 ** L * (2 * np.eye(N) - 1 * (np.eye(N, k=1) + np.eye(N, k=-1)))


def ref_C_1d(L):
    """
    1d gradient
    """
    N = 2 ** L - 1
    return 2 ** (L/2) * (np.eye(N+1, N) - np.eye(N+1, N, k=-1))


def ref_CM_1d(L):
    """
    1d "half" mass matrix
    """
    N = 2 ** L - 1
    a = 1 / (2 * np.sqrt(3))
    b = 1 / 2
    return 2 ** L * np.tensordot(np.array([b, a]), np.eye(N+1, N), 0) + np.tensordot(np.array([b, -a]), np.eye(N+1, N, k=-1), 0)


def ref_M_1d(L):
    """
    1d mass matrix
    """
    N = 2 ** L - 1
    return 2 ** (2 * L) * (2/3) * np.eye(N) + (1/6) * (np.eye(N, k=1) + np.eye(N, k=-1))


def ref_C(L, DIM):
    """
    D-dimensional gradient
    """
    N = (2 ** L - 1) ** DIM

    C1d = np.tensordot(np.array([1, 0]), ref_C_1d(L), 0)
    M1d = ref_CM_1d(L)
    print(C1d.shape)
    print(M1d.shape)

    Ys = []
    for i in range(DIM):
        Y = np.array([[[1]]])
        for j in range(DIM):
            if i == j:
                Y = np.kron(C1d, Y)
            else:
                Y = np.kron(M1d, Y)
        print(Y.shape)
        Ys.append(np.reshape(Y, (-1, N)))
    X = np.vstack(Ys)

    return X


def ref_S(L, DIM):
    """
    D-dimensional stiffenss matrix
    """
    N = (2 ** L - 1) ** DIM
    X = np.zeros((N, N))

    S1d = ref_S_1d(L)
    M1d = ref_M_1d(L)

    for i in range(DIM):
        Y = np.array([[1]])
        for j in range(DIM):
            if i == j:
                Y = np.kron(Y, S1d)
            else:
                Y = np.kron(Y, M1d)
        X += Y

    return X


def ref_PAP(L, DIM):
    """
    symetrically preconditioned stiffness matrix
    """
    P = ref_P(L, DIM)
    S = ref_S(L, DIM)
    return P.T @ S @ P


def ref_CP(L, DIM):
    """
    Preconditioned gradient
    """
    P = ref_P(L, DIM)
    C = ref_C(L, DIM)
    return C @ P


def ref_CP_1d(L, DIM):
    """
    Preconditioned 1-dimensional gradient

    This has a slightly different indexing than ref_CP(L, 1)
    """
    P = ref_P(L, DIM)
    C = ref_C_1d(L)
    return C @ P


def define_noise_model(error1, error2):
    """
    Noise model with given one- and two-qubit gate errors
    """

    from qiskit_aer.noise import depolarizing_error, NoiseModel

    error_gate1 = depolarizing_error(error1, 1)
    error0 = depolarizing_error(0, 1)
    error_gate2 = depolarizing_error(error2, 2)

    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(error0, "reset")
    noise_model.add_all_qubit_quantum_error(error0, "measure")
    noise_model.add_all_qubit_quantum_error(error_gate1, ["u1", "u2", "u3"])
    noise_model.add_all_qubit_quantum_error(error_gate2, ["cx"])

    return noise_model


if __name__ == "__main__":

    np.set_printoptions(
        suppress=True, precision=3, threshold=np.inf, linewidth=1000)

    LOG_L = 2
    DIM = 1
    L = 2 ** LOG_L

    ######
    # Block encoding of C_F
    ######

    qc = C_F(LOG_L, DIM)

    # Output circuit for block encoding of C_F

    simp = qc.simplify(True)
    out = simp._U.decompose(["C", "C1d", "T", "c_T1", "CX_Pi_dg", "Pi_j_dg"], reps=2)
    print(out)

    # Test matrix encoded by circuit ...

    C_exp, P_dom, P_img = qc.get_encoded_matrix(return_projections=True, projection_hint=None)
    C_exp_norm = np.linalg.norm(C_exp, ord=2)

    # ...and compare it with classically computed reference

    if DIM == 1:
        C_ref = ref_CP_1d(L, DIM)
    else:
        C_ref = ref_CP(L, DIM)
    C_ref_norm = np.linalg.norm(C_ref, ord=2)

    print(f"subnormalization = {1/C_exp_norm}")
    normalization = C_ref_norm/C_exp_norm
    print(f"normalization = {normalization}")

    error_frob = np.linalg.norm(C_ref - C_exp * normalization, ord="fro")
    error_spec = np.linalg.norm(C_ref - C_exp * normalization, ord=2)

    print(f"Error (frob): {error_frob}")
    print(f"Error (spec): {error_spec}")

    # Compute the effective condition of the encoded matrix

    P = ref_P(L, DIM)
    Q, _R = np.linalg.qr(P.T)

    print(f"Effective condition: {1/np.linalg.norm(C_exp @ Q, ord=-2)}")

    ######
    # Right hand side and quantity of interest
    ######

    # Right hand and quantity of interset
    test_b = np.ones((2 ** L - 1) ** DIM) / 2 ** L
    test_m = test_b.copy()

    # Apply preconditioner and normalize rhs and quantity of interest
    test_Pb = P.T @ test_b
    test_Pm = P.T @ test_m

    norm_rhs = np.linalg.norm(test_Pb) * np.linalg.norm(test_Pm)

    test_b /= np.linalg.norm(test_Pb)
    test_m /= np.linalg.norm(test_Pm)

    test_Pb /= np.linalg.norm(test_Pb) * np.sqrt(2)
    test_Pm /= np.linalg.norm(test_Pm) * np.sqrt(2)

    # Construct circuit for m and b
    initial_state = np.zeros(2 ** 8)
    initial_state[P_dom] = test_Pb
    initial_state[P_dom + 2 ** 7] = test_Pm

    qc_b = qk.QuantumCircuit(8, name="b")

    level_norms = np.sqrt(2) * np.array([np.linalg.norm(initial_state[i*2**5:(i+1)*2**5]) for i in range(4)])

    qc_b.prepare_state(level_norms, [5, 6])

    for i in range(4):
        qc_b.ccx(5, 6, 7, ctrl_state=i)
        qc_b.ry(np.pi, 3-i)
        # qc_b.cx(6, 3-i, ctrl_state=0)
        angle = np.arctan(np.sqrt((2**(3-i)-1)/2**(3-i)))
        if i == 3:
            qc_b.ry(-np.pi / 2, 3-i)
            qc_b.append(RYGate(2 * (angle - np.pi/4)).control(3), list(range(1, 4)) +  [3-i])
        else:
            qc_b.cry(-np.pi / 2, 7, 3-i)
            qc_b.append(RYGate(2 * (angle - np.pi/4)).control(1 + i), [7] + list(range(4-i, 4)) +  [3-i])
        # qc_b.append(HGate().control(i + 1), [6] + list(range(4-i, 4)) +  [3-i])
    qc_b.x(7)

    # Verify circuit for m and b
    x = np.zeros(2**qc_b.num_qubits)
    x[0] = 1
    res = _run_circuit(qc_b, x)[:2**7]
    print(f"Error in right hand side: {np.linalg.norm(initial_state[:2**7] * np.sqrt(2) - res)}")

    ######
    # QSP & Solver
    ######

    # J = 2
    angles = Wx_to_R(np.array([0.4840389288693594, -0.5882474595254135, -0.5882474595254125, 2.0548352556642566]))

    # J = 3
    # angles = Wx_to_R(np.array([-1.3159339251463265, -0.4393974349416314, 0.4452378475344494, 0.44523784753445184, -0.4393974349416263, 0.25486240164856544]))

    # J = 4
    # angles = Wx_to_R(np.array([0.3329987485149657, -0.0023918349460009813, -2.247065453959354, 0.48132185830505314, 0.4813218583050566, 0.8945271996304389, -0.00239183494600137, -1.2377975782799293]))

    # J = 5
    # angles = Wx_to_R(np.array([-0.6076998438496887, 0.609208329716424, -0.11400911596534327, -1.0925147543305, -0.18747923204621375, -0.18747923204621175, 2.049077899259295, -0.11400911596533464, -2.5323843238733543, 0.9630964829452084]))

    qc_inv = qsp(qc, angles)

    c = qk.ClassicalRegister(4 + 7, name="c")
    x = qk.QuantumRegister(qc._embedding_size, name="x")
    ancillas = qc_inv._U.num_ancillas
    a = qk.QuantumRegister(ancillas, name="a")
    b_qsp = qk.QuantumRegister(1, name="b_qsp")

    qc_inv_b = qk.QuantumCircuit(x, b_qsp, a, c)

    # Prepare state
    qc_inv_b.append(qc_b, x[:] + [a[0]])

    # Apply scaled matrix inverse
    qc_inv_b.append(qc_inv._U, x[:] + b_qsp[:] + a[:qc_inv._U.num_ancillas])

    # "Copy" state, so that we can check it is in the projected space later
    for i in range(3):
        qc_inv_b.cx(x[4+i], a[-3+i])

    # Apply C_F again, to obtain a state of which the norm relates to the quantity of interest
    qc_inv_b.append(qc.transpose()._U, x[:] + a[:qc._U.num_ancillas])

    # Measure relevant bits
    qc_inv_b.measure(x, c[4:])
    qc_inv_b.measure(a[-3:] + b_qsp[:], c[:4])

    print(qc_inv_b)

    # The two-qubit errors that we want to simulate
    noises = np.array([0, 1e-5, 1e-4, 7e-3, 1e-3, 1e-2])

    # Number of runs per value in noises
    runs = 200

    # Number of shots per run
    shots = 10000

    results = np.zeros((len(noises), runs))

    for (i, noise) in enumerate(noises):
        noisy_backend = AerSimulator(noise_model=define_noise_model(noise * 1e-2, noise))
        print("Transpiling...")
        circ = qk.transpile(qc_inv_b, noisy_backend, optimization_level=3)
        print("Running...")
        for run in range(runs):
            result = noisy_backend.run(circ, shots=shots).result()
            counts = result.get_counts()

            a = 0
            b = 0
            for k in counts.keys():
                if k[-4:] != "0000":
                    continue
                a += counts[k]
                if k[2] != "0":
                    continue
                if k[3:7] == "1111":
                    continue
                level = int(k[0:2], 2)
                if k[3:3+level] == (level * "1"):
                    b += counts[k]

            result = a / b

            # Reverse normalization
            result /= normalization ** 2
            result *= norm_rhs

            print(f"result: {result}")

            results[i, run] = result

    # Compute and store statistics of measured results

    res_avg = np.average(results, axis=1)
    res_max = np.max(results, axis=1)
    res_min = np.min(results, axis=1)
    res_hi = np.percentile(results, 97.5, axis=1)
    res_lo = np.percentile(results, 2.5, axis=1)
    output = np.stack([noises, res_avg, res_max, res_min, res_lo, res_hi])
    np.savetxt(
        "noise_error.csv",
        output.T,
        fmt="%s;",
        delimiter=''
    )

    ref = np.dot(test_m, np.linalg.solve(ref_S(L, DIM), test_b)) * norm_rhs
    print(f"true solution = {ref}")
