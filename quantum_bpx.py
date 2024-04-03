# Copyright (C) 2024 Matthias Deiml, Daniel Peterseim - All rights reserved

import qiskit as qk
from qiskit.circuit.library import QFT, MCXGate, RYGate
from qiskit_aer import AerSimulator, Aer

import numpy as np
import scipy.sparse as sp

from qsp import qsp, Wx_to_R
from block_encoding import CPiNot, BlockEncoding, _mcx_nonsymmetric


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
    else:
        qc.append(QFT(N, do_swaps=False).to_gate(), y[:])
        for j in range(N):
            lam = np.pi / (2**j)
            qc.cp(lam, x, y[j])
        qc.append(QFT(N, do_swaps=False).inverse().to_gate(), y[:])

    return qc


def CRShift(LOG_N, DIM):
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
    a = qk.AncillaRegister(1, name="a")

    if DIM == 1:
        qc = qk.QuantumCircuit(j, k, l, g, name="C")
    else:
        qc = qk.QuantumCircuit(j, k, s, l, g, a, name="C")
        qc.h(s)

    qc.h(k)

    qc.append(CRShift(LOG_L, DIM), j[:] + l[:])
    for d in range(DIM):
        qc.append(CInc(L), [k[d]] + j[d*L:(d+1)*L])
    qc.append(CRShift(LOG_L, DIM).inverse(), j[:] + l[:])

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
    a = qk.AncillaRegister(1, name="a")

    if DIM == 1:
        qc = qk.QuantumCircuit(j, k, l, a)
        qc.append(C(LOG_L, DIM), j[:] + k[:] + l[:] + a[:])
        qc.append(T(LOG_L, DIM), j[:] + l[:] + [a[0]])
    else:
        qc = qk.QuantumCircuit(j, k, s, l, g, a, name="TC")
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
        CX_dom.cswap(l[1], j[0], j[2])

        CX_dom.rccx(l[0], j[0], ae[2])
        CX_dom.cx(l[0], ae[2])

        CX_dom.rccx(l[1], ae[0], ae[3])
        CX_dom.rccx(ae[0], ae[1], ae[3])
        CX_dom.cx(l[1], ae[3])

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
    # Inversion and measurement procedure
    ######

    # QSP angles and normalization factor

    # J = 12
    angles = Wx_to_R(np.array([-2.2283320972846186, 0.792370970962418, 0.045284559228113475, -0.674546025818336, -0.14232550307650094, 0.4900788479211249, -0.195484349444351, 1.9408850703466565, -0.11525779173228821, 0.03926750904844137, 0.6854824933793874, -2.8996835577783746, 1.1822706331366992, -0.3742605291971066, -0.3742605291962322, 1.1822706331327244, 0.24190909581331566, 0.6854824933862489, 0.039267509059546735, -0.11525779171341075, -1.2007075832432634, -0.19548434946293824, 0.49007884791560474, -0.14232550306996183, -0.674546025822347, 0.04528455921685026, 0.7923709709617691, -0.6575357704863706]))
    poly = [1.03687311e+01, -2.20816205e+02, 2.87661244e+03, -2.48949886e+04, 1.48758550e+05, -6.28269017e+05, 1.90372474e+06, -4.16986054e+06, 6.59722842e+06, -7.45485567e+06, 5.86005248e+06, -3.04156146e+06, 9.36486890e+05, -1.29475352e+05]
    qsp_norm = 4.31117904

    # J = 6
    # angles = Wx_to_R(np.array([-2.1726508503438953, 0.25829346822873966, 0.9713592720773033, 0.5663055620532766, -0.8974363445219233, -0.1602820187317635, -0.1602820187317482, 2.2441563090678787, -2.5752870915365187, 0.9713592720773214, 0.2582934682287247, -0.6018545235490137]))
    # poly = [6.51555955, -49.98310509, 173.26705273, -295.99810341, 243.22713684, -76.75916282]
    # qsp_norm = 3.37104786

    # J = 5
    # angles = Wx_to_R(np.array([-0.6076998438496887, 0.609208329716424, -0.11400911596534327, -1.0925147543305, -0.18747923204621375, -0.18747923204621175, 2.049077899259295, -0.11400911596533464, -2.5323843238733543, 0.9630964829452084]))
    # poly = [5.76934379, -33.95294717, 82.03012371, -86.07933114, 32.58188909]
    # qsp_norm = 3.32527027

    # Construct the pseudoinverse C_F^+ using the above angles
    qc_inv = qsp(qc, angles)
    print(qc_inv.simplify()._U)

    # Right hand and quantity of interset
    test_b = np.ones((2 ** L - 1) ** DIM) / 2 ** L
    test_m = np.zeros_like(test_b)
    test_m[8] = 1

    # Apply preconditioner and normalize rhs and quantity of interest
    test_Pb = P.T @ test_b
    test_Pm = P.T @ test_m

    norm_rhs = np.linalg.norm(test_Pb) * np.linalg.norm(test_Pm)

    test_b /= np.linalg.norm(test_Pb)
    test_m /= np.linalg.norm(test_Pm)

    test_Pb /= np.linalg.norm(test_Pb) * np.sqrt(2)
    test_Pm /= np.linalg.norm(test_Pm) * np.sqrt(2)

    # Construct the quantum circuit for the measurement precedure
    ancillas = max(qc_inv._U.num_ancillas, qc_inv._CX_img.num_ancillas())

    # The initial state of the quantum computer which already contains the
    # right hand sides and quantity of interest vectors.
    # This will initial state will be applied artificially in the simulator
    initial_state = np.zeros(2 ** (qc_inv._embedding_size + ancillas))
    initial_state[P_dom] = test_Pb
    initial_state[P_dom + 2 ** qc_inv._embedding_size] = test_Pm

    x = qk.QuantumRegister(qc_inv._embedding_size)
    a = qk.QuantumRegister(ancillas)
    b = [a[0]]
    c = qk.ClassicalRegister(2)

    test_qc = qk.QuantumCircuit(x, a, c)

    # In a real world application you would do something like the following
    # test_qc.h(b)
    # test_qc.append(qc_r.control(1, ctrl_state=0), b[:] + x[:])
    # test_qc.append(qc_m.control(1, ctrl_state=1), b[:] + x[:])

    # Instead we just set the simulator to the corresponding state
    test_qc.set_statevector(initial_state)

    test_qc.h(b)
    test_qc.measure(b, c[0])
    test_qc.reset(b)

    # Apply the pseudoinverse
    test_qc.append(qc_inv._U, x[:] + a[:qc_inv._U.num_ancillas])
    test_qc.append(qc_inv._CX_img.to_circuit(True), x[:] + b[:] + a[1:qc_inv._CX_img.num_ancillas()+1])

    # The coefficent could be applied here

    test_qc.x(b)
    test_qc.measure(b, c[1])

    # Print the complete circuit
    test_qc = test_qc.decompose("CX_Pi")
    print(test_qc)

    # The two-qubit errors that we want to simulate
    noises = np.array([0, 1e-5, 1e-4, 10 ** (-3.5), 1e-3, 1e-2])

    # Number of runs per value in noises
    runs = 200

    # Number of shots per run
    shots = 1000

    results = np.zeros((len(noises), runs))

    for (i, noise) in enumerate(noises):
        noisy_backend = AerSimulator(noise_model=define_noise_model(noise * 1e-2, noise))
        print("Transpiling...")
        circ = qk.transpile(test_qc, noisy_backend, optimization_level=3)
        print("Running...")
        for run in range(runs):
            result = noisy_backend.run(circ, shots=shots).result()
            counts = result.get_counts()
            print(counts)

            total = 0
            for k in counts.keys():
                total += counts[k]

            a = counts["00"] if "00" in counts else 0
            b = counts["01"] if "01" in counts else 0
            result = (a - b) / total

            # Reverse normalization
            result *= qsp_norm ** 2
            result /= normalization ** 2
            result *= norm_rhs

            print(result)
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

    # Output reference solution (should be 63/512 for the given example)
    ref = np.dot(test_m, np.linalg.solve(ref_S(L, DIM), test_b)) * norm_rhs
    print(f"true solution = {ref}")
