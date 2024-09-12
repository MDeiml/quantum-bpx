# Copyright (C) 2024 Matthias Deiml, Daniel Peterseim - All rights reserved

# This file compares the performance of the QSVT linear solver applied to
# the preconditioend and non-preconditioned stiffness matrix.

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.special import bdtrc
from scipy.linalg import qr


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

        X = sp.eye(1)
        for d in range(DIM):
            X = sp.kron(X, Q_mat)

        P.append(2 ** (-j * (2 - DIM)/2) * X)
    P = sp.vstack(P)

    return P.T


def ref_S_1d(L):
    """
    1d stiffness matrix
    """
    N = 2 ** L - 1
    return 2 ** L * (2 * sp.eye(N) - 1 * (sp.eye(N, k=1) + sp.eye(N, k=-1)))


def ref_C_1d(L):
    """
    1d gradient
    """
    N = 2 ** L - 1
    return 2 ** (L/2) * (sp.eye(N+1, N) - sp.eye(N+1, N, k=-1))


def ref_CM_1d(L):
    """
    1d "half" mass matrix
    """
    N = 2 ** L - 1
    a = 1 / (2 * np.sqrt(3))
    b = 1 / 2
    return 2 ** (-L/2) * (sp.kron(sp.csr_array(np.array([[b], [a]])), sp.eye(N+1, N)) + sp.kron(sp.csr_array(np.array([[b], [-a]])), sp.eye(N+1, N, k=-1)))


def ref_M_1d(L):
    """
    1d mass matrix
    """
    N = 2 ** L - 1
    return 2 ** (-L) * ((2/3) * sp.eye(N) + (1/6) * (sp.eye(N, k=1) + sp.eye(N, k=-1)))


def ref_C(L, DIM):
    """
    D-dimensional gradient
    """
    # N = (2 ** L - 1) ** DIM

    C1d = sp.kron(sp.csr_array(np.array([[1], [0]])), ref_C_1d(L))
    M1d = ref_CM_1d(L)

    Ys = []
    for i in range(DIM):
        Y = np.array([[1]])
        for j in range(DIM):
            if i == j:
                Y = sp.kron(C1d, Y)
            else:
                Y = sp.kron(M1d, Y)
        # Ys.append(np.reshape(Y, (-1, N)))
        Ys.append(Y)
    X = sp.vstack(Ys)

    return X


def ref_S(L, DIM):
    """
    D-dimensional stiffenss matrix
    """
    N = (2 ** L - 1) ** DIM
    X = sp.csr_array((N, N))

    S1d = ref_S_1d(L)
    M1d = ref_M_1d(L)

    for i in range(DIM):
        Y = sp.eye(1)
        for j in range(DIM):
            if i == j:
                Y = sp.kron(Y, S1d)
            else:
                Y = sp.kron(Y, M1d)
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


def solver_poly(b, J):
    coef = []

    for j in range(J + 1):
        coef.append(0)
        coef.append(4 * (-1) ** j * bdtrc(b + j, 2 * b, 0.5))

    return np.polynomial.Chebyshev(coef)


def mat_poly(p, A, b):
    Ta = b
    Tb = A @ b
    sol = np.zeros_like(Tb)
    for i in range(1, len(p.coef), 2):
        sol += Tb * p.coef[i]
        Ta = 2 * A.T @ Tb - Ta
        Tb = 2 * A @ Ta - Tb
    return sol


def cond(M):
    return 1 / sp.linalg.eigsh(M, k=1, which="SM", return_eigenvectors=True)[0][0]


L_MAX = 14
DIM = 1

half_matrices = True

S = ref_S(L_MAX, DIM)

b = 2 ** (-L_MAX * DIM) * np.ones(S.shape[1])
m = 2 ** (-L_MAX * DIM) * np.ones(S.shape[1])

x = sp.linalg.spsolve(S, b)
sol = np.dot(x, m)
print(sol)

print("Without preconditioning")

steps_no_prec = []
err_no_prec = []

for L in range(2, 10):

    print(f"L = {L}")

    S = ref_S(L, DIM)

    b = 2 ** (-L * DIM) * np.ones(S.shape[1])
    m = 2 ** (-L * DIM) * np.ones(S.shape[1])

    normalization = sp.linalg.eigsh(S, k=1)[0][0]

    kappa_actual = np.sqrt(cond(S / normalization))

    if half_matrices:
        normalization = np.sqrt(normalization)
        S = ref_C(L, DIM)

    eps = 2 ** (-L)

    for kappa in np.linspace(kappa_actual * 0.8, kappa_actual, 20):
        B = int(np.ceil(kappa ** 2 * np.log(kappa/eps)))
        J = int(np.ceil(np.sqrt(B * np.log(4 * B / eps))))
        p = solver_poly(B, J)
        x_no_prec = mat_poly(p, S / normalization, b) / normalization
        if half_matrices:
            y_no_prec = mat_poly(p, S / normalization, m) / normalization
        else:
            y_no_prec = m
        err = (sol - np.dot(y_no_prec, x_no_prec)) / sol
        if err < eps:
            print(f"kappa = {kappa}")
            print(f"{J} steps")
            print(err)
            err_no_prec.append(err)
            steps_no_prec.append(J)
            break

print("With preconditioning")

steps_prec = []
err_prec = []
for L in range(2, L_MAX):

    print(f"L = {L}")

    S = ref_S(L, DIM)

    b = 2 ** (-L * DIM) * np.ones(S.shape[1])
    m = 2 ** (-L * DIM) * np.ones(S.shape[1])

    PAP = ref_PAP(L, DIM)
    normalization = 4 * DIM * L
    actual_norm = sp.linalg.eigsh(PAP, k=1)[0][0]
    print(normalization, actual_norm)
    assert normalization >= actual_norm

    CP = ref_CP(L, DIM)
    P = ref_P(L, DIM)

    if half_matrices:
        normalization = np.sqrt(normalization)
        PAP = CP

    for kappa in np.linspace(2, 6, 20):

        eps = 2 ** (-L)
        B = int(np.ceil(kappa ** 2 * np.log(kappa/eps)))
        J = int(np.ceil(np.sqrt(B * np.log(4 * B / eps))))
        # if J > B:
        #     break
        p = solver_poly(B, J)

        Pb = P.T @ b
        Pm = P.T @ m

        x_prec = mat_poly(p, PAP / normalization, Pb) / normalization
        if half_matrices:
            y_prec = mat_poly(p, PAP / normalization, Pm) / normalization
        else:
            y_prec = Pm

        temp = abs((sol - np.dot(y_prec, x_prec)) / sol)

        if temp < eps:

            print(f"kappa = {kappa}")
            print(f"{J} steps")
            print(temp)

            err_prec.append(temp)
            steps_prec.append(J)
            break


plt.loglog(steps_no_prec, err_no_prec)
plt.loglog(steps_prec, err_prec)
plt.show()

plt.semilogy(steps_no_prec, err_no_prec)
plt.semilogy(steps_prec, err_prec)
plt.xlim(np.min(steps_prec)*0.8, np.max(steps_prec)*2)
plt.show()

output = np.zeros((4, L_MAX - 2))
output[0, :len(steps_no_prec)] = np.array(steps_no_prec)
output[1, :len(err_no_prec)] = np.array(err_no_prec)
output[2, :] = np.array(steps_prec)
output[3, :] = np.array(err_prec)

np.savetxt(
    "condition.csv",
    output.T,
    fmt="%s;",
    delimiter=''
)
