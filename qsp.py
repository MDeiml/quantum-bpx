# Copyright (C) 2024 Matthias Deiml, Daniel Peterseim - All rights reserved

import qiskit as qk
import numpy as np
from qiskit_aer import AerSimulator
from block_encoding import BlockEncoding, CPiNot


def qsp(A: BlockEncoding, angles: list) -> BlockEncoding:

    x = qk.QuantumRegister(A._embedding_size, name="x")
    b = qk.QuantumRegister(1, name="b")
    a = qk.AncillaRegister(
        max(A._U.num_ancillas, A._CX_dom.num_ancillas(), A._CX_img.num_ancillas()), name="a")
    qc = qk.QuantumCircuit(x, b, a, name=A._U.name + "_qsp")

    qc.h(b)

    backend = AerSimulator()
    opt = qk.transpile(A.simplify()._U, backend, optimization_level=3)

    for (i, angle) in enumerate(angles):

        if i == 0:
            # qc.append(A._CX_dom.to_circuit(), x[:] + b[:] + a[:A._CX_dom.num_ancillas()])
            qc.x(b)
            qc.rz(2 * angle, b)
            qc.x(b)
            qc.append(A._U, x[:] + a[:A._U.num_ancillas])
            qc.append(A._CX_img.to_circuit(), x[:] + b[:] + a[:A._CX_img.num_ancillas()])
            continue

        qc.rz(2 * angle, b)

        if i == len(angles) - 1:
            if i % 2 == 0:
                qc.append(A._CX_dom.to_circuit().inverse(), x[:] + b[:] + a[:A._CX_dom.num_ancillas()])
                qc.append(A._U, x[:] + a[:A._U.num_ancillas])
            else:
                qc.append(A._CX_img.to_circuit().inverse(), x[:] + b[:] + a[:A._CX_img.num_ancillas()])
                qc.append(A._U.inverse(), x[:] + a[:A._U.num_ancillas])
        elif i % 2 == 0:
            qc.append(opt, x[:] + b[:] + a[:])
        else:
            qc.append(opt.inverse(), x[:] + b[:] + a[:])

    qc.h(b)

    if len(angles) % 2 == 0:
        return BlockEncoding(
            qc,
            A._CX_dom.extend_embedding(1, False).logical_and(CPiNot(A._embedding_size + 1, [A._embedding_size])),
            A._CX_dom.extend_embedding(1, False).logical_and(CPiNot(A._embedding_size + 1, [A._embedding_size])))
    else:
        return BlockEncoding(
            qc,
            A._CX_dom.extend_embedding(1, False).logical_and(CPiNot(A._embedding_size + 1, [A._embedding_size])),
            A._CX_img.extend_embedding(1, False).logical_and(CPiNot(A._embedding_size + 1, [A._embedding_size])))


def Wx_to_R(angles):
    """
    Convert between Wx and R conventions.

    See [3]
    """
    d = len(angles)-1
    res = np.zeros(d)
    res[0] = angles[0] + angles[-1] + (d - 1) * np.pi/2
    res[1:] = angles[1:d] - np.pi/2

    return res
