# Copyright (C) 2024 Matthias Deiml, Daniel Peterseim - All rights reserved

import qiskit as qk
import qiskit_aer
import numpy as np
from qiskit.circuit.library import MCXGate
from block_encoding import *


def qsp(A: BlockEncoding, angles: list) -> BlockEncoding:

    x = qk.QuantumRegister(A._embedding_size, name="x")
    b = qk.QuantumRegister(1, name="b")
    a = qk.AncillaRegister(
        max(A._U.num_ancillas, A._CX_dom.num_ancillas(), A._CX_img.num_ancillas()), name="a")
    qc = qk.QuantumCircuit(x, b, a, name=A._U.name + "_qsp")

    qc.h(b)

    for (i, angle) in enumerate(reversed(angles)):
        if i % 2 == 0:
            qc.append(A._U, x[:] + a[:A._U.num_ancillas])
            qc.append(A._CX_img.to_circuit(), x[:] + b[:] + a[:A._CX_img.num_ancillas()])
        else:
            qc.append(A._U.inverse(), x[:] + a[:A._U.num_ancillas])
            qc.append(A._CX_dom.to_circuit(), x[:] + b[:] + a[:A._CX_dom.num_ancillas()])

        qc.rz(-2 * angle, b)

        if i % 2 == 0:
            qc.append(A._CX_img.to_circuit().inverse(), x[:] + b[:] + a[:A._CX_img.num_ancillas()])
        else:
            qc.append(A._CX_dom.to_circuit().inverse(), x[:] + b[:] + a[:A._CX_dom.num_ancillas()])

    qc.h(b)

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
