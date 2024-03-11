# Copyright (C) 2024 Matthias Deiml, Daniel Peterseim - All rights reserved

from __future__ import annotations
import qiskit as qk
from qiskit_aer import Aer
import numpy as np
from qiskit.circuit.library import MCXGate, QFT


def _mcx_nonsymmetric(N, ancillas=True):
    if N == 1:
        qc = qk.QuantumCircuit(2, name="ns_mcx")
        qc.cx(0, 1)

        return qc

    if N == 2:
        qc = qk.QuantumCircuit(3, name="ns_mcx")
        qc.rccx(0, 1, 2)

        return qc

    if N == 3:
        qc = qk.QuantumCircuit(4 if not ancillas else 5, name="ns_mcx")
        qc.rcccx(0, 1, 2, 3)

        return qc

    if ancillas:
        x = qk.QuantumRegister(N, name="x")
        b = qk.QuantumRegister(1, name="b")
        a = qk.AncillaRegister(N-2)

        qc = qk.QuantumCircuit(x, b, a, name="ns_mcx")

        l = x[:]
        anc = a[:]

        while len(l) > 2:
            k = len(l) // 2
            for i in range(k):
                qc.rccx(l[2*i], l[2*i+1], anc[i])
            l = l[len(l)-len(l)%2:] + anc[:k]
            anc = anc[k:]
        qc.rccx(l[0], l[1], b)

        return qc
    else:
        x = qk.QuantumRegister(N, name="x")
        b = qk.QuantumRegister(1, name="b")

        qc = qk.QuantumCircuit(x, b, name="ns_mcx")
        qc.mcx(x, b)

        return qc


def _run_circuit(qc, initial, backend=None):
    if backend is None:
        backend = Aer.get_backend("aer_simulator_statevector")
    circ = qk.QuantumCircuit(qc.num_qubits)
    circ.set_statevector(qk.quantum_info.Statevector(initial))
    circ.append(qc, list(range(qc.num_qubits)))
    circ.save_statevector()
    circ = qk.transpile(circ, backend)
    return backend.run(circ).result().get_statevector().data


class CPiNot:

    def __init__(self, embedding_size, data: qk.QuantumCircuit | list = None, ctrl_state: list = None):

        if isinstance(data, qk.QuantumCircuit):
            assert data.num_qubits - data.num_ancillas == embedding_size + 1
            assert data.num_clbits == 0
        if isinstance(data, list):
            assert all([b >= 0 and b <= embedding_size for b in data])
            if ctrl_state is None:
                ctrl_state = [0 for b in data]
            else:
                assert len(ctrl_state) == len(data)
        else:
            assert ctrl_state is None

        self._embedding_size = embedding_size
        self._data = data
        self._ctrl_state = ctrl_state
        self._avoid_ancillas = False

    def __eq__(self, o):
        return self._embedding_size == o._embedding_size and (
            (self._data is None and o._data is None)
            or
            (
                self._data == o._data and
                self._ctrl_state == o._ctrl_state
            )
        )

    def copy(self) -> CPiNot:
        if self._data is None:
            return CPiNot(self._embedding_size, None)
        elif isinstance(self.data, qk.QuantumCircuit):
            return CPiNot(self._embedding_size, self._data.copy())
        else:
            return CPiNot(self._embedding_size, self._data.copy(), self._ctrl_state.copy())

    def num_ancillas(self) -> int:
        if isinstance(self._data, qk.QuantumCircuit):
            return self._data.num_ancillas
        elif self._data is None or self._avoid_ancillas:
            return 0
        else:
            return max(0, len(self._data) - 2)

    def to_circuit(self, symmetric=False) -> qk.QuantumCircuit:
        if isinstance(self._data, qk.QuantumCircuit):
            return self._data
        else:
            x = qk.QuantumRegister(self._embedding_size, name="x")
            b = qk.QuantumRegister(1, name="b")
            a = qk.AncillaRegister(self.num_ancillas(), name="a")
            qc = qk.QuantumCircuit(x, b, a, name="CX_Pi")
            if self._data is None:
                qc.x(b)
            else:
                bits = []
                for (i, cs) in zip(self._data, self._ctrl_state):
                    bits.append(x[i])
                for i in range(len(bits)):
                    if self._ctrl_state[i] == 0:
                        qc.x(bits[i])
                if symmetric:
                    qc.append(MCXGate(len(bits)), bits[:] + b[:])
                    for i in range(len(bits)):
                        if self._ctrl_state[i] == 0:
                            qc.x(bits[i])
                else:
                    qc.append(_mcx_nonsymmetric(len(bits), ancillas=not self._avoid_ancillas), bits[:] + b[:] + a[:])
            return qc.decompose(["ns_mcx"])

    def logical_and(self, other: CPiNot) -> CPiNot:
        assert self._embedding_size == other._embedding_size

        if self._data is None:
            return other
        if other._data is None:
            return self
        if isinstance(self._data, qk.QuantumCircuit) or isinstance(other._data, qk.QuantumCircuit):
            x = qk.QuantumRegister(self._embedding_size, name="x")
            a1 = qk.AncillaRegister(self.num_ancillas())
            a2 = qk.AncillaRegister(other.num_ancillas())
            a3 = qk.AncillaRegister(2)
            b = qk.QuantumRegister(1, name="b")
            qc = qk.QuantumCircuit(x, b, a1, a2, a3, name="CX_Pi")
            qc.append(self.to_circuit(), x[:] + [a3[0]] + a1[:])
            qc.append(other.to_circuit(), x[:] + [a3[1]] + a2[:])
            qc.rccx(a3[0], a3[1], b)

            return CPiNot(self._embedding_size, qc)
        else:
            bits = self._data + other._data
            ctrl_state = self._ctrl_state + other._ctrl_state
            return CPiNot(self._embedding_size, bits, ctrl_state)

    def extend_embedding(self, n: int, front: bool) -> CPiNot:
        if self._data is None:
            return CPiNot(self._embedding_size + n, None)
        elif isinstance(self._data, list):
            if front:
                return CPiNot(self._embedding_size + n, [i + n for i in self._data], self._ctrl_state)
            else:
                return CPiNot(self._embedding_size + n, self._data, self._ctrl_state)
        else:
            x = qk.QuantumRegister(self._embedding_size + n, name="x")
            a = qk.AncillaRegister(self.num_ancillas(), name="a")
            b = qk.QuantumRegister(1, name="b")
            qc = qk.QuantumCircuit(x, b, a)
            qc.append(
                self._data,
                (x[n:] if front else x[:-n]) +
                b[:] +
                a[:]
            )
            return CPiNot(self._embedding_size + n, qc)

    def control(self, ctrl_state: int = 0) -> CPiNot:
        return self.extend_embedding(1, False).logical_and(
            CPiNot(
                self._embedding_size + 1,
                [self._embedding_size],
                [ctrl_state]
            )
        )

    def xor(self, other: CPiNot) -> CPiNot:
        assert self._embedding_size == other._embedding_size

        if self._data is None:
            return other
        if other._data is None:
            return self

        x = qk.QuantumRegister(self._embedding_size, name="x")
        a = qk.QuantumRegister(max(self.num_ancillas(), other.num_ancillas()), name="a")
        b = qk.QuantumRegister(1, name="b")
        qc = qk.QuantumCircuit(x, b, a)
        qc.append(
            self.to_circuit(True),
            x[:] + b[:] + a[:self.num_ancillas()]
        )
        qc.append(
            other.to_circuit(),
            x[:] + b[:] + a[:other.num_ancillas()]
        )
        return CPiNot(self._embedding_size, qc)

    def projected_indices(self, backend=None):
        x = qk.QuantumRegister(self._embedding_size)
        a = qk.QuantumRegister(self.num_ancillas())
        b = qk.QuantumRegister(1)

        circ = qk.QuantumCircuit(x, a, b)
        circ.append(self.to_circuit(), x[:] + b[:] + a[:])

        N1 = 2**self._embedding_size
        N2 = 2**(self._embedding_size + self.num_ancillas())

        data = np.zeros(2 * N2)
        indices = []
        for i in range(N1):
            data[i] = 1
            out = _run_circuit(circ, data, backend)
            data[i] = 0
            if not np.all(np.isclose(out[N2:], 0)):
                indices.append(i)

        return np.array(indices, dtype=np.int32)


class BlockEncoding:

    def __init__(
        self,
        U: qk.QuantumCircuit,
        CX_dom: qk.CPiNot = None,
        CX_img: qk.CPiNot = None,
        normalization: float = 1,
    ) -> BlockEncoding:
        self._embedding_size = U.num_qubits - U.num_ancillas

        if CX_dom is None:
            CX_dom = CPiNot(self._embedding_size, None)
        if CX_img is None:
            CX_img = CPiNot(self._embedding_size, None)

        assert U.num_clbits == 0

        self._U = U
        self._CX_dom = CX_dom
        self._CX_img = CX_img

        assert normalization > 0

        self._normalization = normalization

    def identity(n: int) -> BlockEncoding:
        qc = qk.QuantumCircuit(n)
        return BlockEncoding(qc)

    def copy(self) -> BlockEncoding:
        return BlockEncoding(
            self._U.copy(),
            self._CX_dom.copy(),
            self._CX_img.copy(),
            self._normalization)

    def transpose(self) -> BlockEncoding:
        U = self._U.inverse()
        CX_dom = self._CX_img
        CX_img = self._CX_dom

        U.name = self._U.name + "^T"

        return BlockEncoding(U, CX_dom, CX_img, self._normalization)

    def scale(self, factor: float) -> BlockEncoding:
        res = self.copy()
        res._normalization *= factor
        return res

    def add_subnormalization(self, factor: float) -> BlockEncoding:
        assert factor <= 1 and factor > 0

        x = qk.QuantumRegister(self._embedding_size + 1, name="x")
        a = qk.AncillaRegister(self._U.num_ancillas, name="a")

        U = qk.QuantumCircuit(
            x, a,
            name=self._U.name)
        U.ry(np.arccos(factor) * 2, x[-1])
        U.append(self._U, x[:-1] + a[:])

        U.decompose()

        CX_dom = self._CX_dom.control()
        CX_img = self._CX_img.control()

        normalization = self._normalization / factor

        return BlockEncoding(U, CX_dom, CX_img, normalization)

    def extend(self, n: int, front: bool = False) -> BlockEncoding:
        x = qk.QuantumRegister(self._embedding_size + n, name="x")
        a = qk.AncillaRegister(self._U.num_ancillas, name="a")

        U = qk.QuantumCircuit(
            x, a,
            name=self._U.name)
        if front:
            U.append(self._U, x[n:] + a[:])
        else:
            U.append(self._U, x[:self._embedding_size] + a[:])
        U = U.decompose()

        if front:
            CX_dom = self._CX_dom.extend_embedding(n, True).logical_and(
                CPiNot(self._embedding_size + n, list(range(n)))
            )
            CX_img = self._CX_img.extend_embedding(n, True).logical_and(
                CPiNot(self._embedding_size + n, list(range(n)))
            )
        else:
            CX_dom = self._CX_dom.extend_embedding(n, False).logical_and(
                CPiNot(self._embedding_size + n, list(range(self._embedding_size, self._embedding_size + n)))
            )
            CX_img = self._CX_img.extend_embedding(n, False).logical_and(
                CPiNot(self._embedding_size + n, list(range(self._embedding_size, self._embedding_size + n)))
            )

        return BlockEncoding(U, CX_dom, CX_img, self._normalization)

    def tensor(self, other) -> BlockEncoding:

        ancillas = self._U.num_ancillas + other._U.num_ancillas

        x = qk.QuantumRegister(self._embedding_size + other._embedding_size, name="x")
        a = qk.AncillaRegister(ancillas, name="a")

        U = qk.QuantumCircuit(
            x, a,
            name=self._U.name + "x" + other._U.name)
        U.append(self._U, x[:self._embedding_size] + a[:self._U.num_ancillas])
        U.append(
            other._U,
            x[self._embedding_size:] +
            a[self._U.num_ancillas:ancillas])
        U = U.decompose()

        CX_dom = self._CX_dom.extend_embedding(
            other._embedding_size,
            False
        ).logical_and(
            other._CX_dom.extend_embedding(self._embedding_size, True)
        )
        CX_img = self._CX_img.extend_embedding(
            other._embedding_size,
            False
        ).logical_and(
            other._CX_img.extend_embedding(self._embedding_size, True)
        )

        normalization = self._normalization * other._normalization

        return BlockEncoding(U, CX_dom, CX_img, normalization)

    def multiply(self, other: BlockEncoding) -> BlockEncoding:
        assert self._embedding_size == other._embedding_size
        assert self._CX_dom == other._CX_img

        U = None
        CX_dom = None
        CX_img = None

        if self._CX_dom._data is None:
            ancillas = max(
                self._U.num_ancillas,
                other._U.num_ancillas,
            )

            x = qk.QuantumRegister(
                max(self._embedding_size, other._embedding_size), name="x")
            a = qk.AncillaRegister(ancillas, name="a")

            U = qk.QuantumCircuit(
                x, a,
                name=self._U.name+"*"+other._U.name)
            U.append(
                other._U,
                x[:other._embedding_size] + a[:other._U.num_ancillas])
            U.append(
                self._U,
                x[:self._embedding_size] + a[:self._U.num_ancillas])

            CX_dom = other._CX_dom.copy()
            CX_img = self._CX_img.copy()
        else:
            ancillas = max(
                self._U.num_ancillas,
                other._U.num_ancillas,
                self._CX_dom.num_ancillas()
            )

            x = qk.QuantumRegister(
                max(self._embedding_size, other._embedding_size) + 1, name="x")
            a = qk.AncillaRegister(ancillas, name="a")

            U = qk.QuantumCircuit(
                x, a,
                name=self._U.name+"*"+other._U.name)
            U.append(
                other._U,
                x[:other._embedding_size] + a[:other._U.num_ancillas])
            U.append(
                self._CX_dom.to_circuit(True),
                x[:self._embedding_size] +
                [x[-1]] +
                a[:self._CX_dom.num_ancillas]
            )
            U.x(x[-1])
            U.append(
                self._U,
                x[:self._embedding_size] + a[:self._U.num_ancillas])

            CX_dom = other._CX_dom.control()
            CX_img = self._CX_img.control()
        U = U.decompose()

        normalization = self._normalization * other._normalization

        return BlockEncoding(U, CX_dom, CX_img, normalization)

    def _block_diagonal(
        self,
        other: BlockEncoding,
        pre_h: bool,
        post_h: bool
    ) -> BlockEncoding:
        ratio = self._normalization / other._normalization
        if not pre_h and not post_h:
            assert np.isclose(ratio, 1)
        if pre_h:
            assert self._CX_dom == other._CX_dom
        if post_h:
            assert self._CX_img == other._CX_img

        if pre_h and post_h:
            ratio = np.sqrt(ratio)

        angle = np.arctan(ratio)

        ancillas = max(self._U.num_ancillas, other._U.num_ancillas)

        x = qk.QuantumRegister(
            max(self._embedding_size, other._embedding_size) + 1, name="x")
        a = qk.AncillaRegister(
            ancillas, name="a")

        U = qk.QuantumCircuit(
            x, a,
            name="diag("+self._U.name+","+other._U.name+")")
        if pre_h:
            U.ry(2 * angle, x[-1])
        U.append(
            self._U.control(1, ctrl_state=0),
            [x[-1]] + x[:self._embedding_size] + a[:self._U.num_ancillas])
        U.append(
            other._U.control(1, ctrl_state=1),
            [x[-1]] + x[:other._embedding_size] + a[:other._U.num_ancillas])
        if post_h:
            U.ry(-2 * angle, x[-1])
        U = U.decompose()

        CX_dom = None
        if pre_h:
            CX_dom = self._CX_dom.control(0)
        else:
            CX_dom = self._CX_dom.control(0).xor(other._CX_dom.control(1))

        CX_img = None
        if post_h:
            CX_img = self._CX_img.control(0)
        else:
            CX_img = self._CX_img.control(0).xor(other._CX_img.control(1))

        return BlockEncoding(U, CX_dom, CX_img, self._normalization)

    def block_diagonal(self, other: BlockEncoding) -> BlockEncoding:
        return self._block_diagonal(other, False, False)

    def block_vertical(self, other: BlockEncoding) -> BlockEncoding:
        return self._block_diagonal(other, True, False)

    def block_horizontal(self, other: BlockEncoding) -> BlockEncoding:
        return self._block_diagonal(other, False, True)

    def add(self, other: BlockEncoding) -> BlockEncoding:
        return self._block_diagonal(other, True, True)

    def simplify(self, barrier=False):
        x = qk.QuantumRegister(self._embedding_size, name="x")
        a = qk.AncillaRegister(max(self._U.num_ancillas, self._CX_dom.num_ancillas(), self._CX_img.num_ancillas()), name="a")
        b = qk.QuantumRegister(1, name="b")
        U = qk.QuantumCircuit(x, b, a)
        U.append(self._CX_dom.to_circuit().inverse(), x[:] + b[:] + a[:self._CX_dom.num_ancillas()])
        if barrier:
            U.barrier()
        U.append(self._U, x[:] + a[:self._U.num_ancillas])
        if barrier:
            U.barrier()
        U.append(self._CX_img.to_circuit(), x[:] + b[:] + a[:self._CX_img.num_ancillas()])
        U = U.decompose()

        return BlockEncoding(
            U,
            CPiNot(self._embedding_size + 1, [self._embedding_size]),
            CPiNot(self._embedding_size + 1, [self._embedding_size]),
            normalization=self._normalization
        )

    def get_encoded_matrix(self, backend=None, return_projections=False, projection_hint=None):

        # This assumes that the projection matrix only contains ones and zeros

        if backend is None:
            backend = Aer.get_backend("aer_simulator_statevector")

        if projection_hint is None:
            print("testing projection to domain")
            P_dom = self._CX_dom.projected_indices(backend)
            print(P_dom)
            print("testing projection to image")
            P_img = self._CX_img.projected_indices(backend)
            print(P_img)
        else:
            P_dom = projection_hint[0]
            P_img = projection_hint[1]

        qc = self._U
        flipped = False

        if len(P_dom) > len(P_img):
            qc = qc.inverse()
            flipped = True
            P_dom, P_img = P_img, P_dom

        data = np.zeros(2 ** qc.num_qubits)
        A = np.zeros((len(P_img), len(P_dom)), dtype=np.csingle)
        for (i, index) in enumerate(P_dom):
            data[index] = 1
            A[:, i] = _run_circuit(qc, data, backend)[P_img]
            data[index] = 0

        if flipped:
            A = A.T
            P_dom, P_img = P_img, P_dom

        if return_projections:
            return A, P_dom, P_img
        else:
            return A

    def output_stats(self, backend=None):
        from qiskit.providers.fake_provider import FakePrague
        if backend is None:
            backend = FakePrague()
        circ = qk.transpile(self._U, backend, optimization_level=3)
        print(circ.count_ops())
