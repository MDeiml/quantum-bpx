# Quantum Realization of the Finite Element Method
*Matthias Deiml, Daniel Peterseim, 2024*

Supplementary material to the paper

<https://arxiv.org/abs/2403.19512>

> This paper presents a quantum algorithm for the solution of prototypical second-order linear elliptic partial differential equations discretized by $d$-linear finite elements on Cartesian grids of a bounded $d$-dimensional domain. An essential step in the construction is a BPX preconditioner, which transforms the linear system into a sufficiently well-conditioned one, making it amenable to quantum computation. We provide a constructive proof demonstrating that our quantum algorithm can compute suitable functionals of the solution to a given tolerance $\texttt{tol}$ with a complexity linear in $\texttt{tol}^{-1}$ for a fixed dimension $d$, neglecting logarithmic terms. This complexity is proportional to that of its one-dimensional counterpart and improves previous quantum algorithms by a factor of order $\texttt{tol}^{-2}$. We also detail the design and implementation of a quantum circuit capable of executing our algorithm, and present simulator results that support the quantum feasibility of the finite element method in the near future, paving the way for quantum computing approaches to a wide range of PDE-related challenges.

## Structure of the code

* `quantum_bpx.py` contains the implementation of the circuits described in Section 6 of the paper and the numerical experiment of Section 7.
* `qsp.py` contains code for implementing quantum signal processing (QSP).
* `block_encoding.py` contains code for managing block encodings as described in Section 3 of the paper, as well as the operations of Proposition 3.5.

## Requirements

This code requires the python packages `numpy`, `scipy`, `qiskit`, and `qiskit_aer`. To install them run
```sh
pip3 install numpy scipy qiskit qiskit_aer
```
or
```sh
pip3 install -r requirements.txt
```

The code was tested with the versions `qiskit 0.43.1` and `qiskit_aer 0.12.0`.

## Citation

If you use this code please cite our paper

```bibtex
@article{DP2024quantum,
      title={Quantum Realization of the Finite Element Method}, 
      author={Matthias Deiml and Daniel Peterseim},
      year={2024},
      eprint={2403.19512},
      archivePrefix={arXiv},
      primaryClass={quant-ph}
}
```
