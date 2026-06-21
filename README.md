# Entanglement Growth from Entangled States: A Unified Perspective on Entanglement Generation and Transport

[![arXiv](https://img.shields.io/badge/arXiv-2510.08344-b31b1b.svg)](https://arxiv.org/abs/2510.08344)

This repository provides the code and data for the paper: **"Entanglement Growth from Entangled States: A Unified Perspective on Entanglement Generation and Transport"** by Chun-Yue Zhang, Zi-Xiang Li, and Shi-Xin Zhang.

Copyright (c) 2025-2026 Chun-Yue Zhang. The code is released under the [Apache License 2.0](LICENSE).

## Repository Structure

The `src/` directory contains two Python scripts:
*   `HCEE_for_various_dynamics.py`: Generates the main numerical results presented in **Fig. 1(d,e)** and **Fig. 2** of the paper.
*   `BAEE_dynamics.py`: Produces the results for **Fig. 3**.

These scripts utilize the QuSpin library to simulate entanglement dynamics via exact diagonalization. The output files contain the parameter settings, random configurations for each sample, and the calculated data.

The data used to plot the figures in the main text are provided in the `data/` directory. They are stored as Python dictionaries in `.pkl` format. Each dictionary contains several keys corresponding to different data arrays. You can load the data using the pickle module, for example:

```python
import pickle

with open('FIG1d.pkl','rb') as f:
    data=pickle.load(f)

print(data)
```

## Citation

If you use this code in your research, please cite the original paper:

```bibtex
@misc{zhang2025entanglementgrowthentangledstates,
      title={Entanglement Growth from Entangled States: A Unified Perspective on Entanglement Generation and Transport}, 
      author={Chun-Yue Zhang and Zi-Xiang Li and Shi-Xin Zhang},
      year={2025},
      eprint={2510.08344},
      archivePrefix={arXiv},
      primaryClass={quant-ph},
      url={https://arxiv.org/abs/2510.08344}, 
}
```
