# Correlation Clustering of Organoid Images
This repository contains the code for the paper "Correlation Clustering of Organoid Images", accepted at GCPR'24.

- Data: [https://doi.org/10.25532/OPARA-598](https://doi.org/10.25532/OPARA-598) (CC BY 4.0)

If you find the paper or source code valuable, please cite
```bibtex
@inproceedings{presberger-2024-organoids,
    author = {Jannik Presberger
        and Rashmiparvathi Keshara
        and David Stein
        and Yung Hae Kim
        and Anne Grapin-Botton
        and Bjoern Andres},
    title = {Correlation Clustering of Organoid Images},
    year = {2024},
    booktitle = {GCPR},
    doi = {10.1007/978-3-031-85181-0_15}
}
```

## Authors
- **Jannik Presberger** (TU Dresden): partial quadratic assignment problem (PQAP) approach, [`code-pqap`](code-pqap)
- **David Stein** (TU Dresden): twin neural networks and correlation clustering, [`code-twin-networks`](code-twin-networks)

## Repository Structure
```md
.
├── code-twin-networks   # twin neural networks (TNI, TNH) + correlation clustering (Python, C++)
├── code-pqap            # partial quadratic assignment problem (PQAP) approach (C++)
├── data                 # image data, filled by download_data.sh
├── download_data.sh     # downloads and verifies the data from OPARA
├── CITATION.cff         # software citation metadata
├── .zenodo.json         # Zenodo metadata
└── LICENSE
```

## Data Preparation
Download the data with
```shell
./download_data.sh
```
This downloads the four image sets (about 433 MB) from [OPARA](https://doi.org/10.25532/OPARA-598), checks their MD5 checksums and unpacks them into
```md
data
├── train-100
├── test-100
├── test-30
└── test-unlabeled
```
To download only some of the sets, pass their names, e.g. `./download_data.sh train-100 test-100 test-30`
(`test-unlabeled` is not needed to reproduce the experiments in the paper).
The script also links the sets into `code-pqap/data/images/image_sets` under the names used by the PQAP code
(`organoid_dataset_100_split_1` = Train-100, `organoid_dataset_100_split_2` = Test-100,
`organoid_dataset_unseen_classes` = Test-30, `organoid_dataset_with_unseen_classes` = Test-100 + Test-30).

## Requirements
- Linux, CMake >= 3.12, a C++17 compiler
- [Gurobi](https://www.gurobi.com/) (tested with 11.0.2 and 12.0.2) with a valid license; set `GUROBI_HOME`
- HDF5 with C++ bindings (e.g. `hdf5-devel` / `libhdf5-dev`)
- Twin networks: Python 3.9 to 3.12, see `code-twin-networks/requirements.txt`
- PQAP: additionally MPI, OpenCV and the Python development headers with NumPy

## Twin Neural Networks
See [`code-twin-networks`](code-twin-networks/README.md).

## PQAP
See [`code-pqap`](code-pqap/README.md).

## License
The code is released under the MIT License, see [`LICENSE`](LICENSE).
Third-party header libraries in `include/` directories keep their own licenses (listed in `LICENSE`).
The data is released under CC BY 4.0 by its rights holder.
