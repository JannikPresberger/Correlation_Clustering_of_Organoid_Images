# Correlation Clustering of Organoid Images - Twin Neural Networks

## Prerequisites
- Gurobi v11.0.2
- HDF5 Header
- You created the correct data directory structure

## Python Virtual Environment
We tested with Python3.9. Run the following commands
```shell
python3.9 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Training the Networks
Run the following command:

```shell
chmod +x train_models.sh # if permission not already set
./train_models.sh
```

## Computing Cost Matrices
Run the following command:

```shell
chmod +x evaluate_models.sh # if permission not already set
./evaluate_models.sh
```

After computing the cost matrices, we can deactivate the virtual environment:
```shell
deactivate
```

## Solving CC Problems
Run the following command:

```shell
cd correlation-clustering
chmod +x run_analysis.sh # if permission not already set
./run_analysis.sh
```
