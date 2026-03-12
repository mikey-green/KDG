# Graph Neural Network Recommendation with Knowledge Distillation

This project implements a **Graph Neural Network based recommendation system** with **Knowledge Distillation (KD)**.

The framework trains a **Teacher GNN model** first and then distills knowledge to a **smaller Student model**.

Supported models:

- LightGCN

- NGCF

The student model learns from the teacher using **multiple distillation objectives**, improving performance while keeping the model lightweight.

## Environment Requirements

The experiments are conducted with the following environment.

```
Python 3.8
CUDA 11.8
PyTorch 2.0.0
NumPy
scikit-learn
tqdm
prettytable
```

Operating System

```
Ubuntu 20.04
```

Install dependencies:

```
pip install torch==2.0.0 numpy scikit-learn tqdm prettytable
```

