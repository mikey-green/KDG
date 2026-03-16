
---

# **Graph Neural Network Recommendation with Knowledge Distillation**

This project implements a **Graph Neural Network based recommendation system** with **Knowledge Distillation (KD)**.

The framework trains a **Teacher GNN model** first and then distills knowledge to a **smaller Student model**.

Supported models:

* LightGCN
* NGCF

The student model learns from the teacher using **multiple distillation objectives**, improving performance while keeping the model lightweight.

---

## 1. Environment Requirements

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

---

## 2. Project Structure

```
project/
│
├── Dataset/                     # dataset directory
│
├── Modules/
│   ├── LightGCN.py
│   ├── NGCF.py
│   └── Student.py            # Student LightGCN model
│
├── Utils/
│   ├── parser.py
│   ├── data_loader.py
│   ├── evaluate.py
|   ├── metrics.py
│   └── helper.py
│
├── main_teacher.py                # Teacher model training
├── main.py                   # Student KD training
│
├── Log/                      # training logs
├── Checkpoints/              # saved model weights
│
└── README.md
```

---

## 3. Dataset

The framework supports common recommendation datasets:

* MovieLens
* Yelp2018
* Amazon
* Alibaba

Dataset structure:

```
data/
│
├── movielens/
│   ├── train.txt
│   ├── test.txt
|   └── valid.txt
│
├── yelp2018/
│
├── amazon/
│
├── ali/
```

Each dataset contains user-item interaction files.

---

## 4. Teacher Model Training

The Teacher model can be either **LightGCN** or **NGCF**.

Example:

```
python teacher.py --dataset movielens --gnn lightgcn --dim 64
```

Important parameters:

```
--dataset        dataset name
--gnn            lightgcn or ngcf
--dim            embedding dimension
--context_hops   number of GNN layers
--batch_size
--lr
--epoch
```

Teacher training uses:

* BPR loss
* negative sampling
* early stopping

The best model will be saved automatically.

Saved model path:

```
Checkpoints/
teacher_lightgcn_movielens_dim64_hop3.pth
```

Training logs:

```
Log/
teacher_lightgcn_movielens_dim64.txt
```

---

## 5. Student Model Training (Knowledge Distillation)

After training the Teacher model, the Student model is trained using **knowledge distillation**.

Run:

```
python main.py --dataset movielens --gnn lightgcn --dim 32
```

Student model:

```
StudentLightGCN
```

The teacher model will be automatically loaded and frozen.

Example checkpoint:

```
Checkpoints/
teacher_lightgcn_movielens_dim64_hop3.pth
```

---

## 6. Knowledge Distillation Strategy

The student model learns from the teacher using **three types of distillation losses**.

### 1. Score Distillation

Match teacher and student prediction scores.

```
score_kd = Wasserstein distance
```

Computed from:

```
positive score
negative score
```

---

### 2. Representation Distillation

Match intermediate GNN embeddings.

Student embeddings are projected to teacher dimension using a **projection layer**.

```
Linear(32 → 64)
```

Loss:

```
MSE(student_layer, teacher_layer)
```

---

### 3. Structure Distillation

Preserve similarity relationships between users and items.

Similarity is computed using:

```
cosine similarity
```

Loss:

```
MSE(sim_student, sim_teacher)
```

---

## 7. Total Training Loss

The final training objective is:

```
Loss =
time_weight * BPR_loss
+
time_weight * kd_weight * (
    0.5 * score_kd
    + 0.4 * representation_kd
    + 0.1 * structure_kd
)
```

Additional:

```
L2 regularization
```

KD weight is dynamically adjusted during training.

---

## 8. Evaluation Metrics

Model performance is evaluated using:

```
Recall@K
NDCG@K
Precision@K
Hit Ratio@K
```

Evaluation is implemented in:

```
Utils/evaluate.py
```

---

## 9. Training Logs

Logs are automatically saved.

Example:

```
Log/

teacher_lightgcn_movielens_dim64.txt
student_lightgcn_movielens_dim32.txt
```

Each log contains:

```
Epoch
training time
testing time
loss
recall
ndcg
precision
hit_ratio
```

---

## 10. Model Checkpoints

Saved models are stored in:

```
Checkpoints/
```

Example:

```
teacher_lightgcn_movielens_dim64_hop3.pth
student_lightgcn_movielens_dim32_hop2_kd.pth
```

---

## 11. Reproducibility

Random seeds are fixed for reproducibility.

```
seed = 2020
```

The following libraries are seeded:

* Python random
* NumPy
* PyTorch

---

## 12. Citation

If you use this code in your research, please cite:

```
@article{gnn_kd_rec_2026,
  title={Graph Neural Network based Recommendation with Knowledge Distillation},
  author={Your Name},
  year={2026}
}
```

---

## 13. Notes

Workflow:

1️⃣ Train Teacher model

```
python teacher.py
```

2️⃣ Train Student model with KD

```
python main.py
```

3️⃣ Evaluate recommendation performance

---

> The proposed framework combines **Graph Neural Networks** and **Knowledge Distillation** to transfer structural and semantic knowledge from a large teacher model to a lightweight student model.


