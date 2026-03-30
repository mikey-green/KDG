
---

# **Graph Neural Network Recommendation with Knowledge Distillation**

This project implements a **Graph Neural Network based recommendation system** with **Knowledge Distillation (KD)**.

The framework trains a **Teacher GNN model** first and then distills knowledge to a **smaller Student model**.

Supported models:

* LightGCN
* NGCF

The student model learns from the teacher using **multiple distillation objectives**, improving performance while keeping the model lightweight.

---
## Environment

The experiments are conducted under the following Python environment.
To reproduce the environment, please install the dependencies using the provided `requirements.txt`.

### Installation

```bash
# create virtual environment (optional but recommended)
conda create -n NAME python=3.8
conda activate NAME

# install dependencies
pip install -r requirements.txt
```

### Main Dependencies

The key libraries used in our experiments include:

* PyTorch
* NumPy
* SciPy
* scikit-learn
* tqdm
* PrettyTable

All exact package versions are listed in `requirements.txt`.

---

## Project Structure

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

## Dataset

The framework supports common recommendation datasets:

* MovieLens
* Yelp2018
* Amazon
* Alibaba

Dataset structure:

```
Dataset/
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

## Teacher Model Training

The Teacher model can be either **LightGCN** or **NGCF**.

Example:

```
python main_teacher.py --dataset movielens --gnn ngcf --dim 64 --lr 0.001 --batch_size 2048 --gpu_id 0 --context_hops 3 --pool mean --ns mixgcf --K 1 --n_negs 32
```

Important parameters:

```
--dataset        dataset name
--gnn            lightgcn or ngcf
--dim            embedding dimension
--context_hops   number of GNN layers
--batch_size
--lr
```

The best model will be saved automatically.

Saved model path:

```
Checkpoints/
teacher_ngcf_movielens_dim64_hop3.pth
```

Training logs:

```
Log/
teacher_ngcf_movielens_dim64.txt
```

---

## Student Model Training (Knowledge Distillation)

After training the Teacher model, the Student model is trained using **knowledge distillation**.

Run:

```
python main.py --dataset movielens --gnn ngcf --dim 32 --lr 0.001 --batch_size 2048 --gpu_id 0 --context_hops 2 --pool mean --ns mixgcf --K 1 --n_negs 32
```

The teacher model will be automatically loaded and frozen.

Example checkpoint:

```
Checkpoints/
student_ngcf_movielens_dim32_hop2.pth
```

---

## Evaluation Metrics

Model performance is evaluated using:

```
Recall@K
NDCG@K
Precision@K
Hit Ratio@K
```
---

## Citation

If you use this code in your research, please cite:

```
@article{gnn_kd_rec_2026,
  title={Graph Neural Network based Recommendation with Knowledge Distillation},
  author={Liang Qiaoxin},
  year={2026}
}
```
---

> The proposed framework combines **Graph Neural Networks** and **Knowledge Distillation** to transfer structural and semantic knowledge from a large teacher model to a lightweight student model.

graph TD
    subgraph Input
        A1[用户ID] --> B1[Embedding Layer]
        A2[物品ID] --> B2[Embedding Layer]
        B1 --> C0_u[用户初始嵌入 e_u^0]
        B2 --> C0_i[物品初始嵌入 e_i^0]
    end

    subgraph "Layer 1 - Graph Convolution"
        C0_u --> D1_u[自身变换: W1 * e_u^0]
        C0_i --> D1_i[邻居聚合: 加权求和邻居物品嵌入]
        C0_i --> D2_i[自身变换: W1 * e_i^0]
        C0_u --> D2_u[邻居聚合: 加权求和邻居用户嵌入]
        
        D1_u --> E1[LeakyReLU]
        D1_i --> E1
        D2_i --> E2[LeakyReLU]
        D2_u --> E2
        
        E1 --> F1_u[用户层1嵌入 e_u^1]
        E2 --> F1_i[物品层1嵌入 e_i^1]
    end

    subgraph "Layer 2 - Graph Convolution"
        F1_u --> G1_u[自身变换: W1 * e_u^1]
        F1_i --> G1_i[邻居聚合: 加权求和邻居物品嵌入]
        F1_i --> G2_i[自身变换: W1 * e_i^1]
        F1_u --> G2_u[邻居聚合: 加权求和邻居用户嵌入]
        
        G1_u --> H1[LeakyReLU]
        G1_i --> H1
        G2_i --> H2[LeakyReLU]
        G2_u --> H2
        
        H1 --> I1_u[用户层2嵌入 e_u^2]
        H2 --> I1_i[物品层2嵌入 e_i^2]
    end

    subgraph "Layer L - 可继续堆叠"
        style LayerL fill:#f9f,stroke:#333
    end

    subgraph "Layer Combination"
        C0_u --> J1[拼接 Concat]
        F1_u --> J1
        I1_u --> J1
        C0_i --> J2[拼接 Concat]
        F1_i --> J2
        I1_i --> J2
        
        J1 --> K_u[用户最终嵌入 e_u]
        J2 --> K_i[物品最终嵌入 e_i]
    end

    subgraph "Prediction"
        K_u --> L[点积 Dot Product]
        K_i --> L
        L --> M[预测评分 ŷ_ui]
    end

    style Input fill:#e1f5fe
    style Layer1 fill:#fff3e0
    style Layer2 fill:#fff3e0
    style LayerCombination fill:#e8f5e9
    style Prediction fill:#fce4ec
