# Graph Neural Network Recommendation with Knowledge Distillation

This project implements a Graph Neural Network based recommendation system with Knowledge Distillation (KD).

The framework trains a Teacher GNN model first and then distills knowledge to a smaller Student model.

Supported models:

- LightGCN

- NGCF

The student model learns from the teacher using multiple distillation objectives, improving performance while keeping the model lightweight.
