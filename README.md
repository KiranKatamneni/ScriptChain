# ScriptChain

# Q2. Can you design a learnable positional encoding method using pytorch? (Create dummy dataset)

# Overview
This notebook implements a Transformer-based sentiment analysis model using learnable positional encoding in PyTorch. The model is trained on a multi-class sentiment dataset, where it classifies sentences into four sentiment categories: Joyful, Frustrated, Excited, and Disappointed.

# Model Architecture
Token Embeddings: Converts input words into dense vectors. 

Positional Encoding: Learns positional embeddings dynamically.
Transformer Encoder Layers: Applies self-attention and feedforward layers.
Fully Connected Classifier: Converts Transformer outputs into sentiment labels.
Cross-Entropy Loss Function: Optimized using the Adam optimizer.

# Dataset
The dataset consists of 100 sentences labeled with four sentiment categories:
Joyful 😃
Frustrated 😡
Excited 🚀
Disappointed 😞

# Training Details

Optimizer: Adam (lr=1e-4 with StepLR scheduler)
Batch Size: 32
Epochs: 30
Dropout: 0.2
Validation Split: 20%


