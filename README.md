# ScriptChain

# Q2. Can you design a learnable positional encoding method using pytorch? (Create dummy dataset)

# Overview
This notebook implements a Transformer-based sentiment analysis model using learnable positional encoding in PyTorch. The model is trained on a multi-class sentiment dataset, where it classifies sentences into four sentiment categories: Joyful, Frustrated, Excited, and Disappointed.

# Model Architecture
Token Embeddings: Converts input words into dense vectors. <br />
Positional Encoding: Learns positional embeddings dynamically.<br />
Transformer Encoder Layers: Applies self-attention and feedforward layers.<br />
Fully Connected Classifier: Converts Transformer outputs into sentiment labels.<br />
Cross-Entropy Loss Function: Optimized using the Adam optimizer.

# Dataset
The dataset consists of 100 sentences labeled with four sentiment categories: <br />
Joyful 😃 <br />
Frustrated 😡 <br />
Excited 🚀 <br />
Disappointed 😞 <br />

# Training Details

-d_model: 128
-num_heads: 8
-num_layers: 3
-max_seq_len: 15
-num_classes: 4
-learning_rate: 1e-4
-step_size: 5
-batch_size: 32
-epochs: 30



