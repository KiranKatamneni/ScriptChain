# ScriptChain

# Q1. Suppose that we design a deep architecture to represent a sequence by stacking self-attention layers with positional encoding. What could be issues? <br />
Designing a deep architecture for sequence representation by stacking multiple self-attention layers with positional encoding introduces several challenges. <br />
One major challenge is the vanishing gradient problem, which slows convergence and leads to poor weight updates in deep models due to inefficient gradient propagation. Overfitting is another concern, especially when stacking too many self-attention layers on limited data, as the model may memorize patterns instead of generalizing effectively to new sequences.

Computational complexity also becomes a bottleneck since memory and processing requirements grow exponentially with the number of layers, while self-attention scales quadratically with sequence length. Even though positional encoding helps retain order information, it may not be sufficient in very deep networks. Higher layers tend to dilute or underutilize positional representations, making it harder for the model to capture long-range dependencies.

Degradation of positional encoding is a major issue when stacking multiple self-attention layers. As the model deepens, positional encodings get mixed with learned word representations at each layer. This repetitive transformation weakens the model’s sensitivity to token positions, making it rely more on semantic content rather than sequence order. The use of layer normalization in Transformers further contributes to this issue, as it rescales input embeddings, reducing the distinctiveness of positional information across layers.

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

* d_model: 128
* num_heads: 8
* num_layers: 3
* max_seq_len: 15
* num_classes: 4
* learning_rate: 1e-4
* step_size: 5
* batch_size: 32
* epochs: 30



