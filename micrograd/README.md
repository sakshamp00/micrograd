
# Micrograd From Scratch (Neural Network + Autograd Engine)

This project is a **from-scratch reimplementation of a tiny deep learning framework** inspired by PyTorch and Karpathy’s micrograd.

It includes:
- A scalar-based **automatic differentiation engine**
- A **neural network library** (Neuron, Layer, MLP)
- An **SGD optimizer**
- An **MSE loss**
- A **full training loop** that learns XOR

---

# 📒 Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Training Example (XOR)](#training-example-xor)
- [Contributing](#contributing)

---

# 📦 Installation

1. Clone the repo:

   ```bash
   git clone https://github.com/sakshamp00/micrograd.git
   cd micrograd
   ```

2. Install dependencies (optional, mainly for plotting):

    ```bash
    pip install matplotlib
    ```

---

# 🛠️ Usage

Just run the training script:

    python train_xor.py
    

You’ll see:
- Loss decreasing over training steps
- Final predictions on the XOR dataset
- Optional loss plot (if matplotlib is installed)

---


# 📁 Project Structure

```micrograd/
├── engine.py       # Core autograd Value class
├── nn.py           # Neuron, Layer, MLP
├── optim.py        # SGD optimizer
├── loss.py         # Loss functions (MSE)
├── train_xor.py    # Training script
└── README.md       # This file
```

---

# 📊 Training Example (XOR)

XOR dataset:

```
0 ⊕ 0 → 0  
0 ⊕ 1 → 1  
1 ⊕ 0 → 1  
1 ⊕ 1 → 0
```
The training script:
- Builds the MLP
- Loops forward → backprop → update
- Prints loss and final accuracy

Example output after training:

```
step 0, loss = 2.31  
step 100, loss = 0.21 
...
step 900, loss = 0.02  

Trained model predictions:
Input: [0.0, 0.0], Predicted: 0.0111, True: 0.0
Input: [0.0, 1.0], Predicted: 0.9785, True: 1.0
Input: [1.0, 0.0], Predicted: 0.9831, True: 1.0
Input: [1.0, 1.0], Predicted: 0.0142, True: 0.0
```
---
# 🤝 Contributing

Contributions are welcome! Whether it’s improving the documentation, adding features like:

- Activation functions (ReLU, Sigmoid)
- Optimizers (Adam)
- Batch support
- More demos (MNIST, regression)

Feel free to open issues or pull requests 🎉