# This file trains a simple MLP to learn the XOR function using micrograd.
from nn import MLP
from optim import SGD
from loss import MSE
import matplotlib.pyplot as plt

# XOR dataset
xs = [
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
]
ys = [0.0, 1.0, 1.0, 0.0]  # Ground truth values

model = MLP(2, [16, 16, 1])  # 2 inputs, two hidden layers with 16 neurons each, 1 output
optimizer = SGD(model.parameters(), lr=0.01)

numsteps = 4000
lossHistory = []
for step in range(numsteps):
    ypred = [model(x) for x in xs]  # Get model predictions
    loss = MSE(ypred, ys)           # Compute loss
    optimizer.zero_grad()           # Reset gradients
    loss.backward()                 # Backpropagate to compute gradients
    optimizer.step()                # Update model parameters
    lossHistory.append(loss.data)
    if step % 100 == 0:
        print(f"Step {step}, Loss = {loss.data:.6f}")

print("\nTrained model predictions:")
for x, y in zip(xs, ys):
    y_pred = model(x).data
    print(f"Input: {x}, Predicted: {y_pred:.4f}, True: {y}")

plt.plot(lossHistory)
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("XOR Training Loss")
plt.show()
