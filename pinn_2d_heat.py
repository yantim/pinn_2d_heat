# 2D Heat Equation PINN - PyTorch Implementation

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Neural Network for 2D Heat Equation
class PINN2D(nn.Module):
    def __init__(self):
        super(PINN2D, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64),  # x, y, t
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)   # u
        )

    def forward(self, xyt):
        return self.net(xyt)

# Initial condition: u(x,y,0) = sin(pi x) sin(pi y)
def initial_condition(x, y):
    return np.sin(np.pi * x) * np.sin(np.pi * y)

# PDE residual: u_t - alpha*(u_xx + u_yy)
def pde_residual(model, xyt, alpha):
    xyt.requires_grad = True
    u = model(xyt)
    grads = torch.autograd.grad(u, xyt, torch.ones_like(u), create_graph=True)[0]
    u_x = grads[:, 0:1]
    u_y = grads[:, 1:2]
    u_t = grads[:, 2:3]
    u_xx = torch.autograd.grad(u_x, xyt, torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
    u_yy = torch.autograd.grad(u_y, xyt, torch.ones_like(u_y), create_graph=True)[0][:, 1:2]
    return u_t - alpha * (u_xx + u_yy)

# Parameters
alpha = 0.01
epochs = 5000
lr = 0.001

# Sample training points
N_ic = 1000
N_bc = 1000
N_f = 10000

# Domain boundaries
x_low, x_high = 0.0, 1.0
y_low, y_high = 0.0, 1.0
t_low, t_high = 0.0, 1.0

# Initial condition data
x_ic = np.random.rand(N_ic, 1)
y_ic = np.random.rand(N_ic, 1)
t_ic = np.zeros((N_ic, 1))
u_ic = initial_condition(x_ic, y_ic)

# Boundary condition data
x_b = np.random.rand(N_bc, 1)
y_b = np.random.rand(N_bc, 1)
t_b = np.random.rand(N_bc, 1)
sides = np.random.choice(4, N_bc)
for i in range(N_bc):
    if sides[i] == 0: x_b[i] = 0.0
    elif sides[i] == 1: x_b[i] = 1.0
    elif sides[i] == 2: y_b[i] = 0.0
    else: y_b[i] = 1.0
u_b = np.zeros((N_bc, 1))

# Collocation points
x_f = np.random.rand(N_f, 1)
y_f = np.random.rand(N_f, 1)
t_f = np.random.rand(N_f, 1)

# Convert to tensors
xyt_ic = torch.tensor(np.hstack([x_ic, y_ic, t_ic]), dtype=torch.float32, requires_grad=True).to(device)
u_ic = torch.tensor(u_ic, dtype=torch.float32).to(device)

xyt_bc = torch.tensor(np.hstack([x_b, y_b, t_b]), dtype=torch.float32, requires_grad=True).to(device)
u_bc = torch.tensor(u_b, dtype=torch.float32).to(device)

xyt_f = torch.tensor(np.hstack([x_f, y_f, t_f]), dtype=torch.float32, requires_grad=True).to(device)

# Model and optimizer
model = PINN2D().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Training
for epoch in range(epochs):
    optimizer.zero_grad()
    loss_ic = torch.mean((model(xyt_ic) - u_ic)**2)
    loss_bc = torch.mean((model(xyt_bc) - u_bc)**2)
    loss_pde = torch.mean(pde_residual(model, xyt_f, alpha)**2)
    loss = loss_ic + loss_bc + loss_pde
    loss.backward()
    optimizer.step()
    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

# Plotting prediction at t=0.5
res = 100
x = np.linspace(0, 1, res)
y = np.linspace(0, 1, res)
X, Y = np.meshgrid(x, y)
T = 0.5 * np.ones_like(X)
xyt_plot = np.stack([X.ravel(), Y.ravel(), T.ravel()], axis=1)
xyt_tensor = torch.tensor(xyt_plot, dtype=torch.float32).to(device)
u_pred = model(xyt_tensor).cpu().detach().numpy().reshape(res, res)

plt.figure(figsize=(6, 5))
plt.contourf(X, Y, u_pred, levels=50, cmap='inferno')
plt.colorbar(label='Temperature')
plt.title('PINN Prediction at t=0.5')
plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout()
plt.show()
