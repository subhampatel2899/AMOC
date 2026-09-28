import os
import sys
sys.path.append(os.path.abspath('..'))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from timeit import default_timer
from model.utils import *  
from model.fno2d import FNO2d
import tqdm
device = torch.device('cuda:0')
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
# Model configurations
ntrain = 400
batch_size = 10
learning_rate = 0.001
epochs = 500
step_size = 100
gamma = 0.75
width = 32
layers = 4
sub = 2
h = int(((128 - 1) / sub) + 1)
s = int(((128 - 1) / sub) + 1)
grid_range = [1, 1]
in_channel = 4
modes = 12
ntest = 10
# Data Loader
data = np.load("data/train_dataset.npz", allow_pickle=True)
k = torch.tensor(data["k"], dtype=torch.float32)
f = torch.tensor(data["f"], dtype=torch.float32)
p = torch.tensor(data["p"], dtype=torch.float32)
data_test = np.load("data/test_dataset_in.npz", allow_pickle=True)
k_test = torch.tensor(data_test["k"], dtype=torch.float32)
f_test = torch.tensor(data_test["f"], dtype=torch.float32)
p_test = torch.tensor(data_test["p"], dtype=torch.float32)
# Subsample and prepare training and testing data
x_train_1 = k[:ntrain, ::sub, ::sub][:, :h, :s].reshape(ntrain, h, s, 1)
x_train_2 = f[:ntrain, ::sub, ::sub][:, :h, :s].reshape(ntrain, h, s, 1)
x_train = torch.cat([x_train_1, x_train_2], dim=-1)

y_train = p[:ntrain, ::sub, ::sub][:, :h, :s].reshape(ntrain, h, s, 1)

x_test_1 = k_test[-ntest:, ::sub, ::sub][:, :h, :s].reshape(ntest, h, s, 1)
x_test_2 = f_test[-ntest:, ::sub, ::sub][:, :h, :s].reshape(ntest, h, s, 1)
x_test = torch.cat([x_test_1, x_test_2], dim=-1)

y_test = p_test[-ntest:, ::sub, ::sub][:, :h, :s].reshape(ntest, h, s, 1)

x_normalizer = UnitGaussianNormalizer(x_train)
x_train = x_normalizer.encode(x_train)
x_test = x_normalizer.encode(x_test)

y_normalizer = UnitGaussianNormalizer(y_train)
y_train = y_normalizer.encode(y_train)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train),
                                          batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test),
                                         batch_size=batch_size, shuffle=False)

# Model
model = FNO2d(width=width, modes=modes, layers=layers, size=[h, s],
                      in_channel=in_channel, grid_range=grid_range).to(device)

# Optimizer and scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)



train_loss = torch.zeros(epochs)
test_loss = torch.zeros(epochs)
mse_train = torch.zeros(epochs)
time_arr = np.zeros(epochs)
myloss = LpLoss(size_average=False)
y_normalizer.to(device)

for ep in tqdm.tqdm(range(epochs)):
    model.train()
    t1 = default_timer()
    train_mse = 0.0
    train_l2 = 0.0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x)
        out = out.reshape(out.shape[0], h, s, 1)
        out = y_normalizer.decode(out)
        y = y_normalizer.decode(y)

        mse = F.mse_loss(out.view(out.shape[0], -1), y.view(y.shape[0], -1))
        loss = myloss(out.view(out.shape[0], -1), y.view(y.shape[0], -1))
        loss.backward()
        optimizer.step()

        train_mse += mse.item()
        train_l2 += loss.item()

    scheduler.step()


    model.eval()
    test_l2 = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            out = out.reshape(out.shape[0], h, s, 1)
            out = y_normalizer.decode(out)
            test_l2 += myloss(out.view(out.shape[0], -1), y.view(y.shape[0], -1)).item()


    train_mse /= len(train_loader)
    train_l2 /= ntrain
    test_l2 /= ntest

    mse_train[ep] = train_mse
    train_loss[ep] = train_l2
    test_loss[ep] = test_l2

    t2 = default_timer()
    time_arr[ep] = t2 - t1
    if (ep % 50 == 0) or (ep == epochs - 1):
        print("Epoch-{}, Time-{:0.4e}, Train-MSE-{:0.4f}, Train-L2-{:0.4f}, Test-L2-{:0.4f}"
                .format(ep, t2-t1, train_mse, train_l2, test_l2))
# save the model
# torch.save(model.state_dict(), 'model/fno_darcy.pth')
# Testing
# load the model
model.load_state_dict(torch.load('model/fno_darcy.pth'))
# Evaluation
model.eval()
test_l2 = 0.0
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        out = out.reshape(out.shape[0], h, s, 1)
        out = y_normalizer.decode(out)
        test_l2 += myloss(out.view(out.shape[0], -1), y.view(y.shape[0], -1)).item()
        
l2_error = np.linalg.norm(out.cpu().numpy() - y.cpu().numpy()) / np.linalg.norm(y.cpu().numpy())
print(f"L2 relative error: {l2_error:.8f}")
stddev = np.std([out.cpu().numpy() - y.cpu().numpy()])
print(f"Standard Deviation of L2 errors: {stddev:.8f}")
# np.savez('data/darcy_pred_fno_out.npz', predictions=out.squeeze(-1).cpu().numpy())