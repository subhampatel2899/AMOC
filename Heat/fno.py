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
torch.cuda.manual_seed(seed)

# Model configurations
ntrain = 400
batch_size = 20
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
in_channel = 3
modes = 20
ntest = 10
# Data Loader
train_data = np.load("data/heat_data_1.npz", allow_pickle=True)
test_data = np.load("data/heat_data_in.npz", allow_pickle=True)
eps_train = torch.tensor(train_data["eps"], dtype=torch.float32)
u_train = torch.tensor(train_data["u"], dtype=torch.float32)
eps_test = torch.tensor(test_data["eps"], dtype=torch.float32)
u_test = torch.tensor(test_data["u"], dtype=torch.float32)

x_train = eps_train[:ntrain, ::sub, ::sub][:, :h, :s]
y_train = u_train[:ntrain, ::sub, ::sub][:, :h, :s]

x_test = eps_test[:ntest, ::sub, ::sub][:, :h, :s]
y_test = u_test[:ntest, ::sub, ::sub][:, :h, :s]

x_train = x_train.float()
y_train = y_train.float()
x_test = x_test.float()
y_test = y_test.float()
# normalize epsilon
eps_max = x_train.max()
x_train = x_train / eps_max
x_test = x_test / eps_max

# log normalize output
y_normalizer = LogNormalizer()
y_normalizer.fit(y_train)

y_train = y_normalizer.encode(y_train)
x_train = x_train.reshape(ntrain, h, s, 1)
x_test = x_test.reshape(ntest, h, s, 1)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False  )
# Model
model = FNO2d(width=width, modes=modes, layers=layers, size=[h, s],
                      in_channel=in_channel, grid_range=grid_range).to(device)

# Optimizer and scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

myloss = LpLoss(size_average=False)

train_loss = torch.zeros(epochs)
test_loss = torch.zeros(epochs)
mse_train = torch.zeros(epochs)
time_arr = np.zeros(epochs)


y_normalizer.to(device)
for ep in tqdm.tqdm(range(epochs)):
    model.train()
    t1 = default_timer()

    train_mse = 0.0
    train_l2 = 0.0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)   # normalized log space
        mse = F.mse_loss(out.view(out.shape[0], -1),y.view(y.shape[0], -1))
        loss = myloss(out.view(out.shape[0], -1),y.view(y.shape[0], -1))
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
            out = y_normalizer.decode(out.reshape(out.shape[0], h, s))
            test_l2 += myloss(out.view(out.shape[0], -1),y.view(y.shape[0], -1)).item()

    train_mse /= len(train_loader)   
    train_l2 /= ntrain              
    test_l2 /= ntest

    mse_train[ep] = train_mse
    train_loss[ep] = train_l2
    test_loss[ep] = test_l2

    t2 = default_timer()
    time_arr[ep] = t2 - t1
    if (ep) % 50 == 0:
        print(f"Epoch-{ep}, Time-{t2-t1:.4f}, "
            f"Train-MSE-{train_mse:.6f}, "
            f"Train-L2-{train_l2:.6f}, "
            f"Test-L2-{test_l2:.6f}")
        
# save the model
# torch.save(model.state_dict(), 'model/fno_heat.pth')
# load the model
model.load_state_dict(torch.load('model/fno_heat.pth'))
# Testing
predictions = []
for i in range(ntest):
    x_train = eps_train[:ntrain, ::sub, ::sub][:, :h, :s]
    y_train = u_train[:ntrain, ::sub, ::sub][:, :h, :s]

    x_test1 = eps_test[i:i+1, ::sub, ::sub][:, :h, :s]
    y_test = u_test[i:i+1, ::sub, ::sub][:, :h, :s]

    # ensure float (NO torch.tensor again!)
    x_train = x_train.float()
    y_train = y_train.float()
    x_test = x_test1.float()
    y_test = y_test.float()
    # normalize epsilon (simple scaling works best)
    eps_max = x_train.max()
    x_train = x_train / eps_max
    x_test = x_test / eps_max

    # log normalize output
    y_normalizer = LogNormalizer()
    y_normalizer.fit(y_train)

    y_train = y_normalizer.encode(y_train)
    x_train = x_train.reshape(ntrain, h, s, 1)
    x_test = x_test.reshape(1, h, s, 1)

    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)
    model.eval()
    y_normalizer.to(device)

    x = x_test.to(device)
    y = y_test.to(device)          
    test_l2 = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            out = y_normalizer.decode(out.reshape(out.shape[0], h, s))
            predictions.append(out.cpu().numpy())
            test_l2 += myloss(out.view(out.shape[0], -1),y.view(y.shape[0], -1)).item()
    l2_error = torch.norm(out - y, p=2) / torch.norm(y, p=2)
    print(f"xtest: {x_test1[0, 0, 0].item():.4f} L2 relative error: {l2_error.item():.4f}, test L2 loss: {test_l2:.6f}")
    
predictions = np.concatenate(predictions, axis=0)
np.savez("data/heat_pred_fno_out.npz", pred=predictions)
