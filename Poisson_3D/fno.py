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
from model.fno3d import FNO3d
import tqdm
device = torch.device('cuda:1')
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
ntrain = 400
ntest = 10

modes = 20
width = 32

batch_size = 10
batch_size2 = batch_size
inchannel = 4
epochs = 500
learning_rate = 0.001
scheduler_step = 100
scheduler_gamma = 0.75
sub = 2
S = 128 // sub
T_in = 10
T =  128 // sub
# load data
data = np.load('data/train_dataset.npz', allow_pickle=True)
f_load = torch.tensor(data['F'], dtype=torch.float)
U_load = torch.tensor(data['U'], dtype=torch.float)
data_test = np.load('data/test_dataset_out.npz', allow_pickle=True)
f_load_test = torch.tensor(data_test['F'], dtype=torch.float)
U_load_test = torch.tensor(data_test['U'], dtype=torch.float)
f_train = f_load[:ntrain, ::sub, ::sub, ::sub].unsqueeze(-1)
U_train = U_load[:ntrain, ::sub, ::sub, ::sub]
f_test = f_load_test[:ntest, ::sub, ::sub, ::sub].unsqueeze(-1)
U_test = U_load_test[:ntest, ::sub, ::sub, ::sub]
# normalizer
f_normalizer = UnitGaussianNormalizer(f_train)
f_train = f_normalizer.encode(f_train)
f_test = f_normalizer.encode(f_test)
U_normalizer = UnitGaussianNormalizer(U_train)
U_train = U_normalizer.encode(U_train)
gridx = torch.tensor(np.linspace(0, 1, S), dtype=torch.float)
gridx = gridx.reshape(1, S, 1, 1, 1).repeat([1, 1, S, T, 1])
gridy = torch.tensor(np.linspace(0, 1, S), dtype=torch.float)
gridy = gridy.reshape(1, 1, S, 1, 1).repeat([1, S, 1, T, 1])
gridt = torch.tensor(np.linspace(0, 1, T+1)[1:], dtype=torch.float)
gridt = gridt.reshape(1, 1, 1, T, 1).repeat([1, S, S, 1, 1])
train_a = torch.cat((gridx.repeat([ntrain,1,1,1,1]), gridy.repeat([ntrain,1,1,1,1]),
                       gridt.repeat([ntrain,1,1,1,1]), f_train), dim=-1)
test_a = torch.cat((gridx.repeat([ntest,1,1,1,1]), gridy.repeat([ntest,1,1,1,1]),
                       gridt.repeat([ntest,1,1,1,1]), f_test), dim=-1)
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, U_train), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, U_test), batch_size=batch_size, shuffle=False)
model = FNO3d(inchannel,modes, modes, modes, width).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
U_normalizer.to(device)
myloss = LpLoss(size_average=False)

for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_mse = 0
    train_l2 = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x).view(batch_size, S, S, T)
        out = U_normalizer.decode(out)
        y = U_normalizer.decode(y)
        mse = F.mse_loss(out, y, reduction='mean')

        l2 = myloss(out.view(batch_size, -1), y.view(batch_size, -1))
        l2.backward()

        optimizer.step()
        train_mse += mse.item()
        train_l2 += l2.item()

    scheduler.step()

    model.eval()
    test_l2 = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x).view(batch_size, S, S, T)
            out = U_normalizer.decode(out)
            test_l2 += myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()

    train_mse /= len(train_loader)
    train_l2 /= ntrain
    test_l2 /= ntest

    t2 = default_timer()
    print(ep, t2-t1, train_mse, train_l2, test_l2)
#save the model
# torch.save(model.state_dict(), 'model/fno_poisson_3d.pth')
# Testing
# load model
model.load_state_dict(torch.load('model/fno_poisson_3d.pth'))
predictions = []
exact = []

for ntest in range(0, 10):
    sample = 1
    batch_size = 10
    T=64
    model.eval()
    test_l2 = 0.0
    myloss = LpLoss(size_average=False)
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)

            out = model(x).view(batch_size, S, S, T)
            out = U_normalizer.decode(out)
            
            test_l2 += myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()
    
    l2_error = torch.norm(out[ntest] - y[ntest], p=2) / torch.norm(y[ntest], p=2)
    print(f"L2 relative error: {l2_error.item():.4f}")


predictions = out.cpu().numpy()
exact = y.cpu().numpy()
# np.savez("data/poisson_pred_fno_out.npz", pred=predictions)
