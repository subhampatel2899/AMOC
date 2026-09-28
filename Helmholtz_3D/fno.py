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

epochs = 500
learning_rate = 0.001
scheduler_step = 100
gamma = 0.75
sub = 1
S = 64 // sub
inchannel = 5
# load data
data = np.load('data/helmholtz_train.npz', allow_pickle=True)
f_load = torch.tensor(data['f'], dtype=torch.float)
k_load = torch.tensor(data['k'], dtype=torch.float)
U_load = torch.tensor(data['u'], dtype=torch.float)
data_test = np.load('data/helmholtz_test_out.npz', allow_pickle=True)
f_load_test = torch.tensor(data_test['f'], dtype=torch.float)
k_load_test = torch.tensor(data_test['k'], dtype=torch.float)
U_load_test = torch.tensor(data_test['u'], dtype=torch.float)
k = k_load.view(k_load.shape[0], 1, 1).expand(k_load.shape[0], S, S)
train_k = k.reshape(ntrain,S,S,1,1).repeat([1,1,1,S,1])
train_f = f_load[:ntrain,::sub, ::sub, ::sub].reshape(ntrain,S,S,S,1)
train_a = torch.cat([train_f, train_k], dim=-1)


k_test = k_load_test.view(k_load_test.shape[0], 1, 1).expand(k_load_test.shape[0], S, S)
test_k = k_test.reshape(ntest,S,S,1,1).repeat([1,1,1,S,1])
test_f = f_load_test[:ntest,::sub, ::sub, ::sub].reshape(ntest,S,S,S,1)
test_a = torch.cat([test_f, test_k], dim=-1)


x_normalizer = UnitGaussianNormalizer(train_a)
train_a = x_normalizer.encode(train_a)
test_a = x_normalizer.encode(test_a)

train_U_b = U_load[:ntrain,::sub,::sub,::sub]
test_U_b = U_load_test[:ntest,::sub,::sub,::sub]
y_normalizer = UnitGaussianNormalizer(train_U_b)
train_U_b = y_normalizer.encode(train_U_b)

# pad locations (x,y,t)
gridx = torch.tensor(np.linspace(0, 1, S), dtype=torch.float)
gridx = gridx.reshape(1, S, 1, 1, 1).repeat([1, 1, S, S, 1])
gridy = torch.tensor(np.linspace(0, 1, S), dtype=torch.float)
gridy = gridy.reshape(1, 1, S, 1, 1).repeat([1, S, 1, S, 1])
gridt = torch.tensor(np.linspace(0, 1, S), dtype=torch.float)
gridt = gridt.reshape(1, 1, 1, S, 1).repeat([1, S, S, 1, 1])

train_a = torch.cat((gridx.repeat([ntrain,1,1,1,1]), gridy.repeat([ntrain,1,1,1,1]),
                       gridt.repeat([ntrain,1,1,1,1]), train_a), dim=-1)
test_a = torch.cat((gridx.repeat([ntest,1,1,1,1]), gridy.repeat([ntest,1,1,1,1]),
                       gridt.repeat([ntest,1,1,1,1]), test_a), dim=-1)
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_U_b), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_U_b), batch_size=batch_size, shuffle=False)
model = FNO3d(inchannel,modes, modes, modes, width).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=gamma)
y_normalizer.to(device)
myloss = LpLoss(size_average=False)
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_mse = 0
    train_l2 = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x).view(batch_size, S, S, S)
        out = y_normalizer.decode(out)
        y = y_normalizer.decode(y)
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
            out = model(x).view(batch_size, S, S, S)
            out = y_normalizer.decode(out)
            test_l2 += myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()

    train_mse /= len(train_loader)
    train_l2 /= ntrain
    test_l2 /= ntest

    t2 = default_timer()
    print(ep, t2-t1, train_mse, train_l2, test_l2)
#save the model
# torch.save(model.state_dict(), 'model/fno_helm_64.pth')
# Testing
# load model
model.load_state_dict(torch.load('model/fno_helm_64.pth'))

predictions = []
exact = []

# for ntest in range(0, 10):
model.eval()
test_l2 = 0.0
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        out = model(x).view(batch_size, S, S, S)
        out = y_normalizer.decode(out)
        test_l2 += myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()


l2_error = torch.norm(out - y, p=2) / torch.norm(y, p=2)
print(f"L2 relative error: {l2_error.item():.6f}")

# np.savez("data/helm_pred_fno_in.npz", pred=out.cpu().numpy(), k=k_load_test.cpu().numpy(), exact=y.cpu().numpy())
