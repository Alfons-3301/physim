import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
import time

# Polar coding functions
def polar_transform_matrix(n):
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    for i in range(1, n):
        G = np.kron(G, F) % 2
    return G

def polar_encode(u, n):
    G = polar_transform_matrix(n)
    return np.mod(np.dot(u, G), 2)

def polar_bhattacharyya(n, Z0):
    Z = [Z0]
    for _ in range(n):
        Z_new = []
        for z in Z:
            Z_new.append(min(2*z - z*z, 1.0))
            Z_new.append(z*z)
        Z = Z_new
    return np.array(Z)

def select_good_channels(Z, target):
    sorted_idx = np.argsort(Z)
    cum = np.cumsum(Z[sorted_idx])
    K = np.searchsorted(cum, target)
    return sorted_idx[:K], K

# Toeplitz hash functions
def generate_toeplitz_hash_matrix(m, n, seed=None):
    if seed is not None:
        np.random.seed(seed)
    c = np.random.randint(0, 2, size=(m,))
    r = np.random.randint(0, 2, size=(n,))
    r[0] = c[0]
    from scipy.linalg import toeplitz
    T = toeplitz(c, r) % 2
    return T.astype(int)

def toeplitz_hash(x, T):
    return np.mod(T.dot(x), 2)

# MINE module
class Mine(nn.Module):
    def __init__(self, input_size=2, hidden_size=128):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.normal_(self.fc2.weight, std=0.02)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.normal_(self.fc3.weight, std=0.02)
        nn.init.constant_(self.fc3.bias, 0)
    def forward(self, input):
        out = F.elu(self.fc1(input))
        out = F.elu(self.fc2(out))
        out = self.fc3(out)
        return out

def mutual_information(joint, marginal, mine_net):
    t = mine_net(joint)
    et = torch.exp(mine_net(marginal))
    mi_lb = torch.mean(t) - torch.log(torch.mean(et))
    return mi_lb, t, et

def learn_mine(batch, mine_net, mine_net_optim, ma_et, ma_rate=0.01):
    joint, marginal = batch
    joint = torch.autograd.Variable(torch.FloatTensor(joint)).cuda()
    marginal = torch.autograd.Variable(torch.FloatTensor(marginal)).cuda()
    mi_lb, t, et = mutual_information(joint, marginal, mine_net)
    ma_et = (1 - ma_rate) * ma_et + ma_rate * torch.mean(et)
    loss = -(torch.mean(t) - (1/ma_et.mean()).detach() * torch.mean(et))
    mine_net_optim.zero_grad()
    autograd.backward(loss)
    mine_net_optim.step()
    return mi_lb, ma_et, loss.item()

def sample_batch(data, batch_size, sample_mode='joint'):
    dim = data.shape[1]//2
    if sample_mode == 'joint':
        idx = np.random.choice(range(data.shape[0]), size=batch_size, replace=False)
        batch = data[idx]
    else:
        jidx = np.random.choice(range(data.shape[0]), size=batch_size, replace=False)
        midx = np.random.choice(range(data.shape[0]), size=batch_size, replace=False)
        batch = np.concatenate([data[jidx][:, :dim], data[midx][:, dim:2*dim]], axis=1)
    return batch

def train(data, mine_net, mine_net_optim, batch_size=100, iter_num=100):
    result = []
    ma_et = 1.
    for i in range(iter_num):
        batch = (sample_batch(data, batch_size), sample_batch(data, batch_size, sample_mode='marginal'))
        mi_lb, ma_et, loss = learn_mine(batch, mine_net, mine_net_optim, ma_et)
        result.append(mi_lb.detach().cpu().numpy())
    return result

def ma(a, window_size=50):
    return [np.mean(a[i:i+window_size]) for i in range(0, len(a)-window_size)]

def MINE_esti(data, batch_size=100, iter_num=100):
    input_size = data.shape[1]
    mine_net = Mine(input_size=input_size, hidden_size=2).cuda()
    mine_net_optim = optim.Adam(mine_net.parameters(), lr=1e-2)
    result = train(data, mine_net, mine_net_optim, batch_size=batch_size, iter_num=iter_num)
    result_ma = ma(result, window_size=50)
    return result_ma[-1]

# Main script
def H2(p):
    return 0 if (p==0 or p==1) else (-p*np.log2(p) - (1-p)*np.log2(1-p))

def main():
    p_main = 0.1
    p_eve = 0.2
    Z0_main = 2 * np.sqrt(p_main*(1-p_main))
    n = 12
    N = 2**n
    target_error = 1e-6
    Z_main = polar_bhattacharyya(n, Z0_main)
    good_idx, _ = select_good_channels(Z_main, target_error)
    print("Good channels:", len(good_idx))
    C_main = 1 - H2(p_main)
    C_eve = 1 - H2(p_eve)
    sec_cap = max(C_main - C_eve, 0)
    K_sec_unconstrained = int(np.floor(sec_cap * N))
    K_sec = min(K_sec_unconstrained, len(good_idx))
    print("Secure bits:", K_sec)
    secure_idx = good_idx[:K_sec]
    T = generate_toeplitz_hash_matrix(K_sec, K_sec, seed=42)
    msg = np.random.randint(0, 2, size=(K_sec,))
    secret_key = toeplitz_hash(msg, T)
    u = np.zeros(N, dtype=int)
    u[secure_idx] = secret_key
    x = polar_encode(u, n)
    # Simulate channels
    noise_b = (np.random.rand(N) < p_main).astype(int)
    y_b = np.mod(x + noise_b, 2)
    noise_e = (np.random.rand(N) < p_eve).astype(int)
    z_e = np.mod(x + noise_e, 2)
    # Assume Bob decodes ideally
    u_hat = y_b#u.copy()
    rec_key = u_hat[secure_idx]
    print("Recovered key:", rec_key)
    # Leakage estimation at Eve: collect M samples of (S, Z) pairs.
    M = 1000
    samples = []
    for _ in range(M):
        msg = np.random.randint(0, 2, size=(K_sec,))
        secret_key = toeplitz_hash(msg, T)
        u = np.zeros(N, dtype=int)
        u[secure_idx] = secret_key
        x = polar_encode(u, n)
        noise_e = (np.random.rand(N) < p_eve).astype(int)
        z_e = np.mod(x + noise_e, 2)
        S = secret_key.astype(np.float32)
        Z = z_e[secure_idx].astype(np.float32)
        sample = np.concatenate([S, Z])
        samples.append(sample)
    data = np.stack(samples, axis=0)
    leakage_est = MINE_esti(data, batch_size=100, iter_num=1000)
    print("Estimated leakage (MI) at Eve:", leakage_est)

if __name__ == '__main__':
    main()
