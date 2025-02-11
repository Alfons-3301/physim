import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import math

# -----------------------------------------------------------------------------
# Dynamic MINE network definition (as provided previously)
# -----------------------------------------------------------------------------
class DynamicMine(nn.Module):
    def __init__(self,
                 input_size=2,
                 hidden_sizes=None,
                 output_size=1,
                 activation=F.elu,
                 use_resnet=False,
                 resnet_every=2,
                 init_std=0.02,
                 init_bias=0.0):
        """
        Parameters:
            input_size (int): Dimension of the input.
            hidden_sizes (list of int): List specifying neurons per hidden layer.
                                        Defaults to [128, 128] (original MINE).
            output_size (int): Dimension of the output.
            activation (callable): Activation function (default: F.elu).
            use_resnet (bool): Whether to add a ResNet-like connection every 'resnet_every' layers.
            resnet_every (int): Frequency (in layers) to add a residual connection.
            init_std (float): Standard deviation for weight initialization.
            init_bias (float): Bias initialization constant.
        """
        super(DynamicMine, self).__init__()
        if hidden_sizes is None:
            hidden_sizes = [128, 128]
        self.activation = activation
        self.use_resnet = use_resnet
        self.resnet_every = resnet_every if use_resnet else None

        self.layers = nn.ModuleList()
        self.residual_mappings = nn.ModuleList() if use_resnet else None

        in_dim = input_size
        block_start_dim = in_dim

        for i, h_dim in enumerate(hidden_sizes):
            layer = nn.Linear(in_dim, h_dim)
            nn.init.normal_(layer.weight, std=init_std)
            nn.init.constant_(layer.bias, init_bias)
            self.layers.append(layer)
            in_dim = h_dim

            if self.use_resnet and ((i + 1) % self.resnet_every == 0):
                if block_start_dim != h_dim:
                    mapping = nn.Linear(block_start_dim, h_dim)
                    nn.init.normal_(mapping.weight, std=init_std)
                    nn.init.constant_(mapping.bias, init_bias)
                else:
                    mapping = nn.Identity()
                self.residual_mappings.append(mapping)
                block_start_dim = h_dim

        self.fc_out = nn.Linear(in_dim, output_size)
        nn.init.normal_(self.fc_out.weight, std=init_std)
        nn.init.constant_(self.fc_out.bias, init_bias)

    def forward(self, x):
        if self.use_resnet:
            out = x
            res_block_index = 0
            block_input = out
            for i, layer in enumerate(self.layers):
                out = self.activation(layer(out))
                if (i + 1) % self.resnet_every == 0:
                    mapping = self.residual_mappings[res_block_index]
                    residual = mapping(block_input)
                    out = self.activation(out + residual)
                    block_input = out
                    res_block_index += 1
        else:
            out = x
            for layer in self.layers:
                out = self.activation(layer(out))
        out = self.fc_out(out)
        return out

# -----------------------------------------------------------------------------
# Helper functions (training, sampling, moving average, etc.)
# -----------------------------------------------------------------------------
def mutual_information(joint, marginal, mine_net):
    """
    Computes the MINE lower bound estimate.
    """
    t = mine_net(joint)
    et = torch.exp(mine_net(marginal))
    mi_lb = torch.mean(t) - torch.log(torch.mean(et))
    return mi_lb, t, et

def learn_mine(batch, mine_net, mine_net_optim, ma_et, ma_rate=0.01):
    """
    Performs a single gradient update on the MINE network.
    """
    joint, marginal = batch
    joint = torch.autograd.Variable(torch.FloatTensor(joint)).cuda()
    marginal = torch.autograd.Variable(torch.FloatTensor(marginal)).cuda()
    mi_lb, t, et = mutual_information(joint, marginal, mine_net)
    ma_et = (1 - ma_rate) * ma_et + ma_rate * torch.mean(et)
    loss = -(torch.mean(t) - (1 / ma_et.mean()).detach() * torch.mean(et))
    mine_net_optim.zero_grad()
    loss.backward()
    mine_net_optim.step()
    return mi_lb, ma_et, loss.item()

def sample_batch(data, batch_size, sample_mode='joint'):
    """
    Samples a mini-batch from data.
    """
    dim = data.shape[1] // 2
    if sample_mode == 'joint':
        indices = np.random.choice(data.shape[0], size=batch_size, replace=False)
        batch = data[indices]
    else:
        joint_indices = np.random.choice(data.shape[0], size=batch_size, replace=False)
        marginal_indices = np.random.choice(data.shape[0], size=batch_size, replace=False)
        batch = np.concatenate([data[joint_indices][:, 0:dim],
                                data[marginal_indices][:, dim:2*dim]], axis=1)
    return batch

def train(data, mine_net, mine_net_optim, batch_size=100, iter_num=1000):
    """
    Trains the MINE network.
    """
    mi_estimates = []
    ma_et = 1.
    for i in range(iter_num):
        batch = (sample_batch(data, batch_size=batch_size),
                 sample_batch(data, batch_size=batch_size, sample_mode='marginal'))
        mi_lb, ma_et, train_loss = learn_mine(batch, mine_net, mine_net_optim, ma_et)
        mi_estimates.append(mi_lb.detach().cpu().numpy())
    return mi_estimates

def ma(a, window_size=50):
    """
    Computes the moving average of a list.
    """
    return [np.mean(a[i:i + window_size]) for i in range(0, len(a) - window_size)]

def MINE_esti_bsc(error_prob, num_samples=2000, bit_dim=1,
                  batch_size=100, iter_num=1000,
                  hidden_sizes=None, use_resnet=False, resnet_every=2, lr=1e-2):
    # gen BSC data:
    X = np.random.randint(0, 2, size=(num_samples, bit_dim))
    noise = np.random.binomial(1, error_prob, size=(num_samples, bit_dim))
    Y = (X + noise) % 2
    data = np.concatenate((X, Y), axis=1).astype(np.float32)
    
    input_size = data.shape[1] 
    if hidden_sizes is None:
        hidden_sizes = [128, 128]
    mine_net = DynamicMine(input_size=input_size,
                           hidden_sizes=hidden_sizes,
                           use_resnet=use_resnet,
                           resnet_every=resnet_every).cuda()
    mine_net_optim = optim.Adam(mine_net.parameters(), lr=lr)
    
    mi_estimates = train(data, mine_net, mine_net_optim, batch_size=batch_size, iter_num=iter_num)
    smoothed_mi = ma(mi_estimates, window_size=50)
    final_mi = smoothed_mi[-1] if len(smoothed_mi) > 0 else mi_estimates[-1]
    if bit_dim > 1:
        norm_fac = math.log2(bit_dim)
    else:
        norm_fac = 1
    
    return final_mi/norm_fac, mine_net #renormalize ... weird bug 

# -----------------------------------------------------------------------------
# Example usage
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # Set the error probability for the BSC channel.
    results = []
    real_MIs = []
    start_time = time.time()
    error_probabilities = np.linspace(0.0, 0.5, 20) 
    # Vary the correlation coefficient rou in the range [-0.9, 1) with step 0.1.
    for error_probability in error_probabilities:
        final_mi, trained_net = MINE_esti_bsc(error_prob=error_probability,
                                            num_samples=2000,
                                            bit_dim=256,
                                            batch_size=200,
                                            iter_num=10000,
                                            hidden_sizes=[128, 128],
                                            use_resnet=False,
                                            resnet_every=2,
                                            lr=1e-3)
        
        end_time = time.time()
        # For a BSC channel with uniform input, the true MI is 1 - H(p),
        # where H(p) is the binary entropy.
        H = -error_probability * np.log2(error_probability) - (1 - error_probability) * np.log2(1 - error_probability) if error_probability not in [0,1] else 0
        true_mi = 1 - H
        results.append(final_mi)
        real_MIs.append(true_mi)
        print("Training completed in {:.2f} seconds.".format(end_time - start_time))
        print("Estimated MI: {:.4f}".format(final_mi))
        print("True MI (in bits): {:.4f}".format(true_mi))

    plt.figure(figsize=(10, 6))
    plt.plot(error_probabilities, real_MIs, label='Real MI', marker='o', linestyle='-', color='blue')
    plt.plot(error_probabilities, results, label='Estimated MI', marker='x', linestyle='--', color='red')
    plt.xlabel('Error probability')
    plt.ylabel('Mutual Information (MI)')
    plt.title('Real vs Estimated Mutual Information for the BSC')
    plt.legend()
    plt.grid(True)
    plt.show()