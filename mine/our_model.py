import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time

# -----------------------------------------------------------------------------
# Dynamic MINE network definition
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
            hidden_sizes (list of int): List of neurons per hidden layer.
                                        Defaults to [128, 128] (original MINE).
            output_size (int): Dimension of the output.
            activation (callable): Activation function (default: F.elu).
            use_resnet (bool): If True, a ResNet-like connection is added every 'resnet_every' layers.
            resnet_every (int): Number of layers after which to add a residual connection.
            init_std (float): Standard deviation for weight initialization.
            init_bias (float): Bias initialization constant.
        """
        super(DynamicMine, self).__init__()
        if hidden_sizes is None:
            hidden_sizes = [128, 128]
        self.activation = activation
        self.use_resnet = use_resnet
        self.resnet_every = resnet_every if use_resnet else None

        # Create the list of hidden layers.
        self.layers = nn.ModuleList()
        # When using ResNet connections, we maintain a parallel list for mapping layers.
        self.residual_mappings = nn.ModuleList() if use_resnet else None

        in_dim = input_size
        # Keep track of the input dimension at the start of each residual block.
        block_start_dim = in_dim

        for i, h_dim in enumerate(hidden_sizes):
            layer = nn.Linear(in_dim, h_dim)
            nn.init.normal_(layer.weight, std=init_std)
            nn.init.constant_(layer.bias, init_bias)
            self.layers.append(layer)
            in_dim = h_dim

            # Add a residual mapping every 'resnet_every' layers.
            if self.use_resnet and ((i + 1) % self.resnet_every == 0):
                if block_start_dim != h_dim:
                    mapping = nn.Linear(block_start_dim, h_dim)
                    nn.init.normal_(mapping.weight, std=init_std)
                    nn.init.constant_(mapping.bias, init_bias)
                else:
                    mapping = nn.Identity()
                self.residual_mappings.append(mapping)
                block_start_dim = h_dim

        # Final output layer.
        self.fc_out = nn.Linear(in_dim, output_size)
        nn.init.normal_(self.fc_out.weight, std=init_std)
        nn.init.constant_(self.fc_out.bias, init_bias)

    def forward(self, x):
        if self.use_resnet:
            out = x
            res_block_index = 0
            # Store the input for the current residual block.
            block_input = out
            for i, layer in enumerate(self.layers):
                out = self.activation(layer(out))
                # When the block is complete, add the residual connection.
                if (i + 1) % self.resnet_every == 0:
                    mapping = self.residual_mappings[res_block_index]
                    residual = mapping(block_input)
                    out = self.activation(out + residual)
                    # Start a new residual block.
                    block_input = out
                    res_block_index += 1
        else:
            out = x
            for layer in self.layers:
                out = self.activation(layer(out))
        # Final output layer (typically no activation).
        out = self.fc_out(out)
        return out

# -----------------------------------------------------------------------------
# Utility functions and training code
# -----------------------------------------------------------------------------
def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Example: print parameter count for a DynamicMine with two layers of 256 neurons each.
print("Parameter count:",
      count_parameters(DynamicMine(hidden_sizes=[256, 256])))

def mutual_information(joint, marginal, mine_net):
    """
    Estimate the mutual information lower bound.
    
    Parameters:
        joint: Samples from the joint distribution.
        marginal: Samples from the product of marginals.
        mine_net: The MINE network.
    Returns:
        mi_lb: The lower bound estimate.
        t: The network output for the joint samples.
        et: The exponentiated network output for the marginal samples.
    """
    t = mine_net(joint)
    et = torch.exp(mine_net(marginal))
    mi_lb = torch.mean(t) - torch.log(torch.mean(et))
    return mi_lb, t, et

def learn_mine(batch, mine_net, mine_net_optim, ma_et, ma_rate=0.01):
    """
    Perform one gradient update for the MINE network.
    
    Parameters:
        batch: A tuple (joint, marginal) of training samples.
        mine_net: The MINE network.
        mine_net_optim: Optimizer for the MINE network.
        ma_et: Exponential moving average of et.
        ma_rate: Update rate for the moving average.
    Returns:
        mi_lb: Current mutual information lower bound.
        ma_et: Updated moving average.
        loss.item(): The training loss.
    """
    joint, marginal = batch
    joint = torch.autograd.Variable(torch.FloatTensor(joint)).cuda()
    marginal = torch.autograd.Variable(torch.FloatTensor(marginal)).cuda()
    mi_lb, t, et = mutual_information(joint, marginal, mine_net)
    ma_et = (1 - ma_rate) * ma_et + ma_rate * torch.mean(et)

    # Unbiased loss using moving average.
    loss = -(torch.mean(t) - (1 / ma_et.mean()).detach() * torch.mean(et))
    mine_net_optim.zero_grad()
    loss.backward()  # Use standard PyTorch backward
    mine_net_optim.step()
    return mi_lb, ma_et, loss.item()

def sample_batch(data, batch_size, sample_mode='joint'):
    """
    Sample a mini-batch from the data.
    
    Parameters:
        data: Numpy array of shape (N, 2*dim), where the first half are x and the second half are y.
        batch_size: Number of samples to select.
        sample_mode: 'joint' to sample corresponding pairs, 'marginal' to mix samples.
    Returns:
        batch: A mini-batch of samples.
    """
    dim = data.shape[1] // 2
    if sample_mode == 'joint':
        indices = np.random.choice(data.shape[0], size=batch_size, replace=False)
        batch = data[indices]
    else:
        joint_indices = np.random.choice(data.shape[0], size=batch_size, replace=False)
        marginal_indices = np.random.choice(data.shape[0], size=batch_size, replace=False)
        batch = np.concatenate([data[joint_indices][:, 0:dim],
                                data[marginal_indices][:, dim:2*dim]],
                               axis=1)
    return batch

def train(data, mine_net, mine_net_optim, batch_size=100, iter_num=100):
    """
    Train the MINE network.
    
    Parameters:
        data: The data samples.
        mine_net: The MINE network.
        mine_net_optim: Optimizer for the network.
        batch_size: Mini-batch size.
        iter_num: Number of training iterations.
    Returns:
        result: List of mutual information estimates over iterations.
    """
    result = []
    ma_et = 1.
    for i in range(iter_num):
        batch = (sample_batch(data, batch_size=batch_size),
                 sample_batch(data, batch_size=batch_size, sample_mode='marginal'))
        mi_lb, ma_et, train_loss = learn_mine(batch, mine_net, mine_net_optim, ma_et)
        result.append(mi_lb.detach().cpu().numpy())
    return result

def ma(a, window_size=50):
    """Compute moving average for smoothing."""
    return [np.mean(a[i:i + window_size]) for i in range(0, len(a) - window_size)]

def scale_data(input_tensor):
    """Scale data to the range [-1, 1]."""
    min_val = np.min(input_tensor)
    max_val = np.max(input_tensor)
    scaled_tensor = 2 * (input_tensor - min_val) / (max_val - min_val) - 1
    return scaled_tensor

def MINE_esti(data, batch_size=100, iter_num=1000,
              hidden_sizes=None, use_resnet=False, resnet_every=2, lr=1e-2):
    """
    Estimate mutual information using the MINE estimator.
    
    Parameters:
        data: Numpy array containing the samples (shape: [N, 2*dim]).
        batch_size: Mini-batch size.
        iter_num: Number of training iterations.
        hidden_sizes: List specifying the hidden layers (e.g., [128, 128]).
        use_resnet: Whether to use ResNet-like residual connections.
        resnet_every: Frequency (in layers) to add a residual connection.
        lr: Learning rate.
    Returns:
        final_result: The final mutual information lower bound estimate.
    """
    input_size = data.shape[1]
    # Construct the dynamic MINE network.
    mine_net = DynamicMine(input_size=input_size,
                           hidden_sizes=hidden_sizes,
                           use_resnet=use_resnet,
                           resnet_every=resnet_every).cuda()
    mine_net_optim = optim.Adam(mine_net.parameters(), lr=lr)
    result = train(data, mine_net, mine_net_optim, batch_size=batch_size, iter_num=iter_num)
    result_ma = ma(result, window_size=50)
    final_result = result_ma[-1]
    return final_result

# -----------------------------------------------------------------------------
# Main block: experiment on bivariate normal data
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    times = []
    results = []
    real_MIs = []
    rou_values = np.arange(-0.9, 1, 0.1) 

    dim = 20
    mean_vector = np.zeros(dim)



    # Vary the correlation coefficient rou in the range [-0.9, 1) with step 0.1.
    for rou in rou_values:
        # Generate bivariate normal data.
        
        # Construct covariance matrix: 1s on the diagonal, rho elsewhere
        cov_matrix = np.full((dim, dim), rou)
        np.fill_diagonal(cov_matrix, 1.0)
        data = np.random.multivariate_normal(mean=mean_vector,
                                             cov=cov_matrix,
                                             size=2000)
        print("Data shape for rou =", rou, ":", data.shape)
        start_time = time.time()
        
        # Estimate MI using the dynamic MINE network.
        # To reproduce the original MINE, use hidden_sizes=[128, 128] and use_resnet=False.
        result = MINE_esti(data,
                           batch_size=100,
                           iter_num=10000,
                           hidden_sizes=[512, 512],
                           use_resnet=False,
                           resnet_every=2,
                           lr=1e-2)
        
        end_time = time.time()
        real_MI = -np.log(1 - rou**2) / 2
        real_MIs.append(real_MI)
        times.append(end_time - start_time)
        results.append(result)
        print("Data with rou =", rou, "processed in", end_time - start_time, "seconds.")
        print("Estimated MI:", result, "Real MI:", real_MI)
    
    plt.figure(figsize=(10, 6))
    plt.plot(rou_values, real_MIs, label='Real MI', marker='o', linestyle='-', color='blue')
    plt.plot(rou_values, results, label='Estimated MI', marker='x', linestyle='--', color='red')
    plt.xlabel('Correlation Coefficient (rou)')
    plt.ylabel('Mutual Information (MI)')
    plt.title('Real vs Estimated Mutual Information')
    plt.legend()
    plt.grid(True)
    plt.show()