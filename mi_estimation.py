import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import time
import math
import seaborn as sns
import pandas as pd
from estimators.miEstimator import *
#import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

def sample_correlated_gaussian(rho=0.5, dim=20, batch_size=128, to_cuda=False, cubic = False):
    """Generate samples from a correlated Gaussian distribution."""
    mean = [0,0]
    cov = [[1.0, rho],[rho, 1.0]]
    x, y = np.random.multivariate_normal(mean, cov, batch_size * dim).T

    x = x.reshape(-1, dim)
    y = y.reshape(-1, dim)

    if cubic:
        y = y ** 3

    if to_cuda:
        x = torch.from_numpy(x).float().cuda()
        #x = torch.cat([x, torch.randn_like(x).cuda() * 0.3], dim=-1)
        y = torch.from_numpy(y).float().cuda()
    return x, y

def binary_entropy_inverse(H_val):
    """Solve for p given H(p) = ln(2) - I/N using numerical methods"""
    def H(p):
        return -p * np.log2(p) - (1 - p) * np.log2(1 - p)
    
    p_vals = np.linspace(0.0001, 0.4999, 100000)  # Avoid log(0)
    H_vals = H(p_vals)
    
    # Find the closest value in H_vals to H_val
    idx = (np.abs(H_vals - H_val)).argmin()
    return p_vals[idx]


def sample_bsc_pairs(
        batch_size=128, 
        dim=20, 
        p=0.5, 
        to_cuda=False
    ):
        """
        Samples (X, Y) pairs from a Binary Symmetric Channel (BSC) at 
        an error rate p that yields the desired mutual information.
        
        Args:
            batch_size (int): Number of bitvectors to sample.
            dim (int): Bitvector length (dimensions).
            desired_mi (float): Desired mutual information I(X;Y).
            to_cuda (bool): If True, returns torch.cuda tensors.
        
        Returns:
            X, Y: Either NumPy arrays of shape (batch_size, dim) 
                or torch.FloatTensors on GPU if to_cuda=True.
                Both contain {0,1} bits.
        """
        # Sample random bitvectors for X
        # X unif {0, 1}^(dim)
        X = np.random.randint(0, 2, size=(batch_size, dim), dtype=np.int8)
        
        # Sample error bits E ~ Bernoulli(p) for each bit in X
        E = np.random.rand(batch_size, dim) < p
        E = E.astype(np.int8)
        
        # Through BSC: Y = X ^ E (mod 2)
        Y = (X ^ E).astype(np.int8)
        
        # If requested, move to Torch (GPU)
        if to_cuda:
            X = torch.from_numpy(X).float().cuda() - 0.5
            Y = torch.from_numpy(Y).float().cuda() - 0.5
        else:
            X = X.astype(np.float32) - 0.5
            Y = Y.astype(np.float32) - 0.5
        
        return X, Y


def rho_to_mi(rho, dim):
    result = -dim / 2 * np.log(1 - rho **2)
    return result

def mi_to_rho(mi, dim):
    result = np.sqrt(1 - np.exp(-2 * mi / (dim*np.log2(np.e))))
    return result

############################

sample_dim = 512
batch_size = 640
hidden_size = math.ceil(1024)
learning_rate = 0.0005*2
training_steps = 4000

cubic = False 
model_list = ["CLUBSampleLarge","CLUBSample","L1OutUB"]

p_list = np.linspace(0.1,0.4999,5)

def MutualInfo(p,dim):
        return dim*(1 - (-p * np.log2(p) - (1 - p) * np.log2(1 - p)))

mi_list = [MutualInfo(p,sample_dim) for p in p_list]
print(mi_list)
total_steps = training_steps*len(p_list)
######################

# train MI estimators with samples 

mi_results = dict()
for i, model_name in enumerate(model_list):
    

    mi_est_values = []

    start_time = time.time()
    for i, p in enumerate(p_list):
        model = eval(model_name)(sample_dim, sample_dim, hidden_size).cuda()
        optimizer = torch.optim.Adam(model.parameters(), learning_rate, amsgrad = True)
        for step in range(training_steps):

            batch_x, batch_y = sample_bsc_pairs(p=p, dim=sample_dim, batch_size = batch_size, to_cuda = True,)

            #batch_x, batch_y = sample_correlated_gaussian(rho=mi_to_rho(mi_list[i],dim=sample_dim),dim=sample_dim, batch_size = batch_size, to_cuda = True,)
            model.eval()
            mi_est_values.append(model(batch_x, batch_y).item()*np.log2(np.e))
            
            model.train() 
  
            model_loss = model.learning_loss(batch_x, batch_y)
           
            optimizer.zero_grad()
            model_loss.backward()
            optimizer.step()
            
            del batch_x, batch_y
            torch.cuda.empty_cache()
            if step > 50:
                ma = np.mean(mi_est_values[-50:])
            else:
                ma = 0


            print(f"Processing step: {step}/{training_steps} : MI-Mean:{np.mean(mi_est_values)} MI-latest {ma}", end='\r', flush=True)

        print("\n")
        print("finish training for %s with true MI value = %f" % (model.__class__.__name__, mi_list[i]))
        # torch.save(model.state_dict(), "./model/%s_%d.pt" % (model.__class__.__name__, int(mi_value)))
        torch.cuda.empty_cache()
    end_time = time.time()
    time_cost = end_time - start_time
    print("model %s average time cost is %f s" % (model_name, time_cost/total_steps))
    mi_results[model_name] = mi_est_values


EMA_SPAN = 200
colors = sns.color_palette()
ncols = len(model_list)
nrows = 1
fig, axs = plt.subplots(nrows, ncols, figsize=(3.1 *ncols , 3.4 * nrows))
axs = np.ravel(axs)


xaxis = np.array(list(range(total_steps)))
yaxis_mi = np.repeat(mi_list, training_steps)

for i, model_name in enumerate(model_list):
    plt.sca(axs[i])
    p1 = plt.plot(mi_results[model_name], alpha=0.4, color=colors[0])[0]  #color = 5 or 0
    mis_smooth = pd.Series(mi_results[model_name]).ewm(span=EMA_SPAN).mean()
    
    if i == 0:
        plt.plot(mis_smooth, c=p1.get_color(), label='Estimated MI')
        plt.plot(yaxis_mi, color='k', label='True MI')
        plt.xlabel('Steps', fontsize= 14)
        plt.ylabel('Mutual Information', fontsize = 14)
        plt.legend(loc='upper left', prop={'size':15})
    else:
        plt.plot(mis_smooth, c=p1.get_color())
        plt.yticks([])
        plt.plot(yaxis_mi, color='k')
        plt.xticks([])
    
    plt.ylim(0, sample_dim + sample_dim*0.05)
    plt.xlim(0, total_steps)   
    plt.title(model_name, fontsize=15)
    #plt.subplots_adjust( )

plt.show()
# plt.savefig('mi_est_Gaussian.pdf', bbox_inches=None)