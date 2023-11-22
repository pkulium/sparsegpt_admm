import math
import time

import torch
import torch.nn as nn
import transformers

from quant import *


DEBUG = True

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
from utils import *
from snip.snip import *
from snip.train import *
from sparse_op import *

import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from net_utils import admm_solve, faster_admm_solve


class SparseGPT:

    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp, out, blocksize=1024):
        if DEBUG:
            self.inp1 = inp
            self.out1 = out
            return
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def fasterprune(
        self, sparsity, prunen=0, prunem=0, blocksize=128, percdamp=.01
    ):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        if hasattr(self, 'quantizer'):
            if not self.quantizer.ready():
                self.quantizer.find_params(W, weight=True)

        tick = time.time()

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        Losses = torch.zeros(self.rows, device=self.dev)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        mask = None

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            if prunen == 0: 
                if mask is not None:
                    mask1 = mask[:, i1:i2]
                else:
                    tmp = W1 ** 2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
                    thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * sparsity)]
                    mask1 = tmp <= thresh
            else:
                mask1 = torch.zeros_like(W1) == 1

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if prunen != 0 and i % prunem == 0:
                    tmp = W1[:, i:(i + prunem)] ** 2 / (torch.diag(Hinv1)[i:(i + prunem)].reshape((1, -1))) ** 2
                    mask1.scatter_(1, i + torch.topk(tmp, prunen, dim=1, largest=False)[1], True)

                q = w.clone()
                q[mask1[:, i]] = 0

                if hasattr(self, 'quantizer'):
                    q = quantize(
                        q.unsqueeze(1), self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq
                    ).flatten()

                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            W[:, i1:i2] = Q1
            Losses += torch.sum(Losses1, 1) / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

            if DEBUG:
                self.layer.weight.data[:, :i2] = W[:, :i2]
                self.layer.weight.data[:, i2:] = W[:, i2:]
                print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
                print(torch.sum(Losses))

        torch.cuda.synchronize()
        print('time %.2f' % (time.time() - tick))
        print('error', torch.sum(Losses).item())

        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        if DEBUG:
            print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))

    def admmprune(
        self, sparsity, prunen=0, prunem=0, blocksize=128, percdamp=.01
    ):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        if hasattr(self, 'quantizer'):
            if not self.quantizer.ready():
                self.quantizer.find_params(W, weight=True)

        tick = time.time()

        del self.H

        N, M = 2, 4
        s = admm_solve(W, N, M)
        # def admm_solve(z, N, M, rho=1, max_iter=1000, tol=1e-4)
        self.layer.weight.data = s.to(dtype=self.layer.weight.data.dtype)
        if DEBUG:
            print('error for admm:')
            print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
    
    def faster_admm_prune(
        self, sparsity, prunen=0, prunem=0, blocksize=128, percdamp=.01
    ):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        if hasattr(self, 'quantizer'):
            if not self.quantizer.ready():
                self.quantizer.find_params(W, weight=True)

        tick = time.time()

        del self.H

        # apply mask from pgd
        dtype = self.layer.weight.data.dtype
        out_features, in_features = self.layer.weight.shape
        model = nn.Linear(in_features=in_features, out_features=out_features, bias=False).to(self.dev)
        model.weight.data = self.layer.weight.data.clone()
        input = self.inp1.clone().squeeze(0) 
        output = self.out1.clone().squeeze(0) 
        # input = self.input.clone().squeeze(0) 
        # output = self.output.clone().squeeze(0) 

        input = input.to(torch.float32)  # Convert data to Float
        output = output.to(torch.float32)  # Now output has shape [2048, 768]
        model = model.to(torch.float32)  # Convert model parameters to Float

        from torch.utils.data import TensorDataset, DataLoader
        dataset = TensorDataset(input, output)
        train_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)

        # Define your hyperparameter grids
        lr_values = [0.001, 0.01, 0.1]
        rho_values = [0.001, 0.01, 0.1]
        max_iter_values = [10, 100]

        # Initialize variables to store the best hyperparameters and the corresponding minimum loss
        best_lr = None
        best_rho = None
        best_max_iter = None
        min_loss = float('inf')

        # Grid search
        for lr in lr_values:
            for rho in rho_values:
                for max_iter in max_iter_values:
                    # Copy the model for each iteration to avoid cumulative training effects
                    temp_model = copy.deepcopy(model)
                    temp_model.train()

                    with torch.enable_grad():
                        w = faster_admm_solve(temp_model, train_loader, W, lr=lr, rho=rho, max_iter=max_iter, tol=1e-4)
                    # Calculate loss
                    # temp_model.weight.data = temp_model.weight.data.to(self.layer.weight.data.dtype)
                    temp_model.weight.data = w.to(self.layer.weight.data.dtype)
                    current_loss = torch.sum((temp_model(self.inp1) - self.out1) ** 2).item()

                    # Update best hyperparameters if current loss is lower
                    if current_loss < min_loss:
                        min_loss = current_loss
                        best_lr = lr
                        best_rho = rho
                        best_max_iter = max_iter
                        self.layer.weight.data = temp_model.weight.data

        # Print the best hyperparameters and the corresponding loss
        print(f"Best lr: {best_lr}, Best rho: {best_rho}, Best max_iter: {best_max_iter}, Minimum Loss: {min_loss}")
        print(f'self.layer.weight.data:{self.layer.weight.data}')

        del model
        del dataset
        del train_loader

        if DEBUG:
            print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
    
    def faster_snip_prune(
        self, sparsity, prunen=0, prunem=0, blocksize=128, percdamp=.01
    ):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        if hasattr(self, 'quantizer'):
            if not self.quantizer.ready():
                self.quantizer.find_params(W, weight=True)

        tick = time.time()

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        Losses = torch.zeros(self.rows, device=self.dev)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        mask = None

        # apply mask from snip
        model = copy.deepcopy(self.layer)
        input = self.inp1.clone().squeeze(0) 
        output = self.out1.clone().squeeze(0) 

        input = input.to(torch.float32)  # Convert data to Float
        output = output.to(torch.float32)  # Now output has shape [2048, 768]
        model = model.to(torch.float32)  # Convert model parameters to Float

        from torch.utils.data import TensorDataset, DataLoader
        dataset = TensorDataset(input, output)
        train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
        with torch.enable_grad():
            model.train()
            mask = SNIP(model, 0.5, train_loader, self.dev)[0]
        del model
        del dataset
        del train_loader
        mask = mask.to(torch.bool)
        # self.layer.weight.data[mask] = 0
        # return
    
        
        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            if prunen == 0: 
                if mask is not None:
                    mask1 = mask[:, i1:i2]
                else:
                    tmp = W1 ** 2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
                    thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * sparsity)]
                    mask1 = tmp <= thresh
            else:
                mask1 = torch.zeros_like(W1) == 1

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if prunen != 0 and i % prunem == 0:
                    tmp = W1[:, i:(i + prunem)] ** 2 / (torch.diag(Hinv1)[i:(i + prunem)].reshape((1, -1))) ** 2
                    mask1.scatter_(1, i + torch.topk(tmp, prunen, dim=1, largest=False)[1], True)

                q = w.clone()
                q[mask1[:, i]] = 0

                if hasattr(self, 'quantizer'):
                    q = quantize(
                        q.unsqueeze(1), self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq
                    ).flatten()

                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            W[:, i1:i2] = Q1
            Losses += torch.sum(Losses1, 1) / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

            if DEBUG:
                self.layer.weight.data[:, :i2] = W[:, :i2]
                self.layer.weight.data[:, i2:] = W[:, i2:]
                print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
                print(torch.sum(Losses))

        torch.cuda.synchronize()
        print('time %.2f' % (time.time() - tick))
        print('error', torch.sum(Losses).item())

        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        if DEBUG:
            print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))

    def faster_pgd_prune(
        self, sparsity, prunen=0, prunem=0, blocksize=128, percdamp=.01
    ):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        if hasattr(self, 'quantizer'):
            if not self.quantizer.ready():
                self.quantizer.find_params(W, weight=True)

        tick = time.time()

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        Losses = torch.zeros(self.rows, device=self.dev)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        mask = None

        # apply mask from pgd
        out_features, in_features = self.layer.weight.shape
        model = None
        model.weight_old = self.layer.weight.data
        # nn.init.kaiming_normal_(model.weight, mode='fan_out')
        # nn.init.uniform_(model.weight, 0, 1)

        input = self.inp1.clone().squeeze(0) 
        output = self.out1.clone().squeeze(0) 

        input = input.to(torch.float32)  # Convert data to Float
        output = output.to(torch.float32)  # Now output has shape [2048, 768]
        model = model.to(torch.float32)  # Convert model parameters to Float

        from torch.utils.data import TensorDataset, DataLoader
        dataset = TensorDataset(input, output)
        train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
        with torch.enable_grad():
            model.train()
            mask = PGD(model, 0.5, train_loader, self.dev)
            # print(f'shape1 {torch.sum(mask) / (model.weight_mask.shape[0] * model.weight_mask.shape[1])}')
        self.layer.weight.data[model.weight] = 0
        del model
        del dataset
        del train_loader
        return
        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            if prunen == 0: 
                if mask is not None:
                    mask1 = mask[:, i1:i2]
                else:
                    tmp = W1 ** 2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
                    thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * sparsity)]
                    mask1 = tmp <= thresh
            else:
                mask1 = torch.zeros_like(W1) == 1

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if prunen != 0 and i % prunem == 0:
                    tmp = W1[:, i:(i + prunem)] ** 2 / (torch.diag(Hinv1)[i:(i + prunem)].reshape((1, -1))) ** 2
                    mask1.scatter_(1, i + torch.topk(tmp, prunen, dim=1, largest=False)[1], True)

                q = w.clone()
                q[mask1[:, i]] = 0

                if hasattr(self, 'quantizer'):
                    q = quantize(
                        q.unsqueeze(1), self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq
                    ).flatten()

                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            W[:, i1:i2] = Q1
            Losses += torch.sum(Losses1, 1) / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

            if DEBUG:
                self.layer.weight.data[:, :i2] = W[:, :i2]
                self.layer.weight.data[:, i2:] = W[:, i2:]
                print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
                print(torch.sum(Losses))

        torch.cuda.synchronize()
        print('time %.2f' % (time.time() - tick))
        print('error', torch.sum(Losses).item())

        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        if DEBUG:
            print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))

    def faster_vrpge_prune(
        self, sparsity, prunen=0, prunem=0, blocksize=128, percdamp=.01
    ):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        if hasattr(self, 'quantizer'):
            if not self.quantizer.ready():
                self.quantizer.find_params(W, weight=True)

        tick = time.time()

        del self.H
        mask = None

        # apply mask from pgd
        dtype = self.layer.weight.data.dtype
        out_features, in_features = self.layer.weight.shape
        # model = VRPGE(in_features=in_features, out_features=out_features, bias=True).to(self.dev)
        model = VRPGE(
            in_features, out_features, kernel_size=1, stride=1, bias=False
        ).to(self.dev)  
        # Clone and reshape the input
        input = self.inp1.clone().squeeze(0)
        input = input.view(-1, in_features, 1, 1)  # Reshape to (2048, 768, 1, 1)

        # Clone and prepare the output
        output = self.out1.clone().squeeze(0)

        input = input.to(torch.float32)  # Convert data to Float
        output = output.to(torch.float32)  # Now output has shape [2048, 768]
        model = model.to(torch.float32)  # Convert model parameters to Float

        from torch.utils.data import TensorDataset, DataLoader
        dataset = TensorDataset(input, output)
        train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
        with torch.enable_grad():
            model.train()
            # print(f'orign subnet:{model.scores}')
            VRPGE_solve(model, 0.5, train_loader, self.dev)
            # print(f'final subnet:{model.scores}')
            # print(f'ratio:{torch.sum(model.subnet)/ model.subnet.nelement()}')
        # self.layer.weight.data = model.weight.data.clone().to(dtype)
        # self.layer.weight[~model.subnet.data.bool()] = 0


        del model
        del dataset
        del train_loader

        # if isinstance(self.layer, transformers.Conv1D):
            # W = W.t()
        # self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        if DEBUG:
            print(torch.sum((model(input).view(self.out1.shape) - self.out1) ** 2))

    def faster_probmask_prune(
        self, sparsity, prunen=0, prunem=0, blocksize=128, percdamp=.01
    ):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        if hasattr(self, 'quantizer'):
            if not self.quantizer.ready():
                self.quantizer.find_params(W, weight=True)

        tick = time.time()
        del self.H

        # apply mask from pgd
        dtype = self.layer.weight.data.dtype
        out_features, in_features = self.layer.weight.shape
        # model = VRPGE(in_features=in_features, out_features=out_features, bias=True).to(self.dev)
        model = ProbMaskLinear(in_features, out_features, bias=False).to(self.dev)  
        model.weight.data = self.layer.weight.data
        # Clone and reshape the input
        input = self.inp1.clone().squeeze(0) 
        output = self.out1.clone().squeeze(0) 

        input = input.to(torch.float32)  # Convert data to Float
        output = output.to(torch.float32)  # Now output has shape [2048, 768]
        model = model.to(torch.float32)  # Convert model parameters to Float

        from torch.utils.data import TensorDataset, DataLoader
        dataset = TensorDataset(input, output)
        train_loader = DataLoader(dataset, batch_size=256, shuffle=True)

        # Define your hyperparameter grids
        lr_values = [0.001, 0.01, 0.1]
        weight_lr_values = [0]
        max_iter_values = [100, 300]

        # Initialize variables to store the best hyperparameters and the corresponding minimum loss
        best_lr = None
        best_weight_lr = None
        best_max_iter = None
        min_loss = float('inf')

        # Grid search
        for lr in lr_values:
            for weight_lr in weight_lr_values:
                for max_iter in max_iter_values:
                    # Copy the model for each iteration to avoid cumulative training effects
                    temp_model = copy.deepcopy(model)
                    temp_model.train()

                    with torch.enable_grad():
                        mtemp_modelodel = Probmask_solve(temp_model, 0.1, train_loader, self.dev, lr = lr, epochs=max_iter)
                    # Calculate loss
                    # temp_model.weight.data = temp_model.weight.data.to(self.layer.weight.data.dtype)
                    current_loss = torch.sum((temp_model(self.inp1.to(torch.float32)) - self.out1.to(torch.float32)) ** 2).item()

                    if best_lr is None:
                        temp_model.fix_subnet()
                        self.layer.weight.data = (temp_model.subnet * temp_model.weight.data).to(dtype)
                    # Update best hyperparameters if current loss is lower
                    if current_loss < min_loss:
                        min_loss = current_loss
                        best_lr = lr
                        best_weight_lr = weight_lr
                        best_max_iter = max_iter
                        self.layer.weight.data = (temp_model.subnet * temp_model.weight.data).to(dtype)

        # Print the best hyperparameters and the corresponding loss
        print(f"Best lr: {best_lr}, Best best_weight_lr: {best_weight_lr}, Best max_iter: {best_max_iter}, Minimum Loss: {min_loss}")

        if DEBUG:
            print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
    
        del model
        del dataset
        del train_loader

    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        torch.cuda.empty_cache()
