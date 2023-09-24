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

        in_features = self.layer.in_features  # Get the number of input features of the old model
        out_features = self.layer.out_features  # Get the number of output features of the old model
        

        model = nn.Linear(in_features, out_features)  # Create a new linear model with the same specifications
        model = model.to(self.dev)
        model.weight.data = self.layer.weight.data.clone()  # Copy the weights from the old model to the new model
        model.bias.data = self.layer.bias.data.clone()  # Copy the weights from the old model to the new model
        self.lr = 1e-3
        self.adam_epsilon = 1e-8
        self.alpha = 5e-4
        self.rho = 1e-2
        optimizer = PruneAdam(model.named_parameters(), lr=self.lr, eps=self.adam_epsilon)
        self.l1 = False
        self.l2 = False
        self.percent = [0.8, 0.92, 0.991, 0.93]
        train(self, model, self.dev, self.inp1.clone(), self.out1.clone(), optimizer)
        mask = apply_l1_prune(model, self.dev, self) if self.l1 else apply_prune(model, self.dev, self)
        print_prune(model)
        model.weight.data = model.weight.data.to(torch.float16)
        model.bias.data = model.bias.data.to(torch.float16)
        self.layer.weight.data = model.weight.data.clone()
        self.layer.bias.data = model.bias.data.clone()
        del model
        if DEBUG:
            print('error for admm:')
            print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
    
    def snipprune(
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

        model = copy.deepcopy(self.layer)
        input = self.inp1.clone().squeeze(0) 
        output = self.out1.clone().squeeze(0) 

        input = input.to(torch.float32)  # Convert data to Float
        output = output.to(torch.float32)  # Now output has shape [2048, 768]
        model = model.to(torch.float32)  # Convert model parameters to Float

        from torch.utils.data import TensorDataset, DataLoader
        dataset = TensorDataset(input, output)
        train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
        criterion = nn.MSELoss()  # Mean Squared Error Loss for regression
        optimizer = optim.SGD(
            [param for name, param in model.named_parameters() if not 'mask' in name],
            lr=INIT_LR,
            momentum=0.9,
            weight_decay=WEIGHT_DECAY_RATE
        )
        num_epochs = 100
        with torch.enable_grad():
            model.train()
            keep_masks = SNIP(model, 0.05, train_loader, self.dev)  
            apply_prune_mask(model, keep_masks)
            for epoch in range(num_epochs):
                # print('Epoch: {}'.format(epoch + 1))
                # for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
                for batch_idx, (data, target) in enumerate(train_loader):
                    data, target = data.to(device), target.to(device)
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)  # Compute the loss
                    loss.backward()
                    optimizer.step()
                print(f'loss:{loss}')


        # with torch.enable_grad():
        #     model.train()
        #     writer = SummaryWriter()
        #     keep_masks = SNIP(model, 0.05, train_loader, self.dev)  
        #     apply_prune_mask(model, keep_masks)
        #     optimiser, lr_scheduler = experiment(model)
        #     trainer = create_supervised_trainer(model, optimiser, nn.MSELoss(), device)
        #     pbar = ProgressBar()
        #     pbar.attach(trainer)

        #     @trainer.on(Events.ITERATION_COMPLETED)
        #     def log_training_loss(engine):
        #         lr_scheduler.step()
        #         iter_in_epoch = (engine.state.iteration - 1) % len(train_loader) + 1
        #         if engine.state.iteration % LOG_INTERVAL == 0:
        #             # pbar.log_message("Epoch[{}] Iteration[{}/{}] Loss: {:.2f}"
        #             #       "".format(engine.state.epoch, iter_in_epoch, len(train_loader), engine.state.output))
        #             writer.add_scalar("training/loss", engine.state.output,
        #                                 engine.state.iteration)
        #     trainer.run(train_loader, EPOCHS)
        model.weight.data = model.weight.data.to(torch.float16)
        model.bias.data = model.bias.data.to(torch.float16)
        self.layer.weight.data = model.weight.data.clone()
        self.layer.bias.data = model.bias.data.clone()
        del model
        if DEBUG:
            print('error for admm:')
            print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))

    
    def nmprune(
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

        in_features = self.layer.in_features  # Get the number of input features of the old model
        out_features = self.layer.out_features  # Get the number of output features of the old model 
        model = SparseLinear(in_features, out_features, True)
        model.weight.data = self.layer.weight.data.clone()
        model.bias.data = self.layer.bias.data.clone()
        input = self.inp1.clone().squeeze(0) 
        output = self.out1.clone().squeeze(0)   
        from torch.utils.data import TensorDataset, DataLoader
        dataset = TensorDataset(input, output)
        train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
        with torch.enable_grad():
            model.train()
            writer = SummaryWriter()
            optimiser, lr_scheduler = experiment(model)
            trainer = create_supervised_trainer(model, optimiser, nn.MSELoss(), device)
            pbar = ProgressBar()
            pbar.attach(trainer)

            @trainer.on(Events.ITERATION_COMPLETED)
            def log_training_loss(engine):
                lr_scheduler.step()
                iter_in_epoch = (engine.state.iteration - 1) % len(train_loader) + 1
                if engine.state.iteration % LOG_INTERVAL == 0:
                    # pbar.log_message("Epoch[{}] Iteration[{}/{}] Loss: {:.2f}"
                    #       "".format(engine.state.epoch, iter_in_epoch, len(train_loader), engine.state.output))
                    writer.add_scalar("training/loss", engine.state.output,
                                        engine.state.iteration)
            trainer.run(train_loader, EPOCHS)
        self.layer.weight.data = model.weight.data.clone()
        self.layer.bias.data = model.bias.data.clone()
        del model
        if DEBUG:
            print('error for admm:')
            print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))

    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        torch.cuda.empty_cache()
