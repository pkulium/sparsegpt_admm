import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
import types

from typing import Union, Dict, Tuple
import numpy as np

DEBUG = True
TensorType = Union[torch.Tensor, np.ndarray]

def clip_mask(model, lower=0.0, upper=1.0):
    params = [param for name, param in model.named_parameters() if 'neuron_mask' in name]
    with torch.no_grad():
        for param in params:
            param.clamp_(lower, upper)

def snip_forward_conv2d(self, x):
        return F.conv2d(x, self.weight * self.weight_mask, self.bias,
                        self.stride, self.padding, self.dilation, self.groups)


def snip_forward_linear(self, x):
        return F.linear(x, self.weight * self.weight_mask, self.bias)


def SNIP(model, keep_ratio, train_dataloader, device):
    # TODO: shuffle?

    # Grab a single batch from the training dataset
    inputs, targets = next(iter(train_dataloader))
    inputs = inputs.to(device)
    targets = targets.to(device)

    # Let's create a fresh copy of the modelwork so that we're not worried about
    # affecting the actual training-phase
    model = copy.deepcopy(model)

    # Monkey-patch the Linear and Conv2d layer to learn the multiplicative mask
    # instead of the weights
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight))
            nn.init.xavier_normal_(layer.weight_mask)
            layer.weight.requires_grad = False
            

        # Override the forward methods:
        if isinstance(layer, nn.Conv2d):
            layer.forward = types.MethodType(snip_forward_conv2d, layer)

        if isinstance(layer, nn.Linear):
            layer.forward = types.MethodType(snip_forward_linear, layer)

    # Compute gradients (but don't apply them)
    model.zero_grad()
    outputs = model.forward(inputs)
    criterion = nn.MSELoss()  # Mean Squared Error Loss for regression
    loss = criterion(outputs, targets)  # Compute the loss
    loss.backward()

    grads_abs = []
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            grads_abs.append(torch.abs(layer.weight_mask.grad))

    # Gather all scores in a single vector and normalise
    all_scores = torch.cat([torch.flatten(x) for x in grads_abs])
    norm_factor = torch.sum(all_scores)
    all_scores.div_(norm_factor)

    num_params_to_keep = int(len(all_scores) * keep_ratio)
    threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
    acceptable_score = threshold[-1]

    keep_masks = []
    for g in grads_abs:
        keep_masks.append(((g / norm_factor) >= acceptable_score).float())

    # print(torch.sum(torch.cat([torch.flatten(x == 1) for x in keep_masks])))

    return(keep_masks)

def maskNxM(
    parameter: TensorType,
    n: int,
    m: int
) -> TensorType:
    """
    Accepts either a torch.Tensor or numpy.ndarray and generates a floating point mask of 1's and 0's
    corresponding to the locations that should be retained for NxM pruning. The appropriate ranking mechanism
    should already be built into the parameter when this method is called.
    """

    if type(parameter) is torch.Tensor:
        out_neurons, in_neurons = parameter.size()

        with torch.no_grad():
            groups = parameter.reshape(out_neurons, -1, n)
            zeros = torch.zeros(1, 1, 1, device=parameter.device)
            ones = torch.ones(1, 1, 1, device=parameter.device)

            percentile = m / n
            quantiles = torch.quantile(groups, percentile, -1, keepdim=True)
            mask = torch.where(groups > quantiles, ones, zeros).reshape(out_neurons, in_neurons)
    else:
        out_neurons, in_neurons = parameter.shape
        percentile = (100 * m) / n

        groups = parameter.reshape(out_neurons, -1, n)
        group_thresholds = np.percentile(groups, percentile, axis=-1, keepdims=True)
        mask = (groups > group_thresholds).astype(np.float32).reshape(out_neurons, in_neurons)

    return mask

M, N = 4, 2
def SNIP_solve(model, train_loader, lr, max_iter, rho, tol):
    # Monkey-patch the Linear and Conv2d layer to learn the multiplicative mask
    # instead of the weights
    init_constant = 0.5
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight) * init_constant)
            # nn.init.xavier_normal_(layer.weight_mask)
            layer.weight.requires_grad = False
            layer.weight_mask.requires_grad = True
            
        # Override the forward methods:
        if isinstance(layer, nn.Conv2d):
            layer.forward = types.MethodType(snip_forward_conv2d, layer)

        if isinstance(layer, nn.Linear):
            layer.forward = types.MethodType(snip_forward_linear, layer)

    device = 'cuda:0'
    W = torch.zeros_like(model.weight.data)
    u = torch.zeros_like(model.weight.data)

    # Define the optimizer
    optimizer = torch.optim.Adam([model.weight_mask], lr=lr)
    # Define the learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iter)
    criterion = nn.MSELoss()
    # Assume train_loader is already defined and provides batches of (input, output_a)
    for epoch in range(max_iter):  # Number of epochs
        if DEBUG and epoch % 10 == 0:
            print(f'model.weight_mask:{model.weight_mask[0:8, 0:8]}')
            print(f'W:{W[0:8, 0:8]}')
        for input_tensor, label in train_loader:  # label is output_a
            # admm_adjust_learning_rate(optimizer, epoch, config)
            input_tensor, label = input_tensor.to(device), label.to(device)
            optimizer.zero_grad()
            # Forward pass
            output_model = model(input_tensor)
            # Compute the loss
            # loss_mse = criterion(output_model, label) 
            loss_mse = torch.sum((output_model - label) ** 2)
            admm_loss = 0.5*rho*torch.linalg.norm(model.weight - W + u, "fro") ** 2
            print(f'loss_mse:{loss_mse}')
            print(f'admm_loss:{admm_loss}')
            loss_mse += admm_loss
            loss_mse.backward()
            optimizer.step()
            model.weight_mask.data = torch.clamp(model.weight_mask.data, 0, 1)

         # Update W
        with torch.no_grad():
            values = model.weight.data + u
            scores = values.abs()
            mask = maskNxM(scores, M, N)
            W = mask * values

        # Update u
        u += model.weight.data - W

        # Update the learning rate
        scheduler.step()

        # # Update W
        # with torch.no_grad():
        #     values = model.weight.data + u
        #     scores = values.abs()
        #     mask = maskNxM(scores, M, N)
        #     W = mask * values

        # # Update u
        # u += model.weight.data - W

        # Check for convergence
        primal_res = torch.norm(model.weight.data - W)
        dual_res = torch.norm(-rho * (W - values))
        if primal_res < tol and dual_res < tol:
            break

    with torch.no_grad():
        values = model.weight.data + u
        scores = values.abs()
        mask = maskNxM(scores, M, N)
        W = mask 
        
    if DEBUG:
        print(f'primal_res:{primal_res}')
        print(f'dual_res:{dual_res}')
        print(f'model.weight_mask:{model.weight_mask[0:8, 0:8]}')
        print(f'W:{W[0:8, 0:8]}')
        
    return model.weight_mask, W


def sparsity_loss(tensor):
    # Reshape tensor to expose each chunk of 4 as a separate row
    reshaped = tensor.view(-1, 4)
    
    # Calculate the sum along each row (i.e., chunk) and compute the loss for each
    chunk_sums = torch.sum(reshaped, dim=1)
    loss_vector = torch.abs(chunk_sums - 2)  # Loss for each chunk of 4

    # Sum up all individual chunk losses
    total_loss = torch.sum(loss_vector)

    return total_loss


def PGD(model, keep_ratio, train_dataloader, device):

    # Let's create a fresh copy of the modelwork so that we're not worried about
    # affecting the actual training-phase
    # model = copy.deepcopy(model)

    # Monkey-patch the Linear and Conv2d layer to learn the multiplicative mask
    # instead of the weights

    # for layer in model.modules():
    #     if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
    #         layer.weight_mask = nn.Parameter(torch.zeros_like(layer.weight))
    #         # nn.init.xavier_normal_(layer.weight_mask)
    #         nn.init.constant_(layer.weight_mask, keep_ratio)
    #         layer.weight.requires_grad = False
    #         layer.bias.requires_grad = False
            

    #     # Override the forward methods:
    #     if isinstance(layer, nn.Conv2d):
    #         layer.forward = types.MethodType(snip_forward_conv2d, layer)

    #     if isinstance(layer, nn.Linear):
    #         layer.forward = types.MethodType(snip_forward_linear, layer)

    # W_metric = torch.abs(model.weight.data)
    # thresh = torch.sort(W_metric.flatten().cuda())[0][int(model.weight.numel()*keep_ratio)].cpu()
    # W_mask = (W_metric<=thresh).int()
    criterion = nn.MSELoss()  # Mean Squared Error Loss for regression
    # mask_optimizer = torch.optim.SGD([model.weight_mask], lr=0.001, momentum=0.9)
    mask_optimizer = torch.optim.AdamW([model.weight], lr=0.01)
    rho = 0.001  # You can adjust tsshis value to change the strength of the regularization
    total_epoch = 1000
    total_param = model.weight.shape[0] * model.weight.shape[1]
    for epoch in range(total_epoch):
        for i, (inputs, targets) in enumerate(train_dataloader):
            inputs, targets = inputs.to(device), targets.to(device)

            # step 2: calculate loss and update the mask values
            mask_optimizer.zero_grad()
            model.weight.data.copy_(model.weight_org)
            outputs = model.forward(inputs)
            loss = criterion(outputs, targets)  # Compute the loss
            # loss_reg = sparsity_loss(model.weight)
            loss_reg = model.weight.abs().sum()
            loss += loss_reg
            loss.backward()
            mask_optimizer.step()
            clip_mask(model)
            model.weight_org.data.copy_(model.weight.data.clamp_(0,1))
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")
            if epoch % 100 == 0:
                print(model.weight)
    
    # num_params_to_keep = int(model.weight_mask.shape[0] * model.weight_mask.shape[1] * keep_ratio)
    # threshold, _ = torch.topk(torch.flatten(model.weight_mask), num_params_to_keep, sorted=True)
    # acceptable_score = threshold[-1]
    # keep_masks = model.weight_mask > acceptable_score
    # return keep_masks


def adjust_learning_rate(optimizer, epoch):
    lr = 0.1
    lr_schedule = [56,71]
    lr_drops = [0.1, 0.1]
    assert len(lr_schedule) == len(lr_drops), "length of gammas and schedule should be equal"
    for (drop, step) in zip(lr_drops, lr_schedule):
        if (epoch >= step): lr = lr * drop
        else: break
    for param_group in optimizer.param_groups: param_group['lr'] = lr

def assign_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr

import torch.optim as optim
def VRPGE_solve(model, keep_ratio, train_loader, device):

    def solve_v_total(model, total):
        k = total * keep_ratio
        a, b = 0, 0
        b = max(b, model.scores.max())
        def f(v):
            s = (model.scores - v).clamp(0, 1).sum()
            return s - k
        if f(0) < 0:
            return 0, 0
        itr = 0
        while (1):
            itr += 1
            v = (a + b) / 2
            obj = f(v)
            if abs(obj) < 1e-3 or itr > 20:
                break
            if obj < 0:
                b = v
            else:
                a = v
        v = max(0, v)
        return v, itr
    import numpy as np

    def _warmup_lr(base_lr, warmup_length, epoch):
        return base_lr * (epoch + 1) / warmup_length
    
    def assign_learning_rate(optimizer, new_lr):
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr

    def cosine_lr(optimizer, warmup_length, epochs, lr):
        def _lr_adjuster(epoch, iteration, lr):
            if epoch < warmup_length:
                lr = _warmup_lr(lr, warmup_length, epoch)
            else:
                e = epoch - warmup_length
                es = epochs - warmup_length
                lr = 0.5 * (1 + np.cos(np.pi * e / es)) * lr
            assign_learning_rate(optimizer, lr)
            return lr
        return _lr_adjuster
    
    parameters = list(model.named_parameters())
    score_params = [v for n, v in parameters if ("score" in n) and v.requires_grad]
    weight_opt = None

    model.weight.requires_grad = True
    weight_lr = 0.01
    weight_params = [v for n, v in parameters if ("score" not in n) and v.requires_grad]
    weight_opt = torch.optim.SGD(
        weight_params,
        weight_lr,
        momentum=0.9,
        weight_decay=5e-4,
        nesterov=False,
    )

    lr = 12e-3
    optimizer = torch.optim.Adam(
        score_params, lr=lr, weight_decay=0
    )
    epochs = 100
    criterion = nn.MSELoss()
    K = 20
    lr_policy = cosine_lr(optimizer, 0, epochs, lr)
    for epoch in range(epochs):  # Number of epochs
        assign_learning_rate(optimizer, 0.5 * (1 + np.cos(np.pi * epoch / epochs)) * lr)
        assign_learning_rate(weight_opt, 0.5 * (1 + np.cos(np.pi * epoch / epochs)) * weight_lr)
        for i, (image, target) in enumerate(train_loader):
            image = image.cuda('cuda:0', non_blocking=True)
            target = target.cuda('cuda:0', non_blocking=True)
            l = 0
            optimizer.zero_grad()
            if weight_opt is not None:
                weight_opt.zero_grad()
            fn_list = []
            for j in range(K):
                model.j = j
                output = model(image)
                original_loss = criterion(output.view(target.shape), target)
                # original_loss = torch.sum((target - output) ** 2)
                # print(f'original_loss:{original_loss}')
                # print(f'submodel:{model.submodel}')
                loss = original_loss/K
                fn_list.append(loss.item()*K)
                loss.backward(retain_graph=True)
                l = l + loss.item()
            fn_avg = l
            model.scores.grad.data += 1/(K-1)*(fn_list[0] - fn_avg)*getattr(model, 'stored_mask_0') + 1/(K-1)*(fn_list[1] - fn_avg)*getattr(model, 'stored_mask_1')
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3)
            optimizer.step()
            if weight_opt is not None:
                weight_opt.step()
            with torch.no_grad():
                total = model.scores.nelement()
                v, itr = solve_v_total(model, total)
                model.scores.sub_(v).clamp_(0, 1)     

        if epoch % 50 == 0:
            print(f'loss: {loss}')


def Probmask_solve(model, prune_rate, train_loader, device, lr = 12e-3, epochs = 100, weight_lr = 0.1):
    def solve_v_total(model, total):
        k = total * prune_rate
        a, b = 0, 0
        b = max(b, model.scores.max())
        def f(v):
            s = (model.scores - v).clamp(0, 1).sum()
            return s - k
        if f(0) < 0:
            return 0, 0
        itr = 0
        while (1):
            itr += 1
            v = (a + b) / 2
            obj = f(v)
            if abs(obj) < 1e-3 or itr > 20:
                break
            if obj < 0:
                b = v
            else:
                a = v
        v = max(0, v)
        return v, itr
    
    import numpy as np
    def _warmup_lr(base_lr, warmup_length, epoch):
        return base_lr * (epoch + 1) / warmup_length
    
    def assign_learning_rate(optimizer, new_lr):
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr

    def cosine_lr(optimizer, warmup_length, epochs, lr):
        def _lr_adjuster(epoch, iteration, lr):
            if epoch < warmup_length:
                lr = _warmup_lr(lr, warmup_length, epoch)
            else:
                e = epoch - warmup_length
                es = epochs - warmup_length
                lr = 0.5 * (1 + np.cos(np.pi * e / es)) * lr
            assign_learning_rate(optimizer, lr)
            return lr
        return _lr_adjuster
    
    parameters = list(model.named_parameters())
    score_params = [v for n, v in parameters if ("score" in n) and v.requires_grad]
    optimizer = torch.optim.Adam(score_params, lr=lr, weight_decay=0)

    weight_opt = None
    model.weight.requires_grad = True
    weight_params = [v for n, v in parameters if ("score" not in n) and v.requires_grad]
    weight_opt = torch.optim.SGD(
        weight_params,
        weight_lr,
        momentum=0.9,
        weight_decay=5e-4,
        nesterov=False,
    )

    criterion = nn.MSELoss()
    K = 20
    lr_policy = cosine_lr(optimizer, 0, epochs, lr)
    ts = 0.16
    pr_target = prune_rate
    pr_start = 1.0
    ts = int(ts * epochs)
    te = 0.6
    te = int(te * epochs)

    for epoch in range(epochs):  # Number of epochs

        assign_learning_rate(optimizer, 0.5 * (1 + np.cos(np.pi * epoch / epochs)) * lr)
        if weight_opt:
            assign_learning_rate(weight_opt, 0.5 * (1 + np.cos(np.pi * epoch / epochs)) * weight_lr)
        if epoch < ts:
            model.prune_rate = 1
        elif epoch < te:
            model.prune_rate = pr_target + (pr_start - pr_target)*(1-(epoch-ts)/(te-ts))**3
        else:
            model.prune_rate = pr_target
        model.T = 1 / ((1 - 0.03) * (1 - epoch / epochs) + 0.03)

        for i, (input, target) in enumerate(train_loader):
            input = input.cuda('cuda:0', non_blocking=True)
            target = target.cuda('cuda:0', non_blocking=True)
            optimizer.zero_grad()
            if weight_opt is not None:
                weight_opt.zero_grad()
            for j in range(K):
                output = model(input)
                # loss = criterion(output.view(target.shape), target) / K
                loss = (torch.sum(output - target) ** 2) / K
                loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3)
            optimizer.step()
            if weight_opt is not None:
                weight_opt.step()
            with torch.no_grad():
                total = model.scores.nelement()
                v, itr = solve_v_total(model, total)
                model.scores.sub_(v).clamp_(0, 1)     
        # if epoch % 10 == 0:
            # print(f'loss: {loss}')
    return model

def mask_solve(model, train_dataloader, device):
    # Monkey-patch the Linear and Conv2d layer to learn the multiplicative mask
    # instead of the weights
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            layer.weight_mask = nn.Parameter((layer.weight.data != 0).float())
            layer.weight_mask.requires_grad = False

        # Override the forward methods:
        if isinstance(layer, nn.Conv2d):
            layer.forward = types.MethodType(snip_forward_conv2d, layer)

        if isinstance(layer, nn.Linear):
            layer.forward = types.MethodType(snip_forward_linear, layer)

    criterion = nn.MSELoss()  # Mean Squared Error Loss for regression
    # mask_optimizer = torch.optim.SGD([model.weight_mask], lr=0.001, momentum=0.9)
    optimizer = torch.optim.AdamW([model.weight], lr=0.0001)
    total_epoch = 100
    for epoch in range(total_epoch):
        for i, (inputs, targets) in enumerate(train_dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            # step 2: calculate loss and update the mask values
            optimizer.zero_grad()
            outputs = model.forward(inputs)
            # loss = criterion(outputs, targets)  # Compute the loss
            loss = (torch.sum(outputs - targets) ** 2)
            loss.backward()
            optimizer.step()
    print(f"Epoch {epoch}, Loss: {loss.item()}")
    return model
