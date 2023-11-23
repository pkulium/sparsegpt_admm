import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
import types

import abc
import tqdm
import wandb

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


def SNIP(net, keep_ratio, train_dataloader, device):
    # TODO: shuffle?

    # Grab a single batch from the training dataset
    inputs, targets = next(iter(train_dataloader))
    inputs = inputs.to(device)
    targets = targets.to(device)

    # Let's create a fresh copy of the network so that we're not worried about
    # affecting the actual training-phase
    net = copy.deepcopy(net)

    # Monkey-patch the Linear and Conv2d layer to learn the multiplicative mask
    # instead of the weights
    for layer in net.modules():
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
    net.zero_grad()
    outputs = net.forward(inputs)
    criterion = nn.MSELoss()  # Mean Squared Error Loss for regression
    loss = criterion(outputs, targets)  # Compute the loss
    loss.backward()

    grads_abs = []
    for layer in net.modules():
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

def sparsity_loss(tensor):
    # Reshape tensor to expose each chunk of 4 as a separate row
    reshaped = tensor.view(-1, 4)
    
    # Calculate the sum along each row (i.e., chunk) and compute the loss for each
    chunk_sums = torch.sum(reshaped, dim=1)
    loss_vector = torch.abs(chunk_sums - 2)  # Loss for each chunk of 4

    # Sum up all individual chunk losses
    total_loss = torch.sum(loss_vector)

    return total_loss


def PGD(net, keep_ratio, train_dataloader, device):

    # Let's create a fresh copy of the network so that we're not worried about
    # affecting the actual training-phase
    # net = copy.deepcopy(net)

    # Monkey-patch the Linear and Conv2d layer to learn the multiplicative mask
    # instead of the weights

    # for layer in net.modules():
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

    # W_metric = torch.abs(net.weight.data)
    # thresh = torch.sort(W_metric.flatten().cuda())[0][int(net.weight.numel()*keep_ratio)].cpu()
    # W_mask = (W_metric<=thresh).int()
    criterion = nn.MSELoss()  # Mean Squared Error Loss for regression
    # mask_optimizer = torch.optim.SGD([net.weight_mask], lr=0.001, momentum=0.9)
    mask_optimizer = torch.optim.AdamW([net.weight], lr=0.01)
    rho = 0.001  # You can adjust tsshis value to change the strength of the regularization
    total_epoch = 1000
    total_param = net.weight.shape[0] * net.weight.shape[1]
    for epoch in range(total_epoch):
        for i, (inputs, targets) in enumerate(train_dataloader):
            inputs, targets = inputs.to(device), targets.to(device)

            # step 2: calculate loss and update the mask values
            mask_optimizer.zero_grad()
            net.weight.data.copy_(net.weight_org)
            outputs = net.forward(inputs)
            loss = criterion(outputs, targets)  # Compute the loss
            # loss_reg = sparsity_loss(net.weight)
            loss_reg = net.weight.abs().sum()
            loss += loss_reg
            loss.backward()
            mask_optimizer.step()
            clip_mask(net)
            net.weight_org.data.copy_(net.weight.data.clamp_(0,1))
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")
            if epoch % 100 == 0:
                print(net.weight)
    
    # num_params_to_keep = int(net.weight_mask.shape[0] * net.weight_mask.shape[1] * keep_ratio)
    # threshold, _ = torch.topk(torch.flatten(net.weight_mask), num_params_to_keep, sorted=True)
    # acceptable_score = threshold[-1]
    # keep_masks = net.weight_mask > acceptable_score
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
                # print(f'subnet:{model.subnet}')
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

def mask_solve(net, train_dataloader, device):
    # Monkey-patch the Linear and Conv2d layer to learn the multiplicative mask
    # instead of the weights
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            layer.weight_mask = nn.Parameter((layer.weight.data != 0).float())
            layer.weight_mask.requires_grad = False

        # Override the forward methods:
        if isinstance(layer, nn.Conv2d):
            layer.forward = types.MethodType(snip_forward_conv2d, layer)

        if isinstance(layer, nn.Linear):
            layer.forward = types.MethodType(snip_forward_linear, layer)

    criterion = nn.MSELoss()  # Mean Squared Error Loss for regression
    # mask_optimizer = torch.optim.SGD([net.weight_mask], lr=0.001, momentum=0.9)
    optimizer = torch.optim.AdamW([net.weight], lr=0.01)
    total_epoch = 100
    for epoch in range(total_epoch):
        for i, (inputs, targets) in enumerate(train_dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            # step 2: calculate loss and update the mask values
            optimizer.zero_grad()
            outputs = net.forward(inputs)
            # loss = criterion(outputs, targets)  # Compute the loss
            loss = (torch.sum(outputs - targets) ** 2)
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")
