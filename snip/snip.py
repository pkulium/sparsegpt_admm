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


import torch.optim as optim
def pro_mask(model, keep_ratio, train_dataloader, device):
   # Define the optimizer, you can use any optimizer of your choice
    prune_rate = 0.5
    def solve_v_total(model, total):
        k = total * prune_rate
        a, b = 0, 0
        b = max(b, model.scores.max())
        def f(v):
            s = 0
            s += (model.scores - v).clamp(0, 1).sum()
            return s - k
        if f(0) < 0:
            return 0
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
        return v
    
    import torch.optim as optim
    model.train_weights = False
    model.weight.requires_grad = False
    model.bias.requires_grad = False

    # Define the optimizer, loss function, and regularization strength
    parameters = list(model.named_parameters())
    # weight_params = [v for n, v in parameters if ("score" not in n) and v.requires_grad]
    score_params = [v for n, v in parameters if ("score" in n) and v.requires_grad]
    optimizer = torch.optim.Adam(
        score_params, lr=0.01, weight_decay=1e-4
    )
    mse_loss = nn.MSELoss()
    # lambda_sparsity = 0.1  # Regularization strength for sparsity constraint
    # Assume train_loader is already defined and provides batches of (input, output_a)
    for epoch in range(10):  # Number of epochs
        for input_tensor, label in train_dataloader:  # label is output_a
            optimizer.zero_grad()
            
            # Forward pass
            output_model = model(input_tensor)
            
            # Compute the loss
            loss_mse = mse_loss(output_model, label)  # Compare output_model with label (output_a)
            loss_mse.backward()

            optimizer.step()

            with torch.no_grad():
                total = model.scores.nelement()
                v = solve_v_total(model, total)
                model.scores.sub_(v).clamp_(0, 1)

        # Print the loss values at the end of each epoch
        # print(f"Epoch {epoch}, MSE Loss: {loss_mse.item()}, Sparsity Constraint: {sparsity_constraint.item()}, Total Loss: {loss.item()}")


def adjust_learning_rate(optimizer, epoch):
    lr = 0.1
    lr_schedule = [56,71]
    lr_drops = [0.1, 0.1]
    assert len(lr_schedule) == len(lr_drops), "length of gammas and schedule should be equal"
    for (drop, step) in zip(lr_drops, lr_schedule):
        if (epoch >= step): lr = lr * drop
        else: break
    for param_group in optimizer.param_groups: param_group['lr'] = lr

import torch.optim as optim
def VRPEG(model, keep_ratio, train_loader, device):
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
    model.weight.requires_grad = False
    parameters = list(model.named_parameters())
    score_params = [v for n, v in parameters if ("score" in n) and v.requires_grad]
    optimizer = torch.optim.Adam(
        score_params, lr=0.1, weight_decay=1e-4
    )
    epochs = 50
    criterion = nn.L1Loss()
    K = 20
    for epoch in range(epochs):  # Number of epochs
        # for i, (image, target) in tqdm.tqdm(
            # enumerate(train_loader), ascii=True, total=len(train_loader)
        # ):
        for i, (image, target) in enumerate(train_loader):
            image = image.cuda('cuda:0', non_blocking=True)
            target = target.cuda('cuda:0', non_blocking=True)
            l = 0
            optimizer.zero_grad()
            fn_list = []
            for j in range(K):
                model.j = j
                output = model(image)
                original_loss = criterion(output, target)
                loss = original_loss/K
                fn_list.append(loss.item()*K)
                loss.backward(retain_graph=True)
                l = l + loss.item()
            fn_avg = l
            model.scores.grad.data += 1/(K-1)*(fn_list[0] - fn_avg)*getattr(model, 'stored_mask_0') + 1/(K-1)*(fn_list[1] - fn_avg)*getattr(model, 'stored_mask_1')
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3)
            optimizer.step()
            with torch.no_grad():
                total = model.scores.nelement()
                v, itr = solve_v_total(model, total)
                model.scores.sub_(v).clamp_(0, 1)     
        if epoch % 10 == 0:
            print(f'loss: {loss}')
            model.fix_subnet()
            print(model.subnet)

