import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
import types

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


def PGD(net, keep_ratio, train_dataloader, device):

    # Let's create a fresh copy of the network so that we're not worried about
    # affecting the actual training-phase
    # net = copy.deepcopy(net)

    # Monkey-patch the Linear and Conv2d layer to learn the multiplicative mask
    # instead of the weights
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            W_metric = torch.abs(layer.weight.data)
            thresh = torch.sort(W_metric.flatten().cuda())[0][int(layer.weight.numel()*keep_ratio)].cpu()
            W_mask = (W_metric>=thresh)
            layer.weight_mask = nn.Parameter(W_mask.to(layer.weight.data.dtype))
            # layer.weight_mask = nn.Parameter(torch.zeros_like(layer.weight))
            # nn.init.xavier_normal_(layer.weight_mask)
            layer.weight.requires_grad = False
            

        # Override the forward methods:
        if isinstance(layer, nn.Conv2d):
            layer.forward = types.MethodType(snip_forward_conv2d, layer)

        if isinstance(layer, nn.Linear):
            layer.forward = types.MethodType(snip_forward_linear, layer)

    criterion = nn.MSELoss()  # Mean Squared Error Loss for regression
    mask_optimizer = torch.optim.SGD([net.weight_mask], lr=0.001, momentum=0.9)
    for epoch in range(10):
        for i, (inputs, targets) in enumerate(train_dataloader):
            inputs, targets = inputs.to(device), targets.to(device)

            # step 2: calculate loss and update the mask values
            mask_optimizer.zero_grad()
            outputs = net.forward(inputs)
            loss = criterion(outputs, targets)  # Compute the loss
            loss.backward()
            mask_optimizer.step()
            clip_mask(net)
    
    # num_params_to_keep = int(net.weight_mask.shape[0] * net.weight_mask.shape[1] * keep_ratio)
    # threshold, _ = torch.topk(torch.flatten(net.weight_mask), num_params_to_keep, sorted=True)
    # acceptable_score = threshold[-1]
    # keep_masks = net.weight_mask > acceptable_score
    # return keep_masks

    thresh = torch.sort(net.weight_mask.flatten().cuda())[0][int(layer.weight.numel()*keep_ratio)].cpu()
    W_mask = W_metric>=thresh
    print(f'shape {int(layer.weight.numel())}: {torch.sum(W_mask)}')
    return W_mask