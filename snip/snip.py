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
    mask_optimizer = torch.optim.AdamW([net.weight], lr=0.001)
    rho = 0.01  # You can adjust tsshis value to change the strength of the regularization
    total_epoch = 50
    total_param = net.weight.shape[0] * net.weight.shape[1]
    for epoch in range(total_epoch):
        for i, (inputs, targets) in enumerate(train_dataloader):
            inputs, targets = inputs.to(device), targets.to(device)

            # step 2: calculate loss and update the mask values
            mask_optimizer.zero_grad()
            net.weight.data.copy_(net.weight_org)
            outputs = net.forward(inputs)
            loss = criterion(outputs, targets)  # Compute the loss
            print(net.weight)
            # l1_reg = rho / 2 * (torch.sum(torch.abs(net.weight)) - total_param).norm()
            # loss += l1_reg
            loss.backward()
            mask_optimizer.step()
            clip_mask(net)
            net.weight_org.data.copy_(net.weight.data.clamp_(0,1))
        if epoch == 0 or epoch == total_epoch - 1:
            print(f"Epoch {epoch}, Loss: {loss.item()}")
    
    # num_params_to_keep = int(net.weight_mask.shape[0] * net.weight_mask.shape[1] * keep_ratio)
    # threshold, _ = torch.topk(torch.flatten(net.weight_mask), num_params_to_keep, sorted=True)
    # acceptable_score = threshold[-1]
    # keep_masks = net.weight_mask > acceptable_score
    # return keep_masks

import torch.optim as optim
def VRPEG(model, keep_ratio, train_dataloader, device):
   # Define the optimizer, you can use any optimizer of your choice
    def solve_v_total(model, total):
        k = total * 0.5
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
            # sparsity_constraint = lambda_sparsity * torch.abs(torch.sum(model.weight) - 0.5 * model.weight.numel())
            # loss = loss_mse + sparsity_constraint
            
            # Backward pass and optimization
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

# import torch.optim as optim
# def VRPEG(model, keep_ratio, train_dataloader, device):
#     epochs = 85
#     final_temp = 200
#     iters_per_reset = epochs-1
#     lr = 0.1
#     decay = 0.0001
#     rounds = 3
#     rewind_epoch = 2
#     lmbda = 1e-8

#     temp_increase = final_temp**(1./iters_per_reset)

#     trainable_params = filter(lambda p: p.requires_grad, model.parameters())
#     num_params = sum([p.numel() for p in trainable_params])
#     print("Total number of parameters: {}".format(num_params))

#     # weight_params = map(lambda a: a[1], filter(lambda p: p[1].requires_grad and 'mask' not in p[0], model.named_parameters()))
#     # mask_params = map(lambda a: a[1], filter(lambda p: p[1].requires_grad and 'mask' in p[0], model.named_parameters()))
#     weight_params = [model.weight, model.bias]
#     mask_params = [model.mask_weight]

#     model.ticket = False
#     weight_optim = optim.SGD(weight_params, lr=lr, momentum=0.9, nesterov=True, weight_decay=decay)
#     mask_optim = optim.SGD(mask_params, lr=lr, momentum=0.9, nesterov=True)
#     optimizers = [weight_optim, mask_optim]
#     criterion = nn.MSELoss()  # Mean Squared Error Loss for regression
#     for outer_round in range(rounds):
#         model.temp = 1
#         print('--------- Round {} -----------'.format(outer_round))
#         for epoch in range(epochs):
#             # print('\t--------- Epoch {} -----------'.format(epoch))
#             model.train()
#             if epoch > 0: model.temp *= temp_increase  
#             if outer_round == 0 and epoch == rewind_epoch: model.checkpoint()
#             for optimizer in optimizers: adjust_learning_rate(optimizer, epoch)

#             for batch_idx, (data, target) in enumerate(train_dataloader):
#                 data, target = data.to(device), target.to(device)
#                 for optimizer in optimizers: optimizer.zero_grad()
#                 output = model(data)
#                 masks = [model.mask_weight]
#                 entries_sum = sum(m.sum() for m in masks)
#                 loss = criterion(output, target) + lmbda * entries_sum
#                 loss.backward()
#                 for optimizer in optimizers: optimizer.step()
#         if outer_round != rounds-1: model.prune(model.temp)

#     print('--------- Training final ticket -----------')
#     optimizers = [optim.SGD(weight_params, lr=lr, momentum=0.9, nesterov=True, weight_decay=decay)]
#     model.ticket = True
#     model.rewind_weights()


#     for epoch in range(epochs):
#         # print('\t--------- Epoch {} -----------'.format(epoch))
#         model.train()
#         if epoch > 0: model.temp *= temp_increase  
#         if outer_round == 0 and epoch == rewind_epoch: model.checkpoint()
#         for optimizer in optimizers: adjust_learning_rate(optimizer, epoch)

#         for batch_idx, (data, target) in enumerate(train_dataloader):
#             data, target = data.to(device), target.to(device)
#             for optimizer in optimizers: optimizer.zero_grad()
#             output = model(data)
#             masks = [model.mask_weight]
#             entries_sum = sum(m.sum() for m in masks)
#             loss = criterion(output, target) + lmbda * entries_sum
#             loss.backward()
#             for optimizer in optimizers: optimizer.step()
#     return model.mask