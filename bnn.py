import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model 
import transformers
from datasets import load_dataset
import torch.nn.functional as F    
from peft.utils.other import transpose
from transformers import TrainerCallback
from admm import Custom_Config, ADMM
from torch import nn
from transformers import Trainer
import pickle
from tqdm import tqdm

import os
os.environ["WANDB_DISABLED"] = "true"

class CastOutputToFloat(nn.Sequential):
    def forward(self, x): 
        return super().forward(x).to(torch.float32)

class ADMMCallback(TrainerCallback):
    def __init__(self, admm):
        self.admm = admm

    def on_train_begin(self, args, state, control, model, **kwargs):
        optimizer = kwargs['optimizer']
        # Add the custom optimizer for the special_param
        special_optimizer = custom_optimizer(model)
        optimizer.add_param_group(special_optimizer.param_groups[0])
    
    def on_step_end(self, args, state, control, model=None, **kwargs):
        clip_mask(model)
            
        
    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        self.update_Z(args, state, control, model, **kwargs)
        self.update_U(args, state, control, model, **kwargs)
    
    def update_X(self):
        print('update_X')
        pass

    def update_Z(self, args, state, control, model=None, **kwargs):
        for name, module in model.named_modules():
            if 'q_proj' in name[-6:] or 'v_proj' in name[-6:]: 
                 # apply mask from pgd
                with torch.no_grad():
                    module.lora_mask.data = get_n_m_sparse_matrix(module.lora_mask.data)
                updated_prun_mask = pgd_prun_mask(module, name, admm)
                module.last_input = None                
                module.last_expected_output = None
                with torch.no_grad():
                    module.prun_mask.data = updated_prun_mask
                if state.global_step % 50 == 0:
                    print('-' * 64)
                    print(f'admm loss: {torch.sum(torch.abs(module.prun_mask - module.lora_mask))}')
                    print(f'prun mask: {module.prun_mask}')
                    print(f'lora mask: {module.lora_mask}')
                    print(f'admm u:    {admm.ADMM_U[name]}')

    def update_U(self, args, state, control, model=None, **kwargs):
        for name, module in model.named_modules():
            if 'q_proj' in name[-6:] or 'v_proj' in name[-6:]: 
                trainer.admm.ADMM_U[name] = trainer.admm.ADMM_U[name].data + module.prun_mask.data - module.lora_mask.data
                # trainer.admm.rho[name] *= 1.3
                # trainer.admm.ADMM_U[name].clamp_(0, 1.0)

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            if is_peft_available() and isinstance(model, PeftModel):
                model_name = unwrap_model(model.base_model)._get_name()
            else:
                model_name = unwrap_model(model)._get_name()
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        # print(f'loss nature {loss}')
        admm_loss = 0
        for name, module in self.model.named_modules():
            if 'q_proj' in name[-6:] or 'v_proj' in name[-6:]:
                admm_loss += self.admm.rho[name] / 2 * (module.lora_mask - self.admm.ADMM_U[name]).norm()
            # if name == 'base_model.model.model.decoder.layers.0.self_attn.v_proj':
                # print(f'loss:{self.admm.ADMM_U[name]}')
        loss += admm_loss
        # print(f'loss admm {admm_loss}')
        return (loss, outputs) if return_outputs else loss
    
def Binarize(tensor,quant_mode='det'):
    if quant_mode=='det':
        # return tensor.sign()
        return (tensor > 0).float()
    if quant_mode=='bin':
        return (tensor>=0).type(type(tensor))*2-1
    else:
        return tensor.add_(1).div_(2).add_(torch.rand(tensor.size()).add(-0.5)).clamp_(0,1).round().mul_(2).add_(-1)
    
from torch.nn.functional import linear, conv2d
class BNNLinear(nn.Linear):

    def __init__(self, *kargs, **kwargs):
        super(BNNLinear, self).__init__(*kargs, **kwargs)
        self.register_buffer('weight_org', self.weight.data.clone())

    def forward(self, input):

        if (input.size(1) != 784) and (input.size(1) != 3072):
            input.data=Binarize(input.data)
            
        self.weight.data=Binarize(self.weight_org)
        out = linear(input , self.weight * self.layer)

        if not self.layer.bias is None:
            out += self.layer.bias.view(1, -1).expand_as(out)

        return out
    
class Custom_Config:
    pass

class ADMM:
    def __init__(self,config):
        self.ADMM_X = {}
        self.ADMM_U = {}
        self.ADMM_Z = {}
        self.rho = {}
        self.model = config.model
        self.prune_ratios = None    #code name -> prune ratio
        self.init(config)
        
    def init(self,config):
        """
        Args:
            config: configuration file that has settings for prune ratios, rhos
        called by ADMM constructor. config should be a .yaml file          

        """          
        self.prune_ratios = config.prune_ratios
        self.rhos = config.rhos
        def pgd_prun_mask_forward(self, input: torch.Tensor) -> torch.Tensor:
            return F.linear(input, transpose(self.prun_mask * self.weight, self.fan_in_fan_out), bias=self.bias)

        self.sparsity_type = config.sparsity_type
        for name, module in self.model.named_modules():
            if 'q_proj' in name[-6:] or 'v_proj' in name[-6:]:
                self.rho[name] = 0.001
                with torch.no_grad():
                    m = get_n_m_sparse_matrix(torch.rand_like(module.prun_mask.data))
                self.ADMM_U[name] = m.data.to(dtype=module.weight.dtype, device = module.prun_mask.device)
                self.ADMM_U[name].requires_grad = False

                self.ADMM_Z[name] = nn.Linear(module.in_features, module.out_features, True).to(torch.float32)
                input = layer_calibrations[name[11:]][0].squeeze(0) 
                output = layer_calibrations[name[11:]][1].squeeze(0)
                input = input.to(torch.float32)  # Convert data to Float
                output = output.to(torch.float32)  # Now output has shape [2048, 768]
                from torch.utils.data import TensorDataset, DataLoader
                dataset = TensorDataset(input, output)
                train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
                self.ADMM_Z[name].train_loader = train_loader

                self.ADMM_Z[name].prun_mask = nn.Parameter(torch.ones_like(module.weight).to(module.weight.dtype))
                self.ADMM_Z[name].eval()
                with torch.no_grad():
                    self.ADMM_Z[name].weight.data = module.weight.data.clone()
                    self.ADMM_Z[name].bias.data = module.bias.data.clone()
                    self.ADMM_Z[name].prun_mask.data = module.prun_mask.data.clone()
                    self.ADMM_Z[name].prun_mask.requires_grad = True
                    self.ADMM_Z[name]._linear = pgd_prun_mask_forward.__get__(self.ADMM_Z[name])


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()            
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def masked_self_forward_linear(self, input: torch.Tensor) -> torch.Tensor:
    if self.prun_masked:
        return F.linear(input, transpose(self.prun_mask * self.weight, self.fan_in_fan_out), bias=self.bias)
    else:
        return F.linear(input, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

def masked_forward_linear(self, x: torch.Tensor) -> torch.Tensor:
    if self.active_adapter not in self.lora_A.keys():
        return self._linear(x)

    previous_dtype = x.dtype
    if self.disable_adapters:
        if (self.r[self.active_adapter] > 0) and self.merged:
            self.unmerge()
        result = self._linear(x)
    elif (self.r[self.active_adapter] == 0) or self.merged:
        result = self._linear(x)
    else:
        lora_A = self.lora_A[self.active_adapter]
        lora_B = self.lora_B[self.active_adapter]
        dropout = self.lora_dropout[self.active_adapter]
        scaling = self.scaling[self.active_adapter]

        result = self._linear(x)
        x = x.to(lora_A.weight.dtype)
        if not self.lora_masked:
            result += lora_B(lora_A(dropout(x))) * scaling
        else:
            tmp = self.lora_mask * (lora_A.weight.transpose(0, 1) @ lora_B.weight.transpose(0, 1))
            result += (dropout(x) @ tmp) * scaling
    result = result.to(previous_dtype)
    return result

def random_binary_tensor(n, m):
    # Calculate the number of ones needed
    num_ones = (n * m) // 2
    
    # Create a tensor with all zeros
    tensor = torch.zeros((n, m), dtype=torch.int)
    
    # Get the indices of the tensor
    indices = torch.arange(n * m)
    
    # Shuffle the indices and select the first num_ones indices to set to 1
    ones_indices = indices[torch.randperm(n * m)][:num_ones]
    
    # Set the selected indices to 1
    tensor.view(-1)[ones_indices] = 1
    
    return tensor

def get_n_m_sparse_matrix(w):
    N, M = 2, 4
    length = w.numel()
    group = int(length / M)
    w_tmp = w.t().detach().abs().reshape(group, M)
    index = torch.argsort(w_tmp, dim=1)[:, :int(M - N)]
    mask = torch.ones(w_tmp.shape, device=w_tmp.device)
    mask = mask.scatter_(dim=1, index=index, value=0).reshape(w.t().shape)
    return mask

def add_masked_layers(model):
    for name, module in model.named_modules():
        if 'q_proj' in name[-6:] or 'v_proj' in name[-6:]:
            row, col = module.weight.shape
            module.lora_mask = nn.Parameter(random_binary_tensor(row, col).to(module.weight.dtype))
            module.lora_mask.requires_grad = False
            module.prun_mask = nn.Parameter(torch.ones_like(module.weight).to(module.weight.dtype))
            module.prun_mask.requires_grad = False
            # Modify forward method
            module.forward = masked_forward_linear.__get__(module)
            module.lora_masked = True 
            module._linear = masked_self_forward_linear.__get__(module)
            module.prun_masked = True 

def clip_mask(model, lower=0.0, upper=1.0):
    params = [param for name, param in model.named_parameters() if 'lora_mask' in name]
    with torch.no_grad():
        for param in params:
            param.clamp_(lower, upper)

def custom_optimizer(model):
    # Access the model's parameters
    params = list(model.named_parameters())
    for name, param in params: 
        if 'lora_mask' in name:
            param.requires_grad = True
    # Identify the special_param
    special_params = [param for name, param in params if 'lora_mask' in name]

    # Define a parameter group with a custom learning rate for the special_param
    param_groups = [
        {'params': special_params, 'lr': 0.01}
    ]

    # Use AdamW for the special_param
    optimizer = transformers.AdamW(param_groups)
    return optimizer 

def sparse_loss(mask):
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            
import copy
def pgd_prun_mask(module, module_name, admm):
    def clip_mask(model, lower=0.0, upper=1.0):
        params = [param for name, param in model.named_parameters() if 'prun_mask' in name]
        with torch.no_grad():
            for param in params:
                param.clamp_(lower, upper)
    model = admm.ADMM_Z[module_name]
    with torch.no_grad():
        lora_mask = module.lora_mask.clone()

    criterion = nn.MSELoss()  
    mask_optimizer = torch.optim.AdamW([model.prun_mask], lr=0.01)
    total_epoch = 1
    device = 'cuda:0'
    for epoch in range(total_epoch):
        for i, (inputs, targets) in enumerate(admm.ADMM_Z[module_name].train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            mask_optimizer.zero_grad()
            outputs = model.forward(inputs)
            loss = criterion(outputs, targets)  # Compute the loss
            l1_reg = admm.rho[module_name] / 2 * (model.prun_mask - lora_mask + admm.ADMM_U[module_name]).norm()
            l2_reg = sparse_loss(model.prun_mask)
            loss += l1_reg
            loss.backward()
            mask_optimizer.step()
            clip_mask(model)
        # if epoch == 0 or epoch == total_epoch - 1:
            # print(f"Epoch {epoch}, Loss: {loss.item()}")
    with torch.no_grad():
        model.prun_mask.data = get_n_m_sparse_matrix(model.prun_mask.data)
    return model.prun_mask.data

from sparsegpt import *
from modelutils import *

def get_opt(model):
    import torch
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import OPTForCausalLM
    model = OPTForCausalLM.from_pretrained(model, torch_dtype='auto', device_map="auto")
    model.seqlen = model.config.max_position_embeddings
    return model

@torch.no_grad()
def get_layer_calibrations(model, dataloader, dev):
    layer_calibrations = {}
    print('Starting ...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers

    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev) 
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev) 
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.to(dev) 
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.cpu()
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']

    print('Ready.')

    for i in range(len(layers)):
        layer = layers[i].to(dev)

        subset = find_layers(layer)
        
        gpts = {}
        for name in subset:
            if (not (args.minlayer <= i < args.maxlayer and args.prune_only in name)) == (not args.invert):
              continue
            gpts[name] = SparseGPT(subset[name])
            if args.wbits < 16:
                gpts[name].quantizer = Quantizer()
                gpts[name].quantizer.configure(
                    args.wbits, perchannel=True, sym=False, mse=False
                )

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp
        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print('Loading ...')
            # sparsity = args.sparsity
            # gpts[name].fasterprune(
            #     sparsity, prunen=args.prunen, prunem=args.prunem, percdamp=args.percdamp, blocksize=args.blocksize
            # )
            name_ = f'model.model.decoder.layers.{i}.{name}'
            layer_calibrations[name_] = (gpts[name].inp1, gpts[name].out1)
            gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    return layer_calibrations

def calc_perplexity(encodings, model, max_length):
    stride = 512
    
    seq_len = encodings.input_ids.size(1)
    nlls = []
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            # loss is calculated using CrossEntropyLoss which averages over valid labels
            # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
            # to the left by 1.
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).mean())
    
    return ppl

if __name__ == '__main__':
    import argparse
    from datautils import *

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model', type=str, 
        default = 'facebook/opt-125m',
        help='OPT model to load; pass `facebook/opt-X`.'
    )
    parser.add_argument(
        '--dataset', type=str, choices=['wikitext2', 'ptb', 'c4'],
        default = 'c4',
        help='Where to extract calibration data from.'
    )
    parser.add_argument(
        '--seed',
        type=int, default=0, help='Seed for sampling the calibration data.'
    )
    parser.add_argument(
        '--nsamples', type=int, default=128,
        help='Number of calibration data samples.'
    )
    parser.add_argument(
        '--percdamp', type=float, default=.01,
        help='Percent of the average Hessian diagonal to use for dampening.'
    )
    parser.add_argument(
        '--sparsity', type=float, default=0.5,
        help='Target sparsity'
    )
    parser.add_argument(
        '--prunen', type=int, default=0,
        help='N for N:M pruning.'
    )
    parser.add_argument(
        '--prunem', type=int, default=0,
        help='M for N:M pruning.'
    )
    parser.add_argument(
        '--blocksize', type=int, default=128,
        help='Blocksize to use for adaptive mask selection.'
    )
    parser.add_argument(
        '--gmp', action='store_true',
        help='Whether to run the GMP baseline.'
    )
    parser.add_argument(
        '--wbits', type=int, default=16,
        help='Whether to quantize as well.'
    )
    parser.add_argument(
        '--minlayer', type=int, default=-1,
        help='Prune all layers with id >= this.'
    )
    parser.add_argument(
        '--maxlayer', type=int, default=1000,
        help='Prune all layers with id < this.'
    )
    parser.add_argument(
        '--prune_only', type=str, default='',
        help='Prune only layers that contain this text.'
    )
    parser.add_argument(
       '--invert', action='store_true', 
       help='Invert subset.'
    )
    parser.add_argument(
       '--save', type=str, default='',
       help='Path to saved model.'
    )
    parser.add_argument(
       '--log_wandb', action='store_true',
       help='Whether to log to wandb.'
    )
    parser.add_argument(
       '--device', type=str, default='cuda:0',
       help='Path to saved model.'
    )
    args = parser.parse_args()

    layer_calibrations = None
    # if (args.sparsity or args.prunen) and not args.gmp:
    #     model = get_opt(args.model)
    #     model.eval()

    #     dataloader, testloader = get_loaders(
    #         args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
    #     )
    #     tick = time.time()
    #     layer_calibrations = get_layer_calibrations(model, dataloader, DEV)
    #     for n, p in model.named_parameters():
    #         print(n, torch.mean((p == 0).float()))
    #         if 'fc2' in n:
    #             break
    #     print(time.time() - tick)
    #     with open('layer_calibrations_opt_125m', 'wb') as f:
    #         pickle.dump(layer_calibrations, f)

    #     del model
    #     del dataloader
    #     del testloader
    #     del layer_calibrations

    with open('layer_calibrations_opt_125m', 'rb') as f:
        layer_calibrations = pickle.load(f)

    model = AutoModelForCausalLM.from_pretrained(
        "facebook/opt-125m", 
        # load_in_8bit=True, 
        device_map='auto',
    )
    
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
    data = load_dataset("databricks/databricks-dolly-15k")
    data = data.map(lambda samples: tokenizer(samples['instruction'], max_length=1024, truncation=True), batched=True)

    for param in model.parameters():
        param.requires_grad = False    
        if param.ndim == 1:
            param.data = param.data.to(torch.float32)

    model.gradient_checkpointing_enable()  
    model.enable_input_require_grads()
    model.lm_head = CastOutputToFloat(model.lm_head)

    config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, config)
    print_trainable_parameters(model)
    add_masked_layers(model)
    print_trainable_parameters(model)

    # Initialize Z, U, and args as per your requirements
    config = Custom_Config()
    config.model = model 
    config.prune_ratios = 0.5
    config.rhos = 0.01
    config.sparsity_type = None
    admm = ADMM(config)
    # Initialize the callback
    admm_callback = ADMMCallback(admm)

    trainer = CustomTrainer(
        model=model, 
        train_dataset=data['train'],
        args=transformers.TrainingArguments(
            per_device_train_batch_size=4, 
            gradient_accumulation_steps=4,
            warmup_steps=100, 
            num_train_epochs=3,      
            # max_steps=10,           
            learning_rate=2e-4, 
            fp16=True,
            logging_steps=10, 
            output_dir='outputs'
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
        callbacks=[admm_callback]  # Pass the custom callback here
    )

    trainer.admm = admm
    model.config.use_cache = False 
    trainer.train(resume_from_checkpoint = False)
    # model.save_pretrained("lora-muwa-125m-opt")

    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = tokenizer.eos_token_id

    model.eval()
    datasets = 'wikitext' 
    ds = load_dataset("wikitext","wikitext-2-raw-v1", split="test")
    encodings = tokenizer("\n\n".join(ds["text"]), return_tensors="pt")
    ppl = calc_perplexity(encodings, model,1024)
    print(f"wikitext perplexity: {ppl}")
