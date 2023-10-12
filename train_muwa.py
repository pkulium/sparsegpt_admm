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
        self.update_Z(args, state, control, model, **kwargs)
        self.update_U(args, state, control, model, **kwargs)
        
    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        print('update_X')
        pass
    
    def update_X(self):
        print('update_X')
        pass

    def update_Z(self, args, state, control, model=None, **kwargs):
        for name, module in model.named_modules():
            if 'q_proj' in name[-6:] or 'v_proj' in name[-6:]: 
                 # apply mask from pgd
                updated_prun_mask = pgd_prun_mask(module, name, admm)
                module.last_input = None                
                module.last_expected_output = None
                if state.global_step % 10 == 0:
                    print(f'prun mask: {module.prun_mask}')
                    print(f'lora mask: {module.lora_mask}')
                    print(f'admm u:    {admm.ADMM_U[name]}')

    def update_U(self, args, state, control, model=None, **kwargs):
        for name, module in model.named_modules():
            if 'q_proj' in name[-6:] or 'v_proj' in name[-6:]: 
                trainer.admm.ADMM_U[name] = trainer.admm.ADMM_U[name].data + module.prun_mask.data - module.lora_mask.data

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
                admm_loss += self.admm.rho[name] / 2 * (module.lora_mask.data - self.admm.ADMM_U[name]).norm()
            # if name == 'base_model.model.model.decoder.layers.0.self_attn.v_proj':
                # print(f'loss:{self.admm.ADMM_U[name]}')
        loss += admm_loss
        # print(f'loss admm {admm_loss}')
        return (loss, outputs) if return_outputs else loss

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
                _, m = get_n_m_sparse_matrix(torch.rand_like(module.prun_mask))
                self.ADMM_U[name] = m.data.to(dtype=module.weight.dtype, device = module.prun_mask.device)
                self.ADMM_U[name].requires_grad = False
                self.ADMM_Z[name] = nn.Linear(module.in_features, module.out_features, True)
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
        with torch.no_grad():
            self.last_input = input
            self.last_expected_output = F.linear(input, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
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
    mask = mask.scatter_(dim=1, index=index, value=0).reshape(w.t().shape).t()
    return w * mask, mask

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
            w, m = get_n_m_sparse_matrix(param)
            param.data = m.to(param.dtype)

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
import copy
def pgd_prun_mask(module, module_name, admm):
    def clip_mask(model, lower=0.0, upper=1.0):
        params = [param for name, param in model.named_parameters() if 'prun_mask' in name]
        with torch.no_grad():
            for param in params:
                param.clamp_(lower, upper)
    model = admm.ADMM_Z[module_name]
    with torch.no_grad():
        inputs = module.last_input.clone()
        module.last_input = None
        inputs = inputs.to(model.weight.dtype)
        targets = module.last_expected_output.clone()
        module.last_expected_output = None
        targets = targets.to(model.weight.dtype)
        lora_mask = module.lora_mask.clone()

    criterion = nn.MSELoss()  
    mask_optimizer = torch.optim.AdamW([model.prun_mask], lr=0.01)
    rho = 0.01  # You can adjust tsshis value to change the strength of the regularization
    total_epoch = 1
    device = 'cuda:0'
    for epoch in range(total_epoch):
        # for i, (inputs, targets) in enumerate(train_loader):
        # inputs, targets = inputs.to(device), targets.to(device)
        # step 2: calculate loss and update the mask values
        mask_optimizer.zero_grad()
        outputs = model.forward(inputs)
        loss = criterion(outputs, targets)  # Compute the loss
        l1_reg = admm.rho[module_name] / 2 * (model.prun_mask - lora_mask + admm.ADMM_U[module_name]).norm()
        loss += l1_reg
        loss.backward()
        mask_optimizer.step()
        clip_mask(model)
        # if epoch == 0 or epoch == total_epoch - 1:
            # print(f"Epoch {epoch}, Loss: {loss.item()}")
    _, model.prun_mask.data = get_n_m_sparse_matrix(model.prun_mask.data)
    return model.prun_mask.data

if __name__ == '__main__':
    import argparse
    from datautils import *

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model', type=str, 
        default = 'facebook/opt-1.3b',
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
        '--sparsity', type=float, default=0,
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
            num_train_epochs=1,      
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
    # model.save_pretrained("lora-muwa-1.3b-opt")