import os
import torch
import torch.nn as nn
import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

import os
os.environ["WANDB_DISABLED"] = "true"

model = AutoModelForCausalLM.from_pretrained(
    "facebook/opt-1.3b", 
    # load_in_8bit=True, 
    cache_dir=cache_dir = 'llm_weights',
    device_map='auto',
)
model.seqlen = model.config.max_position_embeddings 
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")

 
import transformers
from datasets import load_dataset
data = load_dataset("databricks/databricks-dolly-15k")
data = data.map(lambda samples: tokenizer(samples['instruction'], max_length=1024, truncation=True), batched=True)

for param in model.parameters():
    param.requires_grad = False    
    if param.ndim == 1:
        param.data = param.data.to(torch.float32)

model.gradient_checkpointing_enable()  
model.enable_input_require_grads()

class CastOutputToFloat(nn.Sequential):
  def forward(self, x): return super().forward(x).to(torch.float32)
model.lm_head = CastOutputToFloat(model.lm_head)
# base_model.model.model.decoder.layers.0.self_attn.v_proj
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

import torch.nn.functional as F    
from peft.utils.other import transpose
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
    mask = mask.scatter_(dim=1, index=index, value=0).reshape(w.t().shape).t()
    return w * mask, mask

from lib.prune import prune_sparsegpt

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

from peft import LoraConfig, get_peft_model 

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

# trainer = transformers.Trainer(
#     model=model, 
#     train_dataset=data['train'],
#     args=transformers.TrainingArguments(
#         per_device_train_batch_size=4, 
#         gradient_accumulation_steps=4,
#         warmup_steps=100, 
#         num_train_epochs=1,                 
#         learning_rate=2e-4, 
#         fp16=True,
#         logging_steps=10, 
#         output_dir='outputs'
#     ),
#     data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
# )
# model.config.use_cache = False 
# trainer.train(resume_from_checkpoint = False)

def clip_mask(model, lower=0.0, upper=1.0):
    params = [param for name, param in model.named_parameters() if 'lora_mask' in name]
    with torch.no_grad():
        for param in params:
            param.clamp_(lower, upper)
            # w, m = get_n_m_sparse_matrix(param)
            # param.data = m.to(param.dtype)

from transformers import TrainerCallback
class ADMMCallback(TrainerCallback):
    def __init__(self):
        pass

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
                module.lora_masked = False 
                module.prun_masked = False
        config = Custom_Config()
        config.nsamples = 10
        config.seed = 10
        config.nsamples = 10
        config.sparsity_ratio = 10
        prune_sparsegpt(config, model, tokenizer, dev='cuda:0', prune_n=None, prune_m=None)

    def update_U(self, args, state, control, model=None, **kwargs):
        for name, module in model.named_modules():
            if 'q_proj' in name[-6:] or 'v_proj' in name[-6:]: 
                trainer.admm.ADMM_U[name] = trainer.admm.ADMM_U[name].data.clone() + module.prun_mask.data.clone() - module.lora_mask.data.clone()

# # Initialize Z, U, and args as per your requirements
from admm import Custom_Config, ADMM
config = Custom_Config()
config.model = model 
config.prune_ratios = 0.5
config.rhos = 0.001
config.sparsity_type = None
admm = ADMM(config)
# Initialize the callback
admm_callback = ADMMCallback()


from torch import nn
from transformers import Trainer

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

trainer = CustomTrainer(
    model=model, 
    train_dataset=data['train'],
    args=transformers.TrainingArguments(
        per_device_train_batch_size=4, 
        gradient_accumulation_steps=4,
        warmup_steps=100, 
        # num_train_epochs=3,      
        max_steps=20,           
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
model.save_pretrained("lora-muwa-1.3b-opt")