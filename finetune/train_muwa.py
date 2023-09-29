import os
import torch
import torch.nn as nn
import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "facebook/opt-125m", 
    # load_in_8bit=True, 
    device_map='auto',
)

tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")

 
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
    return F.linear(input, transpose(self.prun_mask * self.weight, self.fan_in_fan_out), bias=self.bias)

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
        # result += lora_B(lora_A(dropout(x))) * scaling
        tmp = self.lora_mask * (lora_A.weight.transpose(0, 1) @ lora_B.weight.transpose(0, 1))
        result += (dropout(x) @ tmp) * scaling

    result = result.to(previous_dtype)
    return result

def add_masked_layers(model):
    for name, module in model.named_modules():
        if 'q_proj' in name[-6:] or 'v_proj' in name[-6:]:
            module.lora_mask = torch.ones_like(module.weight)
            module.prun_mask = torch.ones_like(module.weight)
            # Modify forward method
            module.forward = masked_forward_linear.__get__(module)
            module._linear = masked_self_forward_linear.__get__(module)

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


from transformers import TrainerCallback
class ADMMCallback(TrainerCallback):
    def __init__(self, admm):
        self.admm = admm
    
    def on_step_end(self, args, state, control, model=None, **kwargs):
        # This will be executed at the end of each training step
        # You can perform optimizer step, zero_grad, etc. here if needed
        # But usually, this is handled by the Trainer itself
        
        # If you need to access or modify model parameters, optimizer, etc.
        # You can access them using the `model` and `trainer` objects
        # For example: model.parameters(), trainer.optimizer, etc.
        pass
        
    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        # This will be executed at the end of each epoch
        # You can perform your X, Z, U updates here
        self.update_X(model)
        self.update_Z(X, self.U, self.args)
        self.update_U(self.U, X, self.Z)
    
    def update_X(self, model):
        pass

    def update_Z(self, X, U, self.args):
        pass

    def update_U(self, U, X, Z):
        pass

# # Initialize Z, U, and args as per your requirements
from admm import Custom_Config, ADMM
config = Custom_Config()
config.model = model 
config.prune_ratios = 0.5
config.rhos = 0.01
config.sparsity_type = None
admm = ADMM(config)
print(admm)
Initialize the callback
admm_callback = ADMMCallback(ADMM)

trainer = transformers.Trainer(
    model=model, 
    train_dataset=data['train'],
    args=transformers.TrainingArguments(
        per_device_train_batch_size=4, 
        gradient_accumulation_steps=4,
        warmup_steps=100, 
        num_train_epochs=1,                 
        learning_rate=2e-4, 
        fp16=True,
        logging_steps=10, 
        output_dir='outputs'
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    callbacks=[admm_callback]  # Pass the custom callback here
)
model.config.use_cache = False 
trainer.train(resume_from_checkpoint = False)