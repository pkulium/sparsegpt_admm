import os
import torch
import torch.nn as nn
import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM



model = AutoModelForCausalLM.from_pretrained(
    "facebook/opt-125m", 
    load_in_8bit=True, 
    device_map='auto',
)

tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")

 
import transformers
from datasets import load_dataset
data = load_dataset("databricks/databricks-dolly-15k")
data = data.map(lambda samples: tokenizer(samples['instruction'], max_length=1024, truncation=True), batched=True)

 
for param in model.parameters():
  param.requires_grad = False    if param.ndim == 1:
        param.data = param.data.to(torch.float32)

model.gradient_checkpointing_enable()  model.enable_input_require_grads()

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

trainer = transformers.Trainer(
    model=model, 
    train_dataset=data['train'],
    args=transformers.TrainingArguments(
        per_device_train_batch_size=4, 
        gradient_accumulation_steps=4,
        warmup_steps=100, 
        num_train_epochs=2,                 learning_rate=2e-4, 
        fp16=True,
        logging_steps=10, 
        output_dir='outputs'
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
)
model.config.use_cache = False 

trainer.train(resume_from_checkpoint = False)












from transformers import TrainerCallback

class ADMMCallback(TrainerCallback):
    def __init__(self, Z, U, admm_args):
        self.Z = Z
        self.U = U
        self.admm_args = admm_args
    
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
        X = update_X(model)
        self.Z = update_Z_l1(X, self.U, self.args) if self.args.l1 else update_Z(X, self.U, self.args)
        self.U = update_U(self.U, X, self.Z)
    
    def compute_loss(self, model, inputs, return_outputs=False):
        # Compute the original loss
        outputs = model(**inputs)
        original_loss = outputs.loss if hasattr(outputs, "loss") else None
        
        # Compute the admm_loss
        output = outputs.logits if hasattr(outputs, "logits") else None
        target = inputs["labels"] if "labels" in inputs else None
        admm_loss_value = admm_loss(self.admm_args, self.args.device, model, self.Z, self.U, output, target)
        
        # Combine the original loss and the admm_loss
        combined_loss = original_loss + admm_loss_value
        
        return (combined_loss, outputs) if return_outputs else combined_loss


# Initialize Z, U, and args as per your requirements
Z, U, args = initialize_Z_and_U(), initialize_U(), initialize_args()

# Initialize the callback
admm_callback = ADMMCallback(Z, U, args)

# Initialize the Trainer with your custom callback
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    callbacks=[admm_callback]  # Pass the custom callback here
)

# Start training
trainer.train()
