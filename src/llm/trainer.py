import torch
import os
from transformers import (
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer as HFTrainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model
from src.dataset.dataset import PhonemeDatasetBuilder
from transformers import PreTrainedTokenizerFast
from transformers import DataCollatorWithPadding
from src.utils import log_info

import pdb
class Trainer:
    def __init__(self, config, logger, tokenizer: PreTrainedTokenizerFast):
        self.config = config
        self.logger = logger
        self.tokenizer = tokenizer
        log_info(logger, 'Initializing Trainer')
        self._setup_conf_params()

        # Dataset
        builder = PhonemeDatasetBuilder(self.config, self.logger)
        builder.tokenizer = self.tokenizer
        
        self.dataset = builder.build()

        # Model with quantization
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=False,   # use load_in_4bit=True if you want 4-bit quantization
            llm_int8_enable_fp32_cpu_offload=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",     # ✅ fix: string not set
            torch_dtype="auto"     # let HF decide best dtype
        )
        
        self.model.resize_token_embeddings(len(self.tokenizer))
        # LoRA adapter
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.model = get_peft_model(self.model, lora_config)

    def _setup_conf_params(self):
        conf_train = self.config['train']
        self.model_name = conf_train['model_name']
        self.output_dir = os.path.join(
            self.config['dataset']['output_dir'], 
            'models', 
            self.model_name.split('/')[1]
        )
        self.batch_size = conf_train['batch_size']
        self.epochs = conf_train['epochs']
        self.grad_accum = conf_train['grad_accum']
        self.lr = conf_train['lr']
        self.eval_steps = conf_train['eval_steps']
        self.save_steps = conf_train['save_steps']
        self.logging_steps = conf_train['logging_steps']
        
        os.makedirs(self.output_dir, exist_ok=True)

    def train(self):
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=self.grad_accum,
            learning_rate=self.lr,
            num_train_epochs=self.epochs,
            logging_steps=self.logging_steps,
            eval_strategy="steps",   
            eval_steps=self.eval_steps,                
            save_strategy="steps",
            save_steps=self.save_steps,
            save_total_limit=5,
            fp16=True,
            optim="adamw_torch",
            report_to="none",
            load_best_model_at_end=True
        )

        data_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer,
            padding=True,
            return_tensors="pt"
        )

        
        
        hf_trainer = HFTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset['train'],
            eval_dataset=self.dataset['test'].select(range(50)),
            data_collator=data_collator
            #tokenizer=self.tokenizer,
            #label_names='labels'
        )

        # ✅ Resume training if checkpoint exists
        last_checkpoint = None
        if os.path.isdir(self.output_dir):
            checkpoints = [os.path.join(self.output_dir, d) for d in os.listdir(self.output_dir) if d.startswith("checkpoint-")]
            if checkpoints:
                last_checkpoint = max(checkpoints, key=os.path.getmtime)
                self.logger.info(f"Resuming training from checkpoint: {last_checkpoint}")

        hf_trainer.train(resume_from_checkpoint=last_checkpoint)

        # === Merge LoRA into base model before saving ===
        self.logger.info("Merging LoRA adapters into base model...")
        merged_model = PeftModel.from_pretrained(self.model, self.output_dir).merge_and_unload()
        
        # Save the merged model and tokenizer
        merged_model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        
        self.logger.info(f"Merged model saved at {self.output_dir}")
