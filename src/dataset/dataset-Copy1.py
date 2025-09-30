import os
import json
from datasets import Dataset, load_from_disk, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer as HFTrainer
from peft import LoraConfig, get_peft_model
import torch
from src.utils import log_info, clean_phonemes

# ---------------------------
# PHONEME DATASET BUILDER
# ---------------------------
class PhonemeDatasetBuilder:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        log_info(logger, 'Initializing PhonemeDatasetBuilder')
        self._setup_config_params()
        self.tokenizer = None  # must be assigned externally before build

    def _setup_config_params(self):
        self.logger.info('Setting PhonemeDatasetBuilder configuration parameters')
        conf_dataset = self.config['dataset']
        conf_train = self.config['train']

        self.dataset_name = conf_dataset['dataset_name']
        self.output_dir = conf_dataset['output_dir']
        os.makedirs(self.output_dir, exist_ok=True)

        self.max_length = conf_train['max_length']
        self.use_prompt = conf_train['use_prompt']

        filename = self.dataset_name.split('/')[-1]
        self.json_path = os.path.join(
            conf_dataset['dataset_dir'],
            self.dataset_name.split('/')[0],
            filename + '.json'
        )

    def preprocess(self, example):
        if self.tokenizer is None:
            raise ValueError("Tokenizer not assigned to PhonemeDatasetBuilder")

        phonemes = clean_phonemes(example['phonemes'])
        target_text = example['text']

        input_text = f"Translate phonemes to text:\n{phonemes}\nAnswer:" if self.use_prompt else phonemes

        tokenized_input = self.tokenizer(
            input_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length"
        )
        tokenized_target = self.tokenizer(
            target_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length"
        )

        labels = [
            -100 if token_id == self.tokenizer.pad_token_id else token_id
            for token_id in tokenized_target["input_ids"]
        ]
        tokenized_input["labels"] = labels
        return tokenized_input

    def build(self, test_size=0.1):
        if not self.use_prompt:
            cache_path = os.path.join(self.output_dir, self.dataset_name.split('/')[0])
        else:
            name = self.dataset_name.split('/')[0]
            cache_path = os.path.join(self.output_dir, f'{name}_prompt')
        if os.path.exists(cache_path):
            self.logger.info(f"Loading preprocessed dataset from {cache_path}")
            dataset = load_from_disk(cache_path)
            return dataset

        with open(self.json_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        self.logger.info("Building dataset from raw JSON")
        data = Dataset.from_list(raw_data)

        # Tokenize
        processed = data.map(
            self.preprocess,
            remove_columns=data.column_names,
            desc="Tokenizing dataset"
        )

        # Train/Test split
        train_indices, test_indices = train_test_split(range(len(processed)), test_size=test_size, random_state=42)
        dataset_dict = DatasetDict({
            "train": processed.select(train_indices),
            "test": processed.select(test_indices)
        })

        # Save processed dataset
        dataset_dict.save_to_disk(cache_path)
        self.logger.info(f"Processed dataset saved to {cache_path}")
        return dataset_dict