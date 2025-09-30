import os
from datasets import Dataset, load_from_disk, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import json
from src.utils import log_info, clean_phonemes

class PhonemeDatasetBuilder:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        log_info(logger, 'Initializing PhonemeDatasetBuilder')
        self._setup_config_params()
        self.tokenizer = None  # Must be assigned externally before build

    def _setup_config_params(self):
        conf_dataset = self.config['dataset']
        conf_train = self.config['train']

        self.dataset_name = conf_dataset['dataset_name']
        self.output_dir = conf_dataset['output_dir']
        os.makedirs(self.output_dir, exist_ok=True)

        self.max_length = conf_train['max_length']
        self.use_prompt = conf_train['use_prompt']
        self.num_workers = conf_dataset['num_workers']

        filename = self.dataset_name.split('/')[-1]
        self.json_path = os.path.join(
            conf_dataset['dataset_dir'],
            self.dataset_name.split('/')[0],
            filename + '.json'
        )

    def preprocess_batch(self, examples):
        if self.tokenizer is None:
            raise ValueError("Tokenizer not assigned to PhonemeDatasetBuilder")

        # Clean phonemes and prepare input
        if self.use_prompt:
            inputs = [f"Translate phonemes to text:\n{clean_phonemes(p)}\nAnswer:" for p in examples['phonemes']]
        else:
            inputs = [clean_phonemes(p) for p in examples['phonemes']]
        targets = examples['text']

        # Batched tokenization
        tokenized_inputs = self.tokenizer(
            inputs,
            truncation=True,
            max_length=self.max_length,
            padding="max_length"
        )
        tokenized_targets = self.tokenizer(
            targets,
            truncation=True,
            max_length=self.max_length,
            padding="max_length"
        )

        # Replace pad token with -100 in labels
        tokenized_inputs["labels"] = [
            [-100 if t == self.tokenizer.pad_token_id else t for t in label_seq]
            for label_seq in tokenized_targets["input_ids"]
        ]
        return tokenized_inputs

    def build(self, test_size=0.1):
        cache_path = os.path.join(
            self.output_dir, 
            f"{self.dataset_name.split('/')[0]}{'_prompt' if self.use_prompt else ''}"
        )
        if os.path.exists(cache_path):
            self.logger.info(f"Loading preprocessed dataset from {cache_path}")
            return load_from_disk(cache_path)

        self.logger.info("Building dataset from raw JSON")
        # Use HF Dataset from JSON for better memory efficiency
        with open(self.json_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        dataset = Dataset.from_list(raw_data)

        # Tokenize in parallel with batching
        processed = dataset.map(
            self.preprocess_batch,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing dataset",
            num_proc=self.num_workers
        )

        # Train/test split
        train_indices, test_indices = train_test_split(range(len(processed)), test_size=test_size, random_state=42)
        dataset_dict = DatasetDict({
            "train": processed.select(train_indices),
            "test": processed.select(test_indices)
        })

        dataset_dict.save_to_disk(cache_path)
        self.logger.info(f"Processed dataset saved to {cache_path}")
        return dataset_dict
