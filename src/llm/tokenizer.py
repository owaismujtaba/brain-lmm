import os
import json
from transformers import AutoTokenizer
from src.utils import log_info, clean_phonemes
from tqdm import tqdm
import pdb

class TokenizerModule:
    def __init__(self, config, logger=None):
        self.logger = logger
        self.config = config
        log_info(logger, "Initializing TokenizerModule")
        self._setup_conf_params()
        self.tokenizer = None

    def _setup_conf_params(self):
        self.logger.info('Setting Tokenizer configuration parameters')
        dataset_conf = self.config['dataset']
        self.dataset_name = dataset_conf['dataset_name']
        self.dataset_dir = dataset_conf['dataset_dir']
        self.data_path = os.path.join(self.dataset_dir, self.dataset_name + ".json")
        
        # Directory to save tokenizer
        self.tokenizer_dir = os.path.join(self.dataset_dir, "tokenizer")
        os.makedirs(self.tokenizer_dir, exist_ok=True)

        # Tokenizer path based on model name
        tokenizer_name = self.config['train']['model_name'].split('/')[0]
        self.tokenizer_path = os.path.join(self.tokenizer_dir, tokenizer_name)
        os.makedirs(self.tokenizer_path, exist_ok=True)

    def build_tokenizer(self):
        self.logger.info("Building LLaMA2 tokenizer with phonemes")

        # Load base LLaMA2 tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.config['train']['model_name'], use_fast=True)

        # Collect phonemes from dataset
        with open(self.data_path, "r") as f:
            raw_data = json.load(f)

        phoneme_set = set()
        for item in tqdm(raw_data, desc="Collecting phonemes"):
            phonemes = clean_phonemes(item["phonemes"])
            phoneme_set.update(phonemes.split())

        # Add phonemes as additional special tokens
        phoneme_tokens = {"additional_special_tokens": list(phoneme_set)}
        tokenizer.add_special_tokens(phoneme_tokens)
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        # Save tokenizer
        tokenizer.save_pretrained(self.tokenizer_path)

        # Confirm saved files
        saved_files = os.listdir(self.tokenizer_path)
        self.logger.info(f"Tokenizer saved at {self.tokenizer_path}, files: {saved_files}")

        self.tokenizer = tokenizer
        self.logger.info(f"LLaMA2 tokenizer extended with {len(phoneme_set)} phonemes and saved.")
        return self.tokenizer

    def load_tokenizer(self):
        
        config_file = os.path.join(self.tokenizer_path, "tokenizer_config.json")

        if not os.path.exists(config_file):
            self.logger.info("Tokenizer not found, building it first.")
            return self.build_tokenizer()
        self.logger.info(f"Tokenizer found at {self.tokenizer_path}, Loading.")
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, use_fast=True)
        self.logger.info("Extended LLaMA2 tokenizer loaded from disk.")
        return self.tokenizer
