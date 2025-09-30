import os
import json
from pathlib import Path
from datasets import load_dataset
from g2p_en import G2p
from tqdm import tqdm
import nltk

from src.utils import log_info
import pdb

class SentenceToPhonemes:
    def __init__(self, config, logger):
        """
        config : dict : configuration dictionary loaded from YAML
        logger : logging.Logger : logger instance
        """
        self.logger = logger
        self.config = config
        log_info(logger, 'Initializing Sentence2Phonemes')
        self._setup_conf_params()
        self.g2p = G2p()

    def _setup_conf_params(self):
        self.logger.info('Setting configuration parameters')
        conf_dataset = self.config['dataset']
        self.dataset_name = conf_dataset['dataset_name']
        self.output_dir = Path(conf_dataset['dataset_dir'])
        self.output_file = self.output_dir / f"{self.dataset_name}.json"
        os.makedirs(self.output_dir, exist_ok=True)

        # Correct NLTK download
        nltk.download("averaged_perceptron_tagger")

    def sentence_to_phonemes(self, sentence: str):
        """
        Converts a single sentence to a list of phonemes with '|' as word separators
        """
        phoneme_sentence = []
        for word in sentence.lower().split():
            try:
                phonemes = [p for p in self.g2p(word) if p != ' ']
                phoneme_sentence.append(" ".join(phonemes))
                phoneme_sentence.append(" | ")
            except Exception as e:
                self.logger.error(f"Failed to convert sentence to phonemes: '{sentence}' - {e}")
                return None  # Skip the sentence if any word fails
                
        if phoneme_sentence and phoneme_sentence[-1] == " | ":
            phoneme_sentence = phoneme_sentence[:-1]  # Remove trailing '|'
            
        return phoneme_sentence

    def create_dataset(self, batch_size: int = 1000):
        self.logger.info(f'Creating dataset from {self.dataset_name}')
        dataset = load_dataset(self.dataset_name, split="train")
    
        total_count = 0
        items_batch = []
        
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        with open(self.output_file, "w", encoding="utf-8") as f:
            f.write("[\n")
            for sentence in tqdm(dataset["text"], desc="Processing sentences"):
                phonemes_list = self.sentence_to_phonemes(sentence)
                if phonemes_list is None:
                    continue
    
                phonemes_str = " ".join(phonemes_list)
                item = {
                    "text": sentence.lower(),
                    "phonemes": phonemes_str
                }
                items_batch.append(item)
                total_count += 1
    
                # Write batch to file
                if len(items_batch) == batch_size:
                    for idx, batch_item in enumerate(items_batch):
                        if total_count > len(items_batch) or idx > 0:
                            f.write(",\n")
                        json.dump(batch_item, f, ensure_ascii=False, indent=2)
                    items_batch = []
    
            # Write remaining items if any
            if items_batch:
                for idx, batch_item in enumerate(items_batch):
                    if total_count > len(items_batch) or idx > 0:
                        f.write(",\n")
                    json.dump(batch_item, f, ensure_ascii=False, indent=2)
    
            f.write("\n]")
    
        self.logger.info(f"Processed {total_count} sentences and wrote to {self.output_file}")
