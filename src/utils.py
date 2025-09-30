
import os
import yaml
import re
import pdb

def clean_phonemes(phoneme_seq: str) -> str:
    """
    Remove stress markers (0, 1, 2) from phoneme sequences.
    Example: 'AH0' -> 'AH', 'IH1' -> 'IH'
    """
    
    return re.sub(r"[012]", "", phoneme_seq)

def load_yaml_config(config_path):
    """
    Loads a YAML configuration file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: Parsed configuration as a dictionary.
    """
    

    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"The configuration file {config_path} does not exist.")

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    return config


def log_info(logger, text):
    """
    logs text with heading
    """
    logger.info('*'*60)
    logger.info(text)
    logger.info('*'*60)
