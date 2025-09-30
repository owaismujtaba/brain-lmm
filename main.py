import os
from pathlib import Path

from src.utils import load_yaml_config, log_info
from src.logger.log import setup_logger

from src.dataset.g2p_conversion import SentenceToPhonemes




CUR_DIR = os.getcwd()
CONFIG_FILEPATH = Path(CUR_DIR, 'config.yaml')

import pdb

if __name__ == "__main__":
    config = load_yaml_config(CONFIG_FILEPATH)
    logger = setup_logger(config)
    
    
    from src.llm.tokenizer import TokenizerModule
    from src.llm.trainer import Trainer
    tokenizer = TokenizerModule(config, logger)
    tokenizer = tokenizer.load_tokenizer()
    
    trainer = Trainer(config, logger, tokenizer=tokenizer)
    trainer.train()
    #
    '''
    log_info(logger, "Configuration Loaded:")
    for key, value in config.items():
        if key=='run':
            logger.info(f"{key}: {value}")

    if config['mode'] == 'dataset':
        data_creator = SentenceToPhonemes(
            config = config,
            logger = logger
        )

        data_creator.create_dataset()

    if config['mode'] == 'train':
        pass
    '''
        

    


    

    