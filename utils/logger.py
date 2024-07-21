'''
使用logging
'''
import logging
import logging.config

def get_logger(config):
    logging.config.fileConfig(config.get("base", "logger_config_path"))
    logger = logging.getLogger()
    return logger