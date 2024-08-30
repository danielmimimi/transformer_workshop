
import logging,coloredlogs
from pathlib import Path


def create_logger(module):

    logger = logging.getLogger(Path(module).name)
    coloredlogs.install(level='DEBUG')
    coloredlogs.install(level='DEBUG', logger=logger)


    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(name)s %(levelname)-8s %(thread)d %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S")
    
    # Disable Warnings from : 
    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
    logging.getLogger('matplotlib.pyplot').setLevel(logging.ERROR)
    logging.getLogger('PIL').setLevel(logging.ERROR)
    
    return logger
