import os
import configparser

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'config.ini')

class AppConfig:
    def __init__(self):
        self.datapath = ''
        cfg = configparser.ConfigParser()
        cfg.read(os.path.abspath(_CONFIG_PATH))
        if 'Paths' in cfg and 'DATAPATH' in cfg['Paths']:
            self.datapath = cfg['Paths']['DATAPATH']
