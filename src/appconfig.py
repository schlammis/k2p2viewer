import os
import sys
import configparser

def _find_config():
    if getattr(sys, 'frozen', False):
        # PyInstaller: look next to the .exe
        return os.path.join(os.path.dirname(sys.executable), 'config.ini')
    return os.path.join(os.path.dirname(__file__), '..', 'config.ini')

_CONFIG_PATH = _find_config()

class AppConfig:
    def __init__(self):
        self.datapath = ''
        cfg = configparser.ConfigParser()
        cfg.read(os.path.abspath(_CONFIG_PATH))
        if 'Paths' in cfg and 'DATAPATH' in cfg['Paths']:
            self.datapath = cfg['Paths']['DATAPATH']
