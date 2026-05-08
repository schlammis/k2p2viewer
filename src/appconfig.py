import os
import sys
import configparser

def _find_config():
    if getattr(sys, 'frozen', False):
        return os.path.join(os.path.dirname(sys.executable), 'config.ini')
    return os.path.join(os.path.dirname(__file__), '..', 'config.ini')

_CONFIG_PATH = _find_config()

def _read_version():
    base = os.path.dirname(os.path.abspath(_CONFIG_PATH))
    for candidate in [base, os.path.join(base, '..')]:
        try:
            with open(os.path.join(candidate, 'version.txt')) as f:
                return f.read().strip()
        except Exception:
            pass
    return 'unknown'

class AppConfig:
    def __init__(self):
        self.balance_name = ''
        self.datapath = ''
        self.loglevel = 10
        self.fit_order = 6
        self.use_sinc = False
        self.drop_n = 0
        self.needs_setup = True
        cfg = configparser.ConfigParser()
        cfg.read(os.path.abspath(_CONFIG_PATH))
        if 'GENERAL' in cfg and 'selected balance' in cfg['GENERAL']:
            g = cfg['GENERAL']
            name = g['selected balance']
            if name and name in cfg and 'datapath' in cfg[name]:
                self.balance_name = name
                self.datapath = cfg[name]['datapath']
                self.needs_setup = False
            try:
                self.loglevel = int(g.get('loglevel', '10'))
            except ValueError:
                self.loglevel = 10
            try:
                self.fit_order = int(g.get('fit_order', '6'))
            except ValueError:
                pass
            self.use_sinc = g.get('use_sinc', 'false').lower() == 'true'
            try:
                self.drop_n = int(g.get('drop_n', '0'))
            except ValueError:
                pass
            if 'loglevel' not in g:
                AppConfig.save(self.balance_name, self.datapath, self.loglevel)

    @staticmethod
    def save(balance_name, datapath, loglevel=10):
        cfg = configparser.ConfigParser()
        cfg.read(os.path.abspath(_CONFIG_PATH))
        if 'GENERAL' not in cfg:
            cfg['GENERAL'] = {}
        cfg['GENERAL']['selected balance'] = balance_name
        cfg['GENERAL']['version'] = _read_version()
        cfg['GENERAL']['loglevel'] = str(loglevel)
        if balance_name not in cfg:
            cfg[balance_name] = {}
        cfg[balance_name]['datapath'] = datapath
        with open(os.path.abspath(_CONFIG_PATH), 'w') as f:
            cfg.write(f)

    @staticmethod
    def save_datapath(balance_name, datapath):
        AppConfig.save(balance_name, datapath)

    @staticmethod
    def get_all_balances():
        """Return {name: datapath} for every configured balance."""
        cfg = configparser.ConfigParser()
        cfg.read(os.path.abspath(_CONFIG_PATH))
        result = {}
        for section in cfg.sections():
            if section.upper() == 'GENERAL':
                continue
            if 'datapath' in cfg[section]:
                result[section] = cfg[section]['datapath']
        return result

    @staticmethod
    def update_datapath(balance_name, datapath):
        """Update only the datapath for an existing balance, without changing selected balance."""
        cfg = configparser.ConfigParser()
        cfg.read(os.path.abspath(_CONFIG_PATH))
        if balance_name not in cfg:
            cfg[balance_name] = {}
        cfg[balance_name]['datapath'] = datapath
        with open(os.path.abspath(_CONFIG_PATH), 'w') as f:
            cfg.write(f)

    @staticmethod
    def save_ui_settings(fit_order, use_sinc, drop_n):
        cfg = configparser.ConfigParser()
        cfg.read(os.path.abspath(_CONFIG_PATH))
        if 'GENERAL' not in cfg:
            cfg['GENERAL'] = {}
        cfg['GENERAL']['fit_order'] = str(fit_order)
        cfg['GENERAL']['use_sinc']  = 'true' if use_sinc else 'false'
        cfg['GENERAL']['drop_n']    = str(drop_n)
        with open(os.path.abspath(_CONFIG_PATH), 'w') as f:
            cfg.write(f)

    @staticmethod
    def delete_balance(name):
        """Remove a balance section from config.ini."""
        cfg = configparser.ConfigParser()
        cfg.read(os.path.abspath(_CONFIG_PATH))
        cfg.remove_section(name)
        if 'GENERAL' in cfg and cfg['GENERAL'].get('selected balance') == name:
            cfg.remove_option('GENERAL', 'selected balance')
        with open(os.path.abspath(_CONFIG_PATH), 'w') as f:
            cfg.write(f)

    @staticmethod
    def rename_balance(old_name, new_name):
        """Rename a balance section in config.ini. Updates 'selected balance' if needed."""
        cfg = configparser.ConfigParser()
        cfg.read(os.path.abspath(_CONFIG_PATH))
        if old_name not in cfg:
            return
        # copy all keys from old section to new section
        cfg[new_name] = dict(cfg[old_name])
        cfg.remove_section(old_name)
        if 'GENERAL' in cfg and cfg['GENERAL'].get('selected balance') == old_name:
            cfg['GENERAL']['selected balance'] = new_name
        with open(os.path.abspath(_CONFIG_PATH), 'w') as f:
            cfg.write(f)
