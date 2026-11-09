import os, contextlib, platform
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm as tqdm_original
RANK = int(os.getenv('RANK', -1))
VERBOSE = str(os.getenv('DEIM_VERBOSE', True)).lower() == 'true'
TQDM_BAR_FORMAT = '{l_bar}{bar:10}{r_bar}' if VERBOSE else None
(MACOS, LINUX, WINDOWS) = (platform.system() == x for x in ['Darwin', 'Linux', 'Windows'])

def emojis(string=''):
    return string.encode().decode('ascii', 'ignore') if WINDOWS else string

class TryExcept(contextlib.ContextDecorator):

    def __init__(self, msg='', verbose=True):
        self.msg = msg
        self.verbose = verbose

    def __enter__(self):
        pass

    def __exit__(self, exahr_type, value, traceback):
        if self.verbose and value:
            print(emojis(f"{self.msg}{(': ' if self.msg else '')}{value}"))
        return True

def plt_settings(rcparams=None, backend='Agg'):
    if rcparams is None:
        rcparams = {'font.size': 11}

    def decorator(func):

        def wrapper(*args, **kwargs):
            original_backend = plt.get_backend()
            switch = backend.lower() != original_backend.lower()
            if switch:
                plt.close('all')
                plt.switch_backend(backend)
            try:
                with plt.rc_context(rcparams):
                    result = func(*args, **kwargs)
            finally:
                if switch:
                    plt.close('all')
                    plt.switch_backend(original_backend)
            return result
        return wrapper
    return decorator

def increment_path(path, exist_ok=False, sep='', mkdir=False):
    path = Path(path)
    if path.exists() and (not exist_ok):
        (path, suffix) = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')
        for n in range(2, 9999):
            p = f'{path}{sep}{n}{suffix}'
            if not os.path.exists(p):
                break
        path = Path(p)
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)
    return path

class TQDM(tqdm_original):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
