import contextlib, time
import torch

class Profile(contextlib.ContextDecorator):

    def __init__(self, t=0.0, device: torch.device=None):
        self.t = t
        self.device = device
        self.cuda = bool(device and str(device).startswith('cuda'))

    def __enter__(self):
        self.start = self.time()
        return self

    def __exit__(self, type, value, traceback):
        self.dt = self.time() - self.start
        self.t += self.dt

    def __str__(self):
        return f'Elapsed time is {self.t} s'

    def time(self):
        if self.cuda:
            torch.cuda.synchronize(self.device)
        return time.time()
