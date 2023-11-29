import time
import torch


class Timer(object):
    def __init__(self, device):
        self.device = device
        if device == "cpu":
            self.starter = self.ender = None
        else:
            self.starter = torch.cuda.Event(enable_timing=True)
            self.ender = torch.cuda.Event(enable_timing=True)
        
    def start(self):
        if self.device == "cpu":
            self.starter = time.time()
        else:
            torch.cuda.synchronize()
            self.starter.record()
    
    def end(self):
        if self.device == "cpu":
            self.ender = time.time()
        else:
            self.ender.record()
        
    def elapsed_time(self):
        if self.device == "cpu":
            return (self.ender - self.starter) * 1000
        else:
            torch.cuda.synchronize()
            return self.starter.elapsed_time(self.ender)