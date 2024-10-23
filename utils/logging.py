import sys
from tqdm import tqdm

class Logger(object):
    def __init__(self, log_path, no_terminal_write: bool=False):
        self.terminal = sys.stdout
        self.log = open(log_path, "a")

        self.no_terminal_write = no_terminal_write
   
    def write(self, message):
        if not self.no_terminal_write:
            self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        self.log.flush()  

    def close(self):
        self.log.close()