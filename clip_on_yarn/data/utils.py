"""Data utils"""
from multiprocessing import Value


class SharedEpoch:
    """Shared epoch store to sync epoch to dataloader worker proc"""

    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value("i", epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):

        return self.shared_epoch.value
