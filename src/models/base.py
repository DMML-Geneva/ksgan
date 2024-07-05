import torch


class BaseModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("epochs_trained", torch.zeros(1, dtype=int))

    def epoch_completed(self):
        self.epochs_trained += 1
