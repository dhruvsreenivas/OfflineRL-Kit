import numpy as np
import torch
import torch.nn as nn

class BaseReward(object):
    def __init__(
        self,
        model: nn.Module,
        optim: torch.optim.Optimizer
    ) -> None:
        super().__init__()
        self.model = model
        self.optim = optim
        
    def get_reward(
        self,
        obs: np.ndarray,
        action: np.ndarray
    ) -> np.ndarray:
        raise NotImplementedError