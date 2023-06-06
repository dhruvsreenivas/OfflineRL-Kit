import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Callable, List, Tuple, Dict, Optional
from offlinerlkit.rewards import BaseReward
from offlinerlkit.utils.scaler import StandardScaler
from offlinerlkit.utils.logger import Logger
from offlinerlkit.buffer import PreferenceDataset


class EnsembleReward(BaseReward):
    def __init__(
        self,
        model: nn.Module,
        optim: torch.optim.Optimizer,
        scaler: Optional[StandardScaler],
        penalty_coef: float = 0.0,
        uncertainty_mode: str = "aleatoric"
    ) -> None:
        super().__init__(model, optim)
        
        self.scaler = scaler
        self._penalty_coef = penalty_coef
        self._uncertainty_mode = uncertainty_mode
        
    @torch.no_grad()
    def get_reward(
        self,
        obs: np.ndarray,
        action: np.ndarray
    ) -> np.ndarray:
        # scale and split back
        if self.scaler is not None:
            obs_act = np.concatenate([obs, action], axis=-1)
            obs_act = self.scaler.transform(obs_act)
            obs, action = np.split(obs_act, [obs.shape[-1]], axis=-1)
        
        ensemble_rewards = self.model(obs, action)
        ensemble_rewards = ensemble_rewards.cpu().numpy()
        
        # choose one from the ensemble (elite set in evaluation)
        batch_size = ensemble_rewards.shape[1]
        model_idxs = self.model.random_elite_idxs(batch_size)
        rewards = ensemble_rewards[model_idxs, np.arange(batch_size)]
        
        info = {}
        # uncertainty penalty
        if self._penalty_coef:
            if self._uncertainty_mode == "aleatoric":
                # reward logits of size (ensemble, batch_size, 1) -> find probs, compute std of categorical distribution?
                probs = np.exp(ensemble_rewards)
                stds = np.sqrt(probs * (1.0 - probs)) # (ensemble, batch_size, 1) -- is this correct?
                penalty = np.amax(np.linalg.norm(stds, axis=2), axis=0) # (batch_size)
            elif self._uncertainty_mode == "pairwise-diff":
                # max difference from mean
                ensemble_mean = np.mean(ensemble_rewards, dim=0)
                diff = ensemble_rewards - ensemble_mean
                penalty = np.amax(np.linalg.norm(diff, axis=2), dim=0)
            elif self._uncertainty_mode == "ensemble-std":
                penalty = np.sqrt(ensemble_rewards.var(0).mean(1))
            else:
                raise ValueError
            
            penalty = np.expand_dims(penalty, 1).astype(np.float32)
            assert penalty.shape == rewards.shape
            rewards = rewards - self._penalty_coef * penalty
            info["penalty"] = penalty
            
        return rewards
    
    # TODO implement train and validate datasets
    
    def select_elites(self, metrics: List) -> List[int]:
        pairs = [(metric, index) for metric, index in zip(metrics, range(len(metrics)))]
        pairs = sorted(pairs, key=lambda x: x[0])
        elites = [pairs[i][1] for i in range(self.model.num_elites)]
        return elites
    
    def save(self, save_path: str) -> None:
        torch.save(self.model.state_dict(), os.path.join(save_path, "reward.pth"))
        if self.scaler is not None:
            self.scaler.save_scaler(save_path)
    
    def load(self, load_path: str) -> None:
        self.model.load_state_dict(torch.load(os.path.join(load_path, "reward.pth"), map_location=self.model.device))
        if self.scaler is not None:
            self.scaler.load_scaler(load_path)

    
    