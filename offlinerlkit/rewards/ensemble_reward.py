import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from typing import Callable, List, Tuple, Dict, Optional
from offlinerlkit.rewards import BaseReward
from offlinerlkit.utils.scaler import StandardScaler
from offlinerlkit.utils.logger import Logger
from offlinerlkit.buffer import PreferenceDataset, filter


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
        
        # choose one from the ensemble (elite set here for evaluation)
        batch_size = ensemble_rewards.shape[1]
        model_idxs = self.model.random_elite_idxs(batch_size)
        rewards = ensemble_rewards[model_idxs, np.arange(batch_size)]
        
        info = {}
        # uncertainty penalty
        if self._penalty_coef:
            if self._uncertainty_mode == "aleatoric":
                # reward logits of size (ensemble, batch_size, 1) -> find probs, compute std of categorical distribution?
                probs = np.exp(ensemble_rewards)
                stds = np.sqrt(probs * (1.0 - probs)) # (ensemble, batch_size, 1) -- is this correct computation?
                penalty = np.amax(np.linalg.norm(stds, axis=2), axis=0) # (batch_size)
            elif self._uncertainty_mode == "pairwise-diff":
                # max difference from mean
                ensemble_mean = np.mean(ensemble_rewards, axis=0)
                diff = ensemble_rewards - ensemble_mean
                penalty = np.amax(np.linalg.norm(diff, axis=2), axis=0)
            elif self._uncertainty_mode == "ensemble-std":
                penalty = np.sqrt(ensemble_rewards.var(0).mean(1))
            else:
                raise ValueError
            
            penalty = np.expand_dims(penalty, 1).astype(np.float32)
            assert penalty.shape == rewards.shape
            rewards = rewards - self._penalty_coef * penalty
            info["penalty"] = penalty
            
        return rewards
    
    # TODO implement train, learn and validate functions
    def train(
        self,
        dataset: PreferenceDataset,
        logger: Logger,
        max_epochs: Optional[float] = None,
        max_epochs_since_update: int = 5,
        batch_size: int = 256, # snippet batch size
        holdout_ratio: float = 0.2
    ) -> None:
        data_size = len(dataset)
        holdout_size = min(int(data_size * holdout_ratio), 1000)
        train_size = data_size - holdout_size
        train_splits, holdout_splits = torch.utils.data.random_split(range(data_size), (train_size, holdout_size))
        
        # split into training and validation
        train_dataset, validation_dataset = filter(dataset, train_splits), filter(dataset, holdout_splits)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
        val_dataloader = DataLoader(validation_dataset, batch_size=batch_size)
        
        # now train
        epoch = 0
        cnt = 0
        holdout_losses = [1e10 for i in range(self.model.num_ensemble)]
        logger.log("Training reward:")
        while True:
            epoch += 1
            train_loss = self.learn(train_dataloader)
            new_holdout_losses = self.validate(val_dataloader)
            holdout_loss = (np.sort(new_holdout_losses)[:self.model.num_elites]).mean()
            
            # log
            logger.logkv("loss/reward_train_loss", train_loss)
            logger.logkv("loss/reward_holdout_loss", holdout_loss)
            logger.set_timestep(epoch)
            logger.dumpkvs(exclude=["policy_training_progress"])
            
            indexes = []
            for i, new_loss, old_loss in zip(range(len(holdout_losses)), new_holdout_losses, holdout_losses):
                improvement = (old_loss - new_loss) / old_loss
                if improvement > 0.01:
                    indexes.append(i)
                    holdout_losses[i] = new_loss
            
            if len(indexes) > 0:
                self.model.update_save(indexes)
                cnt = 0
            else:
                cnt += 1
            
            if (cnt >= max_epochs_since_update) or (max_epochs and (epoch >= max_epochs)):
                break
            
        indexes = self.select_elites(holdout_losses)
        self.model.set_elites(indexes)
        self.model.load_save()
        self.save(logger.model_dir)
        self.model.eval()
        logger.log("elites:{} , holdout loss: {}".format(indexes, (np.sort(holdout_losses)[:self.model.num_elites]).mean()))
        
    def learn(
        self,
        dataloader: DataLoader
    ) -> List[float]:
        
        self.model.train()
        losses = []
        for batch in dataloader:
            ensemble_pred_rew1 = self.model(batch["observations1"], batch["actions1"]).sum(2).squeeze() # size (n_ensemble, batch_size) -> sum(\hat{r}(\tau1))
            ensemble_pred_rew2 = self.model(batch["observations2"], batch["actions2"]).sum(2).squeeze() # size (n_ensemble, batch_size) -> sum(\hat{r}(\tau1))
            
            ensemble_pred_rewsum = ensemble_pred_rew1 + ensemble_pred_rew2
            label_preds = ensemble_pred_rew1 / (ensemble_pred_rewsum + 1e-8) # for normalization
            
            label_gt = batch["label"].unsqueeze(0).repeat(self.model.num_ensemble, 1) # (n_ensemble, batch_size)
            loss = F.binary_cross_entropy(label_preds, label_gt)
        
            # training step
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            
            losses.append(loss.item())
        
        return np.mean(losses)
    
    @torch.no_grad()
    def validate(self, val_dataloader: DataLoader) -> List[float]:
        self.model.eval()
        total_loss = 0.0

        for batch in val_dataloader:
            ensemble_pred_rew1 = self.model(batch["observations1"], batch["actions1"]).sum(2).squeeze() # size (n_ensemble, batch_size) -> sum(\hat{r}(\tau1))
            ensemble_pred_rew2 = self.model(batch["observations2"], batch["actions2"]).sum(2).squeeze() # size (n_ensemble, batch_size) -> sum(\hat{r}(\tau1))
            
            ensemble_pred_rewsum = ensemble_pred_rew1 + ensemble_pred_rew2
            label_preds = ensemble_pred_rew1 / (ensemble_pred_rewsum + 1e-8) # for normalization
            
            label_gt = batch["label"].unsqueeze(0).repeat(self.model.num_ensemble, 1) # (n_ensemble, batch_size)
            loss = F.binary_cross_entropy(label_preds, label_gt)
            total_loss += loss
        
        val_loss = list(total_loss.cpu().numpy())
        return val_loss
    
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

    
    