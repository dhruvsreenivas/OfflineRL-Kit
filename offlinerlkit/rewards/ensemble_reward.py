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
from offlinerlkit.utils.losses import ensemble_cross_entropy


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
        if torch.is_tensor(obs):
            obs = obs.cpu().numpy()
        if torch.is_tensor(action):
            action = action.cpu().numpy()
        # concat, scale and split back
        if self.scaler is not None:
            obs_act = np.concatenate([obs, action], axis=-1)
            obs_act = self.scaler.transform(obs_act)
            obs, action = np.split(obs_act, [obs.shape[-1]], axis=-1)
        
        obs = torch.from_numpy(obs).to(self.model.device)
        action = torch.from_numpy(action).to(self.model.device)
        ensemble_rewards, _ = self.model(obs, action, train=False)
        ensemble_rewards = ensemble_rewards.cpu().numpy()
        # print(f'ensemble reward shape: {ensemble_rewards.shape}') # (n_ensemble, seg_len, 1)
        
        # choose one from the ensemble (elite set here for evaluation)
        batch_size = ensemble_rewards.shape[1]
        model_idxs = self.model.random_elite_idxs(batch_size)
        rewards = ensemble_rewards[model_idxs, np.arange(batch_size)]
        # print(f'reward shape: {rewards.shape}') # (seg_len, 1)
        
        info = {}
        if self._penalty_coef:
            # uncertainty penalty -- TODO fix
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
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        val_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        # fit scaler
        if self.scaler is not None:
            mu, std = train_dataset.statistics(1e-6)
            self.scaler.mu = mu
            self.scaler.std = std

        # now train
        epoch = 0
        cnt = 0
        holdout_losses = [1e10 for i in range(self.model.num_ensemble)]
        logger.log("======== Training reward... ========")
        while True:
            epoch += 1
            train_loss = self.learn(train_dataloader)
            new_holdout_losses, new_val_accs = self.validate(val_dataloader)
            holdout_loss = (np.sort(new_holdout_losses)[:self.model.num_elites]).mean()
            val_acc = (np.sort(new_val_accs)[-self.model.num_elites:]).mean()
            
            # log
            logger.logkv("loss/reward_train_loss", train_loss)
            logger.logkv("loss/reward_holdout_loss", holdout_loss)
            logger.logkv("loss/reward_accuracy", val_acc)
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
            assert isinstance(batch["observations1"], torch.Tensor)
            assert isinstance(batch["observations2"], torch.Tensor)
            assert isinstance(batch["actions1"], torch.Tensor)
            assert isinstance(batch["actions2"], torch.Tensor)
            
            if self.scaler is not None:
                obs_act_1 = torch.cat([batch["observations1"], batch["actions1"]], dim=-1)
                obs_act_1 = self.scaler.transform(obs_act_1)
                obs1, act1 = torch.split(obs_act_1, [batch["observations1"].size(-1), batch["actions1"].size(-1)], dim=-1)

                obs_act_2 = torch.cat([batch["observations2"], batch["actions2"]], dim=-1)
                obs_act_2 = self.scaler.transform(obs_act_2)
                obs2, act2 = torch.split(obs_act_2, [batch["observations2"].size(-1), batch["actions2"].size(-1)], dim=-1)
            else:
                obs1, act1, obs2, act2 = batch["observations1"], batch["actions1"], batch["observations2"], batch["actions2"]

            # testing if there are nans in training
            # print(f"are there nans being fed into TRAINING: {torch.isnan(obs1).any().item(), torch.isnan(act1).any().item(), torch.isnan(obs2).any().item(), torch.isnan(act2).any().item()}")
            
            # compute rewards and sum for trajectories
            ensemble_pred_rew1, masks = self.model(obs1, act1, train=True) # size (n_ensemble, batch_size) -> sum(\hat{r}(\tau1))
            ensemble_pred_rew2 = self.model(obs2, act2, masks=masks, train=True) # size (n_ensemble, batch_size) -> sum(\hat{r}(\tau2))
            # print(f"are any of these rewards nans: {torch.isnan(ensemble_pred_rew1).any().item(), torch.isnan(ensemble_pred_rew2).any().item()}")
            
            ensemble_pred_rew1 = ensemble_pred_rew1.sum(2) # (n_ensemble, batch_size, 1), sum(\hat{r}(\tau1)) -> logits
            ensemble_pred_rew2 = ensemble_pred_rew2.sum(2) # (n_ensemble, batch_size, 1), sum(\hat{r}(\tau2)) -> logits
            
            # get stacked logits before throwing to cross entropy loss
            ensemble_pred_rew = torch.cat([ensemble_pred_rew1, ensemble_pred_rew2], dim=-1) # (n_ensemble, batch_size, 2)

            # ground truth label from preference dataset, sum over ensemble
            label_gt = (1.0 - batch["label"]).long() # (batch_size), 1 if rew1 > rew2 else 0 so have to reverse it, as model outputs 0 if rew1 > rew2 else 1
            reward_loss = ensemble_cross_entropy(ensemble_pred_rew, label_gt, reduction='sum') # done in OPRL paper
        
            # training step
            self.optim.zero_grad()
            reward_loss.backward()
            self.optim.step()
            
            losses.append(reward_loss.item())
            
            # delete batch to save memory?
            del batch
        
        return np.mean(losses)
    
    @torch.no_grad()
    def validate(self, val_dataloader: DataLoader) -> List[float]:
        self.model.eval()
        total_loss = 0.0
        total_acc = 0.0
        
        count = 0
        for batch in val_dataloader:
            assert isinstance(batch["observations1"], torch.Tensor)
            assert isinstance(batch["observations2"], torch.Tensor)
            assert isinstance(batch["actions1"], torch.Tensor)
            assert isinstance(batch["actions2"], torch.Tensor)
            
            if self.scaler is not None:
                obs_act_1 = torch.cat([batch["observations1"], batch["actions1"]], dim=-1)
                obs_act_1 = self.scaler.transform(obs_act_1)
                obs1, act1 = torch.split(obs_act_1, [batch["observations1"].size(-1), batch["actions1"].size(-1)], dim=-1)

                obs_act_2 = torch.cat([batch["observations2"], batch["actions2"]], dim=-1)
                obs_act_2 = self.scaler.transform(obs_act_2)
                obs2, act2 = torch.split(obs_act_2, [batch["observations2"].size(-1), batch["actions2"].size(-1)], dim=-1)
            else:
                obs1, act1, obs2, act2 = batch["observations1"], batch["actions1"], batch["observations2"], batch["actions2"]

            # print(f"are there nans being fed into VALIDATION: {torch.isnan(obs1).any().item(), torch.isnan(act1).any().item(), torch.isnan(obs2).any().item(), torch.isnan(act2).any().item()}")

            ensemble_pred_rew1, _ = self.model(obs1, act1, train=False) # size (n_ensemble, batch_size) -> sum(\hat{r}(\tau1))
            ensemble_pred_rew2, _ = self.model(obs2, act2, train=False) # size (n_ensemble, batch_size) -> sum(\hat{r}(\tau2))
            
            ensemble_pred_rew1 = ensemble_pred_rew1.sum(2) # (n_ensemble, batch_size), sum(\hat{r}(\tau1)) -> logits
            ensemble_pred_rew2 = ensemble_pred_rew2.sum(2) # (n_ensemble, batch_size), sum(\hat{r}(\tau2)) -> logits
            
            # get stacked logits before throwing to cross entropy loss
            ensemble_pred_rew = torch.cat([ensemble_pred_rew1, ensemble_pred_rew2], dim=-1) # (n_ensemble, batch_size, 2)
            lbl_preds = torch.argmax(ensemble_pred_rew, dim=-1) # returns 0 if the first trajectory is maximal, 1 if not

            # ground truth label from preference dataset
            label_gt = (1.0 - batch["label"]).long() # (batch_size) (1 if first trajectory is maximal else 0, so have to reverse)
            reward_loss = ensemble_cross_entropy(ensemble_pred_rew, label_gt, reduction='none') # (n_ensemble,)
            total_loss += reward_loss
            
            reward_acc = (lbl_preds == label_gt.tile(self.model.num_ensemble, 1)).float().mean(-1)
            total_acc += reward_acc
            count += 1
        
        total_loss = total_loss / count
        total_acc = total_acc / count
        
        val_loss = list(total_loss.cpu().numpy())
        val_acc = list(total_acc.cpu().numpy())
        return val_loss, val_acc
    
    def select_elites(self, metrics: List) -> List[int]:
        pairs = [(metric, index) for metric, index in zip(metrics, range(len(metrics)))]
        pairs = sorted(pairs, key=lambda x: x[0])
        elites = [pairs[i][1] for i in range(self.model.num_elites)]
        return elites
    
    def save(self, save_path: str) -> None:
        torch.save(self.model.state_dict(), os.path.join(save_path, "reward.pth"))
    
    def load(self, load_path: str) -> None:
        self.model.load_state_dict(torch.load(os.path.join(load_path, "reward.pth"), map_location=self.model.device))

    
    