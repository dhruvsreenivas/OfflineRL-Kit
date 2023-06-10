import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Callable, List, Tuple, Dict, Optional
from offlinerlkit.dynamics import BaseDynamics
from offlinerlkit.modules import EnsembleDynamicsModel, EnsembleDynamicsModelWithSeparateReward
from offlinerlkit.utils.scaler import StandardScaler
from offlinerlkit.utils.logger import Logger
from offlinerlkit.buffer import PreferenceDataset


class EnsembleDynamics(BaseDynamics):
    def __init__(
        self,
        model: nn.Module,
        optim: torch.optim.Optimizer,
        scaler: StandardScaler,
        terminal_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
        penalty_coef: float = 0.0,
        uncertainty_mode: str = "aleatoric"
    ) -> None:
        super().__init__(model, optim)
        assert isinstance(self.model, EnsembleDynamicsModel) or isinstance(self.model, EnsembleDynamicsModelWithSeparateReward)
        
        self.scaler = scaler
        self.terminal_fn = terminal_fn
        self._penalty_coef = penalty_coef
        self._uncertainty_mode = uncertainty_mode

    @torch.no_grad()
    def step(
        self,
        obs: np.ndarray,
        action: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        "imagine single forward step"
        obs_act = np.concatenate([obs, action], axis=-1)
        obs_act = self.scaler.transform(obs_act)
        
        if isinstance(self.model, EnsembleDynamicsModel):
            mean, logvar = self.model(obs_act)
            mean = mean.cpu().numpy()
            logvar = logvar.cpu().numpy()
            mean[..., :-1] += obs # next state
            std = np.sqrt(np.exp(logvar)) # [n_ensemble, batch_size, (s' + r) dim]
        else:
            assert isinstance(self.model, EnsembleDynamicsModelWithSeparateReward), "model should have separate reward head!"
            mean, logvar, reward = self.model(obs_act)
            mean = mean.cpu().numpy()
            logvar = logvar.cpu().numpy()
            mean += obs
            std = np.sqrt(np.exp(logvar)) # [n_ensemble, batch_size, s' dim]

        # sample from dist
        ensemble_samples = (mean + np.random.normal(size=mean.shape) * std).astype(np.float32)

        # choose one model from ensemble
        num_models, batch_size, _ = ensemble_samples.shape
        model_idxs = self.model.random_elite_idxs(batch_size)
        samples = ensemble_samples[model_idxs, np.arange(batch_size)]
        if isinstance(self.model, EnsembleDynamicsModelWithSeparateReward):
            reward = reward[model_idxs, np.arange(batch_size)]
        
        if isinstance(self.model, EnsembleDynamicsModel):
            next_obs = samples[..., :-1]
            reward = samples[..., -1:]
        else:
            next_obs = samples
        
        terminal = self.terminal_fn(obs, action, next_obs)
        info = {}
        info["raw_reward"] = reward

        if self._penalty_coef:
            if self._uncertainty_mode == "aleatoric":
                penalty = np.amax(np.linalg.norm(std, axis=2), axis=0) # (batch_size)
            elif self._uncertainty_mode == "pairwise-diff":
                next_obses_mean = mean[..., :-1] if isinstance(self.model, EnsembleDynamicsModel) else mean
                next_obs_mean = np.mean(next_obses_mean, axis=0)
                diff = next_obses_mean - next_obs_mean
                penalty = np.amax(np.linalg.norm(diff, axis=2), axis=0)
            elif self._uncertainty_mode == "ensemble_std":
                next_obses_mean = mean[..., :-1]
                penalty = np.sqrt(next_obses_mean.var(0).mean(1))
            else:
                raise ValueError
            
            penalty = np.expand_dims(penalty, 1).astype(np.float32)
            assert penalty.shape == reward.shape
            reward = reward - self._penalty_coef * penalty
            info["penalty"] = penalty
        
        return next_obs, reward, terminal, info
    
    @torch.no_grad()
    def sample_next_obss(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        num_samples: int
    ) -> torch.Tensor:
        obs_act = torch.cat([obs, action], dim=-1)
        obs_act = self.scaler.transform_tensor(obs_act)
        if isinstance(self.model, EnsembleDynamicsModel):
            mean, logvar = self.model(obs_act)
            mean[..., :-1] += obs
            std = torch.sqrt(torch.exp(logvar))
        else:
            mean, logvar, _ = self.model(obs_act)
            mean += obs
            std = torch.sqrt(torch.exp(logvar))

        mean = mean[self.model.elites.data.cpu().numpy()]
        std = std[self.model.elites.data.cpu().numpy()]

        samples = torch.stack([mean + torch.randn_like(std) * std for i in range(num_samples)], 0)
        next_obss = samples[..., :-1] if isinstance(self.model, EnsembleDynamicsModel) else samples
        return next_obss

    def format_samples_for_training(self, data: Dict) -> Tuple[np.ndarray, np.ndarray]:
        obss = data["observations"]
        actions = data["actions"]
        next_obss = data["next_observations"]
        rewards = data["rewards"]
        delta_obss = next_obss - obss
        inputs = np.concatenate((obss, actions), axis=-1)
        targets = np.concatenate((delta_obss, rewards), axis=-1)
        return inputs, targets

    def train(
        self,
        data: Dict,
        preference_dataset: Optional[PreferenceDataset],
        logger: Logger,
        max_epochs: Optional[float] = None,
        max_epochs_since_update: int = 5,
        batch_size: int = 256,
        holdout_ratio: float = 0.2,
        logvar_loss_coef: float = 0.01
    ) -> None:
        # trains on full offline data only right now, add preference data training.
        inputs, targets = self.format_samples_for_training(data)
        data_size = inputs.shape[0]
        holdout_size = min(int(data_size * holdout_ratio), 1000)
        train_size = data_size - holdout_size
        train_splits, holdout_splits = torch.utils.data.random_split(range(data_size), (train_size, holdout_size))
        train_inputs, train_targets = inputs[train_splits.indices], targets[train_splits.indices]
        holdout_inputs, holdout_targets = inputs[holdout_splits.indices], targets[holdout_splits.indices]

        self.scaler.fit(train_inputs)
        train_inputs = self.scaler.transform(train_inputs)
        holdout_inputs = self.scaler.transform(holdout_inputs)
        holdout_losses = [1e10 for _ in range(self.model.num_ensemble)]

        data_idxes = np.random.randint(train_size, size=[self.model.num_ensemble, train_size])
        def shuffle_rows(arr):
            idxes = np.argsort(np.random.uniform(size=arr.shape), axis=-1)
            return arr[np.arange(arr.shape[0])[:, None], idxes]

        epoch = 0
        cnt = 0
        logger.log("Training dynamics:")
        while True:
            epoch += 1
            
            if preference_dataset is not None:
                pref_batch = preference_dataset.sample(batch_size)
            else:
                pref_batch = None
            
            train_loss = self.learn(train_inputs[data_idxes], train_targets[data_idxes], pref_batch, batch_size, logvar_loss_coef)
            new_holdout_losses = self.validate(holdout_inputs, holdout_targets)
            holdout_loss = (np.sort(new_holdout_losses)[:self.model.num_elites]).mean()
            logger.logkv("loss/dynamics_train_loss", train_loss)
            logger.logkv("loss/dynamics_holdout_loss", holdout_loss)
            logger.set_timestep(epoch)
            logger.dumpkvs(exclude=["policy_training_progress"])

            # shuffle data for each base learner
            data_idxes = shuffle_rows(data_idxes)

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
        inputs: np.ndarray,
        targets: np.ndarray,
        pref_batch: Optional[Dict[str, torch.Tensor]],
        batch_size: int = 256,
        logvar_loss_coef: float = 0.01
    ) -> float:
        self.model.train()
        train_size = inputs.shape[1]
        losses = []

        for batch_num in range(int(np.ceil(train_size / batch_size))):
            inputs_batch = inputs[:, batch_num * batch_size:(batch_num + 1) * batch_size]
            targets_batch = targets[:, batch_num * batch_size:(batch_num + 1) * batch_size]
            targets_batch = torch.as_tensor(targets_batch).to(self.model.device)
            
            # get outputs
            if isinstance(self.model, EnsembleDynamicsModel):
                mean, logvar = self.model(inputs_batch)
                inv_var = torch.exp(-logvar)
            else:
                mean, logvar, _ = self.model(inputs_batch)
                targets_batch = targets_batch[..., :-1] # only next state
                inv_var = torch.exp(-logvar)
            
            # Average over batch and dim, sum over ensembles.
            mse_loss_inv = (torch.pow(mean - targets_batch, 2) * inv_var).mean(dim=(1, 2))
            var_loss = logvar.mean(dim=(1, 2))
            loss = mse_loss_inv.sum() + var_loss.sum()
            loss = loss + self.model.get_decay_loss()
            loss = loss + logvar_loss_coef * self.model.max_logvar.sum() - logvar_loss_coef * self.model.min_logvar.sum()
            
            # get reward loss over preference batch
            if pref_batch is not None:
                obs_actions1 = torch.cat([pref_batch["observations1"], pref_batch["actions1"]], dim=-1)
                obs_actions2 = torch.cat([pref_batch["observations2"], pref_batch["actions2"]], dim=-1)
                
                _, _, ensemble_pred_rew1 = self.model(obs_actions1)
                _, _, ensemble_pred_rew2 = self.model(obs_actions2)
                ensemble_pred_rew1 = ensemble_pred_rew1.sum(2).squeeze().exp() # size (n_ensemble, batch_size) -> sum(\hat{r}(\tau1))
                ensemble_pred_rew2 = ensemble_pred_rew2.sum(2).squeeze().exp() # size (n_ensemble, batch_size) -> sum(\hat{r}(\tau2))
                assert (ensemble_pred_rew1 >= 0).all()
                assert (ensemble_pred_rew2 >= 0).all()
                
                # predicted probability of preference reward
                ensemble_pred_rewsum = ensemble_pred_rew1 + ensemble_pred_rew2
                assert (ensemble_pred_rewsum >= ensemble_pred_rew1).all()
                label_preds = ensemble_pred_rew1 / ensemble_pred_rewsum # for normalization it is not necessary cuz everything is > 0
                
                # ground truth label from preference dataset
                label_gt = pref_batch["label"].tile(self.model.num_ensemble, 1) # (num_ensemble,)
                
                print(f'label preds: {label_preds}')
                print(f'label preds size: {label_preds.size()}')
                print(f'ground truth: {label_gt}')
                print(f'ground truth size: {label_gt.size()}')
                
                reward_loss = F.binary_cross_entropy(label_preds, label_gt)
                
                loss = loss + reward_loss

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            losses.append(loss.item())
        return np.mean(losses)
    
    @torch.no_grad()
    def validate(self, inputs: np.ndarray, targets: np.ndarray) -> List[float]:
        self.model.eval()
        targets = torch.as_tensor(targets).to(self.model.device)
        mean, _ = self.model(inputs)
        loss = ((mean - targets) ** 2).mean(dim=(1, 2))
        print(f"validation loss shape: {loss.shape}")
        val_loss = list(loss.cpu().numpy())
        return val_loss
    
    def select_elites(self, metrics: List) -> List[int]:
        pairs = [(metric, index) for metric, index in zip(metrics, range(len(metrics)))]
        pairs = sorted(pairs, key=lambda x: x[0])
        elites = [pairs[i][1] for i in range(self.model.num_elites)]
        return elites

    def save(self, save_path: str) -> None:
        torch.save(self.model.state_dict(), os.path.join(save_path, "dynamics.pth"))
        self.scaler.save_scaler(save_path)
    
    def load(self, load_path: str) -> None:
        self.model.load_state_dict(torch.load(os.path.join(load_path, "dynamics.pth"), map_location=self.model.device))
        self.scaler.load_scaler(load_path)