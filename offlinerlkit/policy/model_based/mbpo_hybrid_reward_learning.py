import numpy as np
import torch
import torch.nn as nn
import gym

from torch.nn import functional as F
from typing import Dict, Union, Tuple
from collections import defaultdict
from offlinerlkit.policy import SACPolicy
from offlinerlkit.dynamics import BaseDynamics
from offlinerlkit.rewards import BaseReward
from offlinerlkit.utils.losses import ensemble_cross_entropy
from offlinerlkit.buffer import ReplayBuffer, PreferenceDataset


class HybridMBPORewardLearningPolicy(SACPolicy):
    """
    Hybrid version of Model-Based Policy Optimization (https://arxiv.org/abs/1906.08253), with reward learning from preferences.
    """
    def __init__(
        self,
        dynamics: BaseDynamics,
        reward: BaseReward,
        actor: nn.Module,
        critic1: nn.Module,
        critic2: nn.Module,
        actor_optim: torch.optim.Optimizer,
        critic1_optim: torch.optim.Optimizer,
        critic2_optim: torch.optim.Optimizer,
        dynamics_optim: torch.optim.Optimizer,
        reward_optim: torch.optim.Optimizer,
        tau: float = 0.005,
        gamma: float  = 0.99,
        alpha: Union[float, Tuple[float, torch.Tensor, torch.optim.Optimizer]] = 0.2,
        online_batch_size: int = 256,
        online_pref_batch_size: int = 256,
        online_to_total_ratio: float = 0.5
    ) -> None:
        super().__init__(
            actor,
            critic1,
            critic2,
            actor_optim,
            critic1_optim,
            critic2_optim,
            tau,
            gamma,
            alpha
        )
        self.dynamics = dynamics
        self.reward = reward
        self.dynamics_optim = dynamics_optim
        self.reward_optim = reward_optim
        
        self.online_batch_size = online_batch_size
        self.online_pref_batch_size = online_pref_batch_size
        self.online_to_total_ratio = online_to_total_ratio
        
    
    def rollout(
        self,
        init_obss: np.ndarray,
        rollout_length: int
    ) -> Tuple[Dict[str, np.ndarray], Dict]:
        """Roll out the given actor inside the learned dynamics."""

        num_transitions = 0
        rewards_arr = np.array([])
        rollout_transitions = defaultdict(list)

        # rollout
        observations = init_obss
        for _ in range(rollout_length):
            actions = self.select_action(observations)
            next_observations, rewards, terminals, info = self.dynamics.step(observations, actions)
            rollout_transitions["obss"].append(observations)
            rollout_transitions["next_obss"].append(next_observations)
            rollout_transitions["actions"].append(actions)
            rollout_transitions["rewards"].append(rewards)
            rollout_transitions["terminals"].append(terminals)

            num_transitions += len(observations)
            rewards_arr = np.append(rewards_arr, rewards.flatten())

            nonterm_mask = (~terminals).flatten()
            if nonterm_mask.sum() == 0:
                break

            observations = next_observations[nonterm_mask]
        
        for k, v in rollout_transitions.items():
            rollout_transitions[k] = np.concatenate(v, axis=0)

        return rollout_transitions, \
            {"num_transitions": num_transitions, "reward_mean": rewards_arr.mean()}
            
            
    def reward_loss(self, preference_batch: Dict[str, torch.Tensor], normalize_input: bool = False) -> torch.Tensor:
        obs_dim, action_dim = preference_batch["observations1"].size(-1), preference_batch["actions1"].size(-1)
        obs1, actions1 = preference_batch["observations1"], preference_batch["actions1"]
        obs2, actions2 = preference_batch["observations2"], preference_batch["actions2"]
        
        # normalize concatenated input
        if normalize_input:
            obs_actions1 = torch.cat([preference_batch["observations1"], preference_batch["actions1"]], dim=-1)
            obs_actions2 = torch.cat([preference_batch["observations2"], preference_batch["actions2"]], dim=-1)
            obs_actions1 = self.reward.scaler.transform(obs_actions1)
            obs_actions2 = self.reward.scaler.transform(obs_actions2)
            obs1, actions1 = torch.split(obs_actions1, [obs_dim, action_dim], dim=-1)
            obs2, actions2 = torch.split(obs_actions2, [obs_dim, action_dim], dim=-1)
        
        ensemble_pred_rew1, masks = self.reward.model(obs1, actions1, train=True)
        ensemble_pred_rew2 = self.reward.model(obs2, actions2, masks=masks, train=True)
        
        ensemble_pred_rew1 = ensemble_pred_rew1.sum(2) # (n_ensemble, batch_size, 1), sum(\hat{r}(\tau1)) -> logits
        ensemble_pred_rew2 = ensemble_pred_rew2.sum(2) # (n_ensemble, batch_size, 1), sum(\hat{r}(\tau2)) -> logits
        
        # get stacked logits before throwing to cross entropy loss
        ensemble_pred_rew = torch.cat([ensemble_pred_rew1, ensemble_pred_rew2], dim=-1) # (n_ensemble, batch_size, 2)
        
        # ground truth label from preference dataset
        label_gt = (1.0 - preference_batch["label"]).long() # (num_ensemble, batch_size)
        
        reward_loss = ensemble_cross_entropy(ensemble_pred_rew, label_gt, reduction='sum') # done in OPRL paper
        return reward_loss
            
            
    def learn(
        self,
        batch: Dict
    ) -> Dict[str, float]:
        """Given a batch of data of (s, a, s') triples, take a learning step on it.
        
        This batch is of hybrid form (i.e. concatenated (s, a, s') triples from the online + offline + model-based datasets, with some fixed ratio).
        """
        
        # combine online real, offline real, and online model-based data together
        assert "real_online" in batch, "Real online data expected in the batch."
        assert "real_offline" in batch, "Real offline data expected in the batch."
        assert "fake" in batch, "Model-based data expected in the batch."
        
        real_online_batch, real_offline_batch, fake_batch = batch["real_online"], batch["real_offline"], batch["fake"]
        mix_batch = {k: torch.cat([real_online_batch[k], real_offline_batch[k], fake_batch[k]], dim=0) for k in real_online_batch.keys()}
        
        # grab real reward from learned reward model
        old_reward_shape = mix_batch["rewards"].size()
        pred_rewards = self.reward.get_reward(mix_batch["observations"], mix_batch["actions"])
        mix_batch["rewards"] = torch.from_numpy(pred_rewards).to(mix_batch["rewards"].device)
        assert mix_batch["rewards"].size() == old_reward_shape, "wrong reward shape!"
        
        return super().learn(mix_batch)
        
        
    def update_dynamics(
        self,
        batch: Tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> Dict[str, float]:
        """Updates the dynamics model(s) on a batch of real data (can be hybrid)."""
        observations, actions, next_observations = batch
        
        # compute loss
        obs_act = np.concatenate([observations, actions], axis=-1)
        obs_act = self.dynamics.scaler.transform(obs_act)
        diff_mean, logvar = self.dynamics.model(obs_act)
        diff_actual = torch.from_numpy(next_observations - observations).to(diff_mean.device)
        
        inv_var = torch.exp(-logvar)
        mse_loss = (torch.pow(diff_mean - diff_actual, 2) * inv_var).mean(dim=(1, 2))
        
        # optimize
        self.dynamics_optim.zero_grad()
        mse_loss.backward()
        self.dynamics_optim.step()
        
        return {"dynamics_update/mse_loss": mse_loss.detach().cpu().item()}
        
        
    def update_reward(
        self,
        preference_batch: Dict[str, torch.Tensor],
        normalize_input: bool = True
    ) -> Dict[str, float]:
        """Updates our reward model with a batch of real preference data (can be hybrid)."""
        loss = self.reward_loss(preference_batch, normalize_input=normalize_input)
        
        # optimize
        self.reward_optim.zero_grad()
        loss.backward()
        self.reward_optim.step()
        
        return {"reward_update/bce_loss": loss.detach().cpu().item()}

    
    def update_dynamics_and_reward(
        self,
        offline_buffer: ReplayBuffer,
        online_buffer: ReplayBuffer,
        offline_pref_dataset: PreferenceDataset,
        online_pref_dataset: PreferenceDataset
    ) -> Tuple[Dict[str, np.ndarray], Dict]:
        """Updates both the dynamics and the reward (i.e. whole world model) over some number of update steps."""
        
        all_loss_info = {
            "dynamics_update/mse_loss": 0, 
            "reward_update/bce_loss": 0
        }
        self.dynamics.model.train()
        self.reward.model.train()
        
        num_update_steps = online_buffer._size // self.online_batch_size
        for _ in range(num_update_steps):
            online_batch = online_buffer.sample(self.online_batch_size)
            offline_batch = offline_buffer.sample(self.online_batch_size * int(1 / self.online_to_total_ratio - 1))
            online_pref_batch = online_pref_dataset.sample(self.online_pref_batch_size)
            offline_pref_batch = offline_pref_dataset.sample(self.online_pref_batch_size * int(1 / self.online_to_total_ratio - 1))
            
            hybrid_batch = {k: torch.cat([online_batch[k], offline_batch[k]]).cpu().numpy() for k in online_batch.keys()}
            hybrid_pref_batch = {k: torch.cat([online_pref_batch[k], offline_pref_batch[k]]) for k in online_pref_batch.keys()}
            
            dynamics_info = self.update_dynamics(hybrid_batch)
            reward_info = self.update_reward(hybrid_pref_batch)
            mdp_info = {**dynamics_info, **reward_info}
            
            for _k in mdp_info:
                all_loss_info[_k] += mdp_info[_k]
        
        self.dynamics.model.eval()
        self.reward.model.eval()
        return {_key: _value / num_update_steps for _key, _value in all_loss_info.items()}