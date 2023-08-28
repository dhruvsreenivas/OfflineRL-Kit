import numpy as np
import torch
import torch.nn as nn
import gym
import os

from torch.nn import functional as F
from typing import Dict, Union, Tuple
from collections import defaultdict
from operator import itemgetter
from offlinerlkit.utils.scaler import StandardScaler
from offlinerlkit.policy import MOPOPolicy
from offlinerlkit.dynamics import BaseDynamics
from offlinerlkit.rewards import BaseReward
from offlinerlkit.utils.losses import ensemble_cross_entropy
from offlinerlkit.policy.model_based.rambo_reward_learning import RAMBORewardLearningPolicy


class HybridRAMBORewardLearningPolicy(RAMBORewardLearningPolicy):
    """
    RAMBO-RL: Robust Adversarial Model-Based Offline Reinforcement Learning <Ref: https://arxiv.org/abs/2204.12581>
    Learns reward adversarially from preference dataset along with dynamics in normal fashion.
    Done in separated fashion (2 completely separate neural networks).
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
        dynamics_adv_optim: torch.optim.Optimizer,
        reward_adv_optim: torch.optim.Optimizer,
        tau: float = 0.005,
        gamma: float = 0.99,
        alpha: Union[float, Tuple[float, torch.Tensor, torch.optim.Optimizer]] = 0.2,
        adv_weight: float = 0,
        adv_train_steps: int = 1000,
        adv_rollout_batch_size: int = 256,
        sl_dynamics_loss_coef: float = 1.0,
        adv_dynamics_loss_coef: float = 1.0,
        adv_reward_loss_coef: float = 1.0,
        sl_reward_loss_coef: float = 1.0,
        reward_batch_size: int = 20,
        adv_rollout_length: int = 5,
        include_ent_in_adv: bool = False,
        use_real_batch_in_policy_update: bool = True,
        scaler: StandardScaler = None,
        device="cpu",
        online_transition_ratio: float = 0.5, 
        online_preference_ratio: float = 0.5,
    ) -> None:
        super().__init__(
            dynamics=dynamics, 
            reward=reward,
            actor=actor,
            critic1=critic1,
            critic2=critic2,
            actor_optim=actor_optim,
            critic1_optim=critic1_optim,
            critic2_optim=critic2_optim,
            dynamics_adv_optim=dynamics_adv_optim,
            reward_adv_optim=reward_adv_optim,
            tau=tau,
            gamma=gamma,
            alpha=alpha,
            adv_weight=adv_weight,
            adv_train_steps=adv_train_steps,
            adv_rollout_batch_size=adv_rollout_batch_size,
            sl_dynamics_loss_coef=sl_dynamics_loss_coef,
            adv_dynamics_loss_coef=adv_dynamics_loss_coef,
            adv_reward_loss_coef=adv_reward_loss_coef,
            sl_reward_loss_coef=sl_reward_loss_coef,
            reward_batch_size=reward_batch_size,
            adv_rollout_length=adv_rollout_length,
            include_ent_in_adv=include_ent_in_adv,
            use_real_batch_in_policy_update=use_real_batch_in_policy_update,
            scaler=scaler,
            device=device,
        )
        self._online_transition_ratio = online_transition_ratio
        self._online_preference_ratio = online_preference_ratio

    def update_dynamics_and_reward(
        self,
        real_buffer,
        online_buffer,
        offline_preference_buffer,
        online_preference_buffer,
    ) -> Tuple[Dict[str, np.ndarray], Dict]:
        all_loss_info = {
            "adv_dynamics_update/all_loss": 0, 
            "adv_dynamics_update/sl_loss": 0, 
            "adv_dynamics_update/adv_loss": 0, 
            "adv_dynamics_update/adv_advantage": 0, 
            "adv_dynamics_update/adv_log_prob": 0,
            "adv_reward_update/reward_bce_loss": 0,
            "adv_update/v_pi": 0, 
            "adv_update/v_dataset": 0, 
            "adv_dynamics_update/sl_loss_dynamics": 0,
            "adv_dynamics_update/reward_max": 0,
            "adv_dynamics_update/reward_min": 0,
            "adv_update/adv_dynamics_loss": 0,
            "adv_update/adv_reward_loss": 0,
        }
        # set in training model
        self.dynamics.model.train()
        self.reward.model.train()
        
        steps = 0
        while steps < self._adv_train_steps:
            # select initial states to roll out from
            init_obss = real_buffer.sample(self._adv_rollout_batch_size)["observations"].cpu().numpy() # this is normalized, as buffer is normalized before training.
            observations = init_obss
            for t in range(self._adv_rollout_length):
                # get policy actions
                actions = super().select_action(observations)

                # get sl data
                online_buffer_size = int(self._adv_rollout_batch_size * self._online_transition_ratio)
                offline_buffer_size = self._adv_rollout_batch_size - online_buffer_size
                offline_observations, offline_actions, offline_next_observations = \
                    itemgetter("observations", "actions", "next_observations")(real_buffer.sample(offline_buffer_size))
                online_observations, online_actions, online_next_observations = \
                    itemgetter("observations", "actions", "next_observations")(online_buffer.sample(online_buffer_size))
                # mix online data with offline data
                sl_observations = torch.cat((online_observations, offline_observations), dim=0)
                sl_actions = torch.cat((online_actions, offline_actions), dim=0)
                sl_next_observations = torch.cat((online_next_observations, offline_next_observations), dim=0)

                # mix online preference buffer and offline preference buffe
                online_preference_batch_size = int(self._reward_batch_size * self._online_preference_ratio)
                offline_preference_batch_size = self._reward_batch_size - online_preference_batch_size
                online_preference_batch = online_preference_buffer.sample(online_preference_batch_size)
                offline_preference_batch = offline_preference_buffer.sample(offline_preference_batch_size)
                preference_batch = {k: torch.cat((online_preference_batch[k], offline_preference_batch[k]), dim=0) for k in online_preference_batch}

                # gather new batch of mixed data and update the model (here we sample from both transition dataset + preference dataset)
                data_batch = (observations, actions, sl_observations, sl_actions, sl_next_observations)
                next_observations, terminals, loss_info = self.dynamics_sstep_and_forward(data_batch, preference_batch)
                for _key in loss_info:
                    if 'reward_max' in _key or 'reward_min' in _key:
                        all_loss_info[_key] = loss_info[_key]
                    else:
                        all_loss_info[_key] += loss_info[_key]
                # nonterm_mask = (~terminals).flatten()
                steps += 1
                # observations = next_observations[nonterm_mask]
                observations = next_observations.copy()
                # if nonterm_mask.sum() == 0:
                    # break
                if steps == 1000:
                    break
        
        # go back to eval mode for rollout
        self.dynamics.model.eval()
        self.reward.model.eval()
        return {_key: _value/steps for _key, _value in all_loss_info.items()}