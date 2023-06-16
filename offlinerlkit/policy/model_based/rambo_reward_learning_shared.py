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
from offlinerlkit.modules import EnsembleDynamicsModel, EnsembleDynamicsModelWithSeparateReward


class RAMBORewardLearningSharedPolicy(MOPOPolicy):
    """
    RAMBO-RL: Robust Adversarial Model-Based Offline Reinforcement Learning <Ref: https://arxiv.org/abs/2204.12581>
    Learns reward adversarially from preference dataset along with dynamics in normal fashion.
    Done in shared fashion (dynamics + reward is one network with different heads).
    """

    def __init__(
        self,
        dynamics_and_reward: BaseDynamics,
        actor: nn.Module,
        critic1: nn.Module,
        critic2: nn.Module,
        actor_optim: torch.optim.Optimizer,
        critic1_optim: torch.optim.Optimizer,
        critic2_optim: torch.optim.Optimizer,
        dynamics_reward_adv_optim: torch.optim.Optimizer,
        tau: float = 0.005,
        gamma: float = 0.99,
        alpha: Union[float, Tuple[float, torch.Tensor, torch.optim.Optimizer]] = 0.2,
        reward_loss_coef: float = 1.0,
        normalize_reward_train: bool = False,
        normalize_reward_eval: bool = True,
        adv_weight: float = 0,
        adv_train_steps: int = 1000,
        adv_rollout_batch_size: int = 256,
        reward_batch_size: int = 20,
        adv_rollout_length: int = 5,
        include_ent_in_adv: bool = False,
        scaler: StandardScaler = None,
        device="cpu"
    ) -> None:
        super().__init__(
            dynamics_and_reward, 
            actor,
            critic1,
            critic2,
            actor_optim,
            critic1_optim,
            critic2_optim,
            tau=tau,
            gamma=gamma,
            alpha=alpha
        )
        
        self._dynamics_reward_adv_optim = dynamics_reward_adv_optim
        self._reward_loss_coef = reward_loss_coef
        self._normalize_reward_train = normalize_reward_train
        self._normalize_reward_eval = normalize_reward_eval
        self._adv_weight = adv_weight
        self._adv_train_steps = adv_train_steps
        self._adv_rollout_batch_size = adv_rollout_batch_size
        self._reward_batch_size = reward_batch_size
        self._adv_rollout_length = adv_rollout_length
        self._include_ent_in_adv = include_ent_in_adv
        self.scaler = scaler
        self.device = device
        
    def load(self, path):
        self.load_state_dict(torch.load(os.path.join(path, "rambo_reward_learn_shared_pretrain.pth"), map_location=self.device))

    def pretrain(self, data: Dict, n_epoch, batch_size, lr, logger) -> None:
        self._bc_optim = torch.optim.Adam(self.actor.parameters(), lr=lr)
        observations = data["observations"]
        actions = data["actions"]
        sample_num = observations.shape[0]
        idxs = np.arange(sample_num)

        logger.log("=== Pretraining policy ===")
        self.actor.train()
        for i_epoch in range(n_epoch):
            np.random.shuffle(idxs)
            sum_loss = 0
            for i_batch in range(sample_num // batch_size):
                batch_obs = observations[i_batch * batch_size: (i_batch + 1) * batch_size]
                batch_act = actions[i_batch * batch_size: (i_batch + 1) * batch_size]
                batch_obs = torch.from_numpy(batch_obs).to(self.device)
                batch_act = torch.from_numpy(batch_act).to(self.device)
                dist = self.actor(batch_obs)
                pred_actions, _ = dist.rsample()
                bc_loss = ((pred_actions - batch_act) ** 2).mean()

                self._bc_optim.zero_grad()
                bc_loss.backward()
                self._bc_optim.step()
                sum_loss += bc_loss.cpu().item()
            print(f"Epoch {i_epoch + 1}: mean bc loss {sum_loss/i_batch}")
        torch.save(self.state_dict(), os.path.join(logger.model_dir, "rambo_reward_learn_shared_pretrain.pth"))

    def update_dynamics_and_reward(
        self,
        real_buffer,
        preference_buffer
    ) -> Tuple[Dict[str, np.ndarray], Dict]:
        all_loss_info = {
            "adv_dynamics_update/all_loss": 0, 
            "adv_dynamics_update/sl_loss": 0, 
            "adv_dynamics_update/adv_loss": 0, 
            "adv_dynamics_update/adv_advantage": 0, 
            "adv_dynamics_update/adv_log_prob": 0,
            "adv_reward_update/reward_bce_loss": 0
        }
        # set in training model
        self.dynamics.model.train()
        
        steps = 0
        while steps < self._adv_train_steps:
            # select initial states to roll out from
            init_obss = real_buffer.sample(self._adv_rollout_batch_size)["observations"].cpu().numpy()
            observations = init_obss
            for t in range(self._adv_rollout_length):
                # get policy actions
                actions = super().select_action(observations)
                
                # real observations
                sl_observations, sl_actions, sl_next_observations = \
                    itemgetter("observations", "actions", "next_observations")(real_buffer.sample(self._adv_rollout_batch_size))
                    
                # gather new batch of offline data and update the model
                offline_batch = (observations, actions, sl_observations, sl_actions, sl_next_observations)
                preference_batch = preference_buffer.sample(self._reward_batch_size)
                next_observations, terminals, loss_info = self.dynamics_step_and_forward(offline_batch, preference_batch, self._normalize_reward_train)
                for _key in loss_info:
                    all_loss_info[_key] += loss_info[_key]
                # nonterm_mask = (~terminals).flatten()
                steps += 1
                # observations = next_observations[nonterm_mask]
                observations = next_observations.copy()
                # if nonterm_mask.sum() == 0:
                    # break
                if steps == 1000:
                    break
        
        self.dynamics.model.eval()
        return {_key: _value/steps for _key, _value in all_loss_info.items()}

    def dynamics_step_and_forward(
        self,
        offline_batch: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        preference_batch: Dict[str, torch.Tensor],
        normalize_reward: bool
    ):
        observations, actions, sl_observations, sl_actions, sl_next_observations = offline_batch
        obs_act = np.concatenate([observations, actions], axis=-1)
        obs_act = self.dynamics.scaler.transform(obs_act)
        diff_mean, logvar, rewards = self.dynamics.model(obs_act) # here we have to assume dynamics.model is EnsembleDynamicsModelWithSeparateReward
        observations = torch.from_numpy(observations).to(diff_mean.device)
        actions = torch.from_numpy(actions).to(diff_mean.device)
        
        # outputs
        diff_obs = diff_mean # in this case we don't care about reward, (n_ensemble, batch_size, state_dim)
        mean = diff_obs + observations
        std = torch.sqrt(torch.exp(logvar))
        
        dist = torch.distributions.Normal(mean, std)
        ensemble_sample = dist.sample()
        ensemble_size, batch_size, _ = ensemble_sample.shape
        
        # select the next observations and get rewards from reward model as opposed to shared dynamics model
        selected_indexes = self.dynamics.model.random_elite_idxs(batch_size)
        sample = ensemble_sample[selected_indexes, np.arange(batch_size)]
        next_observations = sample
        terminals = self.dynamics.terminal_fn(observations.detach().cpu().numpy(), actions.detach().cpu().numpy(), next_observations.detach().cpu().numpy())

        # compute logprob
        log_prob = dist.log_prob(sample).sum(-1, keepdim=True)
        log_prob = log_prob[self.dynamics.model.elites.data, ...]
        prob = log_prob.double().exp()
        prob = prob * (1/len(self.dynamics.model.elites.data))
        log_prob = prob.sum(0).log().type(torch.float32)

        # compute the advantage
        with torch.no_grad():
            next_actions, next_policy_log_prob = self.actforward(next_observations, deterministic=True)
            next_q = torch.minimum(
                self.critic1(next_observations, next_actions), 
                self.critic2(next_observations, next_actions)
            )
            if self._include_ent_in_adv:
                next_q = next_q - self._alpha * next_policy_log_prob

            value = rewards + (1 - torch.from_numpy(terminals).to(mean.device).float()) * self._gamma * next_q

            value_baseline = torch.minimum(
                self.critic1(observations, actions), 
                self.critic2(observations, actions)
            )
            advantage = value - value_baseline
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-6)
        adv_loss = (log_prob * advantage).mean()

        # compute the supervised loss
        sl_input = torch.cat([sl_observations, sl_actions], dim=-1).cpu().numpy()
        sl_target = sl_next_observations - sl_observations # we're only computing s' - s here, no reward term!
        sl_input = self.dynamics.scaler.transform(sl_input)
        if isinstance(self.dynamics.model, EnsembleDynamicsModel):
            sl_mean, sl_logvar = self.dynamics.model(sl_input)
        else:
            sl_mean, sl_logvar, _ = self.dynamics.model(sl_input)
        
        sl_inv_var = torch.exp(-sl_logvar)
        sl_mse_loss_inv = (torch.pow(sl_mean - sl_target, 2) * sl_inv_var).mean(dim=(1, 2))
        sl_var_loss = sl_logvar.mean(dim=(1, 2))
        sl_loss = sl_mse_loss_inv.sum() + sl_var_loss.sum()
        sl_loss = sl_loss + self.dynamics.model.get_decay_loss()
        sl_loss = sl_loss + 0.001 * self.dynamics.model.max_logvar.sum() - 0.001 * self.dynamics.model.min_logvar.sum()
        
        # add reward loss here on preference batch to add to supervised loss
        if isinstance(self.dynamics.model, EnsembleDynamicsModelWithSeparateReward):
            reward_loss = self.reward_loss(preference_batch, normalize_reward)
            sl_loss = sl_loss + self._reward_loss_coef * reward_loss
        
        all_loss = self._adv_weight * adv_loss + sl_loss
        
        # optimization
        self._dynamics_reward_adv_optim.zero_grad()
        all_loss.backward()
        self._dynamics_reward_adv_optim.step()

        return next_observations.cpu().numpy(), terminals, {
            "adv_dynamics_update/all_loss": all_loss.cpu().item(), 
            "adv_dynamics_update/sl_loss": sl_loss.cpu().item(), 
            "adv_dynamics_update/adv_loss": adv_loss.cpu().item(), 
            "adv_dynamics_update/adv_advantage": advantage.mean().cpu().item(), 
            "adv_dynamics_update/adv_log_prob": log_prob.mean().cpu().item(), 
            "adv_reward_update/reward_bce_loss": reward_loss.cpu().item()
        }
        
    def reward_loss(self, preference_batch: Dict[str, torch.Tensor], normalize_reward: bool) -> torch.Tensor:
        obs_actions1 = torch.cat([preference_batch["observations1"], preference_batch["actions1"]], dim=-1)
        obs_actions2 = torch.cat([preference_batch["observations2"], preference_batch["actions2"]], dim=-1)
        
        _, _, ensemble_pred_rew1 = self.dynamics.model(obs_actions1)
        _, _, ensemble_pred_rew2 = self.dynamics.model(obs_actions2)
        
        if normalize_reward:
            ensemble_pred_rew1 = ensemble_pred_rew1 / (ensemble_pred_rew1.std((1, 2), keepdim=True) + 1e-8)
            ensemble_pred_rew2 = ensemble_pred_rew2 / (ensemble_pred_rew2.std((1, 2), keepdim=True) + 1e-8)
        
        # convert to float64 to avoid infs
        ensemble_pred_rew1 = ensemble_pred_rew1.to(dtype=torch.float64)
        ensemble_pred_rew2 = ensemble_pred_rew2.to(dtype=torch.float64)
        
        ensemble_pred_rew1 = ensemble_pred_rew1.sum(2).squeeze().exp() # size (n_ensemble, batch_size) -> sum(\hat{r}(\tau1))
        ensemble_pred_rew2 = ensemble_pred_rew2.sum(2).squeeze().exp() # size (n_ensemble, batch_size) -> sum(\hat{r}(\tau2))
        assert (ensemble_pred_rew1 >= 0).all()
        assert (ensemble_pred_rew2 >= 0).all()
        assert not torch.isinf(ensemble_pred_rew1).any()
        assert not torch.isinf(ensemble_pred_rew2).any()
        
        # predicted reward
        ensemble_pred_rewsum = ensemble_pred_rew1 + ensemble_pred_rew2
        label_preds = ensemble_pred_rew1 / ensemble_pred_rewsum # for normalization it is not necessary cuz everything is > 0
        
        # ground truth label from preference dataset
        label_gt = preference_batch["label"].tile(self.dynamics.model.num_ensemble, 1).to(dtype=torch.float64) # (num_ensemble,)
        loss = F.binary_cross_entropy(label_preds, label_gt)
        return loss

    def rollout(
        self,
        init_obss: np.ndarray,
        rollout_length: int
    ) -> Tuple[Dict[str, np.ndarray], Dict]:

        num_transitions = 0
        rewards_arr = np.array([])
        rollout_transitions = defaultdict(list)

        # rollout
        observations = init_obss
        for _ in range(rollout_length):
            actions = super().select_action(observations)
            next_observations, rewards, terminals, info = self.dynamics.step(observations, actions, self._normalize_reward_eval)
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

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        if self.scaler is not None:
            obs = self.scaler.transform(obs)
        return super().select_action(obs, deterministic)