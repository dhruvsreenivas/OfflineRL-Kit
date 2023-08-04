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


class RAMBORewardLearningPolicy(MOPOPolicy):
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
        scaler: StandardScaler = None,
        device="cpu"
    ) -> None:
        super().__init__(
            dynamics, 
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
        self.reward = reward
        
        self._dynamics_adv_optim = dynamics_adv_optim
        self._reward_adv_optim = reward_adv_optim
        self._adv_weight = adv_weight
        self._adv_train_steps = adv_train_steps
        self._adv_rollout_batch_size = adv_rollout_batch_size
        self._reward_batch_size = reward_batch_size
        self._adv_rollout_length = adv_rollout_length
        self._sl_dynamics_loss_coef = sl_dynamics_loss_coef
        self._adv_dynamics_loss_coef = adv_dynamics_loss_coef
        self._sl_reward_loss_coef = sl_reward_loss_coef
        self._adv_reward_loss_coef = adv_reward_loss_coef
        self._include_ent_in_adv = include_ent_in_adv
        self.scaler = scaler
        self.device = device
        
        # debugging stuff
        self.count = (0, 0, 0)
        
    def load(self, path):
        self.load_state_dict(torch.load(os.path.join(path, "rambo_reward_learn_pretrain.pth"), map_location=self.device))

    def pretrain(self, data: Dict, n_epoch, batch_size, lr, logger) -> None:
        self._bc_optim = torch.optim.Adam(self.actor.parameters(), lr=lr)
        observations = data["observations"]
        actions = data["actions"]
        sample_num = observations.shape[0]
        idxs = np.arange(sample_num)

        logger.log("*** Pretraining policy ***")
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
            print(f"Epoch {i_epoch}, mean bc loss {sum_loss/i_batch}")
        torch.save(self.state_dict(), os.path.join(logger.model_dir, "rambo_reward_learn_pretrain.pth"))

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
                
                # real observations
                sl_observations, sl_actions, sl_next_observations = \
                    itemgetter("observations", "actions", "next_observations")(real_buffer.sample(self._adv_rollout_batch_size)) # again, inputs are all normalized
                    
                # gather new batch of offline data and update the model
                offline_batch = (observations, actions, sl_observations, sl_actions, sl_next_observations)
                preference_batch = preference_buffer.sample(self._reward_batch_size) # unnormalized
                next_observations, terminals, loss_info = self.dynamics_step_and_forward(offline_batch, preference_batch)
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
        
        # log counts
        # c_d, c_pi, c_eq = self.count
        # print(f"ratio of v_dataset > v_pi is {c_d / (c_d + c_pi)}")
        # print(f"number of larger v_dataset is {c_d}")
        # print(f"number of larger v_pi is {c_pi}")
        # print(f"number of tie is {c_eq}")
        
        # go back to eval mode for rollout
        self.dynamics.model.eval()
        self.reward.model.eval()
        return {_key: _value/steps for _key, _value in all_loss_info.items()}

    def dynamics_step_and_forward(
        self,
        offline_batch: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        preference_batch: Dict[str, torch.Tensor]
    ):
        observations, actions, sl_observations, sl_actions, sl_next_observations = offline_batch
        
        # scale
        obs_act = np.concatenate([observations, actions], axis=-1)
        obs_act = self.dynamics.scaler.transform(obs_act)
        diff_mean, logvar = self.dynamics.model(obs_act)
        observations = torch.from_numpy(observations).to(diff_mean.device) # normalized from before
        actions = torch.from_numpy(actions).to(diff_mean.device) # normalized from before
        
        # =================== Adversarial dynamics loss calculation ===================
        
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
        next_observations = sample # this has dynamics gradients
        if self.reward.scaler is not None:
            obs_dim, action_dim = observations.size(-1), actions.size(-1)
            pi_obs_act = torch.cat([observations, actions], dim=-1)
            pi_obs_act = self.reward.scaler.transform(pi_obs_act)
            pi_observations_reward_input, pi_actions_reward_input = torch.split(pi_obs_act, [obs_dim, action_dim], dim=-1)
        else:
            pi_observations_reward_input, pi_actions_reward_input = observations, actions
        pi_ensemble_rewards, reward_mask = self.reward.model(pi_observations_reward_input, pi_actions_reward_input) # (n_ensemble, batch_size, 1) for the moment -- this is for adversarial update
        
        # also get the elite idxs for rewards
        reward_selected_indexes = self.reward.model.random_elite_idxs(batch_size)
        rewards = pi_ensemble_rewards[reward_selected_indexes, np.arange(batch_size)]
        pi_rewards = rewards
        
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
            
            value = rewards + (1 - torch.from_numpy(terminals).to(mean.device).float()) * self._gamma * next_q # don't pass gradients of this to reward
            value_baseline = torch.minimum(
                self.critic1(observations, actions), 
                self.critic2(observations, actions)
            )
            advantage = value - value_baseline
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-6)
            
        adv_dynamics_loss = (1 / (1 - self._gamma + 1e-6)) * (log_prob * advantage).mean()
        
        # expected total return of behaviour policy under current reward model (sl_{input} are all actions from the real offline dataset)
        sl_observations = torch.from_numpy(sl_observations).to(diff_mean.device) if not torch.is_tensor(sl_observations) else sl_observations
        sl_actions = torch.from_numpy(sl_actions).to(diff_mean.device) if not torch.is_tensor(sl_actions) else sl_actions
        
        if self.reward.scaler is not None:
            obs_dim, action_dim = sl_observations.size(-1), sl_actions.size(-1)
            dataset_obs_act = torch.cat([sl_observations, sl_actions], dim=-1)
            dataset_obs_act = self.reward.scaler.transform(dataset_obs_act)
            dataset_observations_reward_input, dataset_actions_reward_input = torch.split(dataset_obs_act, [obs_dim, action_dim], dim=-1)
        else:
            dataset_observations_reward_input, dataset_actions_reward_input = sl_observations, sl_actions
        dataset_ensemble_rewards = self.reward.model(dataset_observations_reward_input, dataset_actions_reward_input, reward_mask) # (n_ensemble, batch_size, 1) for the moment -- this is for adversarial update
        
        # also get the elite idxs for rewards
        dataset_rewards = dataset_ensemble_rewards[reward_selected_indexes, np.arange(batch_size)] # (batch_size, 1)
        v_dataset = dataset_rewards.mean()
        v_pi_model = pi_rewards.mean()
        
        # total adv loss
        adv_reward_loss = (1 / (1 - self._gamma + 1e-6)) * (v_pi_model - v_dataset)
        if self.reward.model.soft_clamp_output:
            adv_reward_loss += 0.001 * self.reward.model.max_reward.sum() - 0.001 * self.reward.model.min_reward.sum()
        adv_loss = self._adv_dynamics_loss_coef * adv_dynamics_loss + self._adv_reward_loss_coef * adv_reward_loss
        
        # =================== Supervised loss here (MLE of the dynamics + the reward BCE loss) ===================

        # compute the supervised loss
        sl_input = torch.cat([sl_observations, sl_actions], dim=-1).cpu().numpy()
        sl_target = sl_next_observations - sl_observations # we're only computing s' - s here, no reward term!
        sl_input = self.dynamics.scaler.transform(sl_input)
        sl_mean, sl_logvar = self.dynamics.model(sl_input)
        sl_inv_var = torch.exp(-sl_logvar)
        sl_mse_loss_inv = (torch.pow(sl_mean - sl_target, 2) * sl_inv_var).mean(dim=(1, 2))
        sl_var_loss = sl_logvar.mean(dim=(1, 2))
        sl_loss = sl_mse_loss_inv.sum() + sl_var_loss.sum()
        sl_loss = sl_loss + self.dynamics.model.get_decay_loss()
        sl_loss = sl_loss + 0.001 * self.dynamics.model.max_logvar.sum() - 0.001 * self.dynamics.model.min_logvar.sum()
        sl_loss_dynamics = sl_loss
        
        # add reward loss here on preference batch to add to supervised loss
        normalize_input = self.reward.scaler is not None
        sl_loss_reward = self.reward_loss(preference_batch, normalize_input)
        sl_loss = self._sl_dynamics_loss_coef * sl_loss_dynamics + self._sl_reward_loss_coef * sl_loss_reward
        
        all_loss = self._adv_weight * adv_loss + sl_loss
        
        # optimization
        self._dynamics_adv_optim.zero_grad()
        self._reward_adv_optim.zero_grad()
        all_loss.backward()
        self._dynamics_adv_optim.step()
        self._reward_adv_optim.step()
        
        # log
        info_dict = {
            "adv_dynamics_update/all_loss": all_loss.cpu().item(), 
            "adv_dynamics_update/sl_loss": sl_loss.cpu().item(), 
            "adv_dynamics_update/adv_loss": adv_loss.cpu().item(), 
            "adv_dynamics_update/adv_advantage": advantage.mean().cpu().item(), 
            "adv_dynamics_update/adv_log_prob": log_prob.mean().cpu().item(), 
            "adv_reward_update/reward_bce_loss": sl_loss_reward.cpu().item(),
            "adv_update/v_pi": v_pi_model.cpu().item(), 
            "adv_update/v_dataset": v_dataset.cpu().item(), 
            "adv_update/adv_dynamics_loss": adv_dynamics_loss.cpu().item(),
            "adv_update/adv_reward_loss": adv_reward_loss.cpu().item(),
            "adv_dynamics_update/sl_loss_dynamics": sl_loss_dynamics.cpu().item(), 
        }
        
        if self.reward.model.soft_clamp_output:
            info_dict.update({
                "adv_dynamics_update/reward_max": self.reward.model.max_reward.sum().cpu().item(),
                "adv_dynamics_update/reward_min": self.reward.model.min_reward.sum().cpu().item()
            })

        return next_observations.cpu().numpy(), terminals, info_dict
        
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
            next_observations, rewards, terminals, info = self.dynamics.step(observations, actions)
            if rewards is None:
                # non-shared case, calculate rewards separately
                rewards = self.reward.get_reward(obs=observations, action=actions) # keep training flag?
            
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
    
    def learn(self, batch: Dict) -> Dict[str, float]:
        real_batch, fake_batch = batch["real"], batch["fake"]
        
        # replace real rewards with what our reward model predicts
        old_reward_shape = real_batch["rewards"].shape
        pred_rewards = self.reward.get_reward(real_batch["observations"], real_batch["actions"])
        real_batch["rewards"] = torch.from_numpy(pred_rewards).to(fake_batch["rewards"].device)
        assert real_batch["rewards"].shape == old_reward_shape, "wrong reward shape!"
        
        # now learn on the real batch + fake batch
        batch = {"real": real_batch, "fake": fake_batch}
        return super().learn(batch)
    
    
    # ====== debug functions ======
    
    
    def sanity_checking(self, real_buffer):
        all_info = real_buffer.sample_all()
        all_obs, all_actions = all_info["observations"], all_info["actions"]
        batch_size = 1000
        reward_selected_indexes = self.reward.model.random_elite_idxs(batch_size)
        
        steps = 0
        sum_v_dataset = []
        sum_v_pi = []
        while steps < int(np.ceil(all_obs.shape[0] / batch_size)):
            obs = torch.from_numpy(all_obs[steps * batch_size : (steps + 1) * batch_size]).to(self.reward.model.device)
            actions = torch.from_numpy(all_actions[steps * batch_size : (steps + 1) * batch_size]).to(self.reward.model.device)
            
            # dataset samples
            v_batch = self.reward.model(obs, actions, train=False)
            v_batch = v_batch[reward_selected_indexes, np.arange(obs.shape[0])]
            sum_v_dataset.append(v_batch.cpu().numpy())

            # policy selection
            v_pi_batch = self.reward.model(obs, super().select_action(obs, deterministic=True), train=False)
            v_pi_batch = v_pi_batch[reward_selected_indexes, np.arange(obs.shape[0])]
            sum_v_pi.append(v_pi_batch.cpu().numpy())
            
            steps += 1
        
        sum_v_dataset = np.concatenate(sum_v_dataset)
        sum_v_pi = np.concatenate(sum_v_pi)
        
        print(f"avg difference (v_dataset - v_pi): {np.mean(sum_v_dataset - sum_v_pi)}")
        print(f"fraction of times the dataset reward is bigger than the policy reward: {(sum_v_dataset > sum_v_pi).astype(np.float32).mean()}")
        print(f"fraction of times the policy reward is bigger than the dataset reward: {(sum_v_pi > sum_v_dataset).astype(np.float32).mean()}")