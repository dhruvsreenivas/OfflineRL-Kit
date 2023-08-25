import time
import os

import numpy as np
import torch
import gym

from typing import Optional, Dict, List, Tuple
from tqdm import tqdm
from collections import deque
from offlinerlkit.buffer import ReplayBuffer, PreferenceDataset
from offlinerlkit.utils.logger import Logger
from offlinerlkit.policy import BasePolicy


# model-based policy trainer with preference-based reward learning
class HybridPrefMBPolicyMppiSacTrainer:
    def __init__(
        self,
        policy: BasePolicy,
        eval_env: gym.Env,
        offline_preference_dataset: PreferenceDataset,
        online_preference_dataset: PreferenceDataset,
        real_buffer: ReplayBuffer,
        fake_buffer: ReplayBuffer,
        online_buffer: ReplayBuffer,
        logger: Logger,
        rollout_setting: Tuple[int, int, int],
        epoch: int = 1000,
        step_per_epoch: int = 1000,
        batch_size: int = 256,
        real_ratio: float = 0.05,
        eval_episodes: int = 10,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        dynamics_update_freq: int = 0,
        gamma: float = 1.0,
        segment_length: float = 60,
        n_samples: int = 25,
        timesteps: int = 20,
        noise_mu: float = 0,
        noise_std: float = 1,
        lambda_: float = 1,
    ) -> None:
        self.policy = policy
        self.eval_env = eval_env
        self.offline_preference_dataset = offline_preference_dataset
        self.online_preference_dataset = online_preference_dataset
        self.real_buffer = real_buffer
        self.fake_buffer = fake_buffer
        self.online_buffer = online_buffer
        self.logger = logger
        self.device = self.offline_preference_dataset.sample(1)['observations1'].device

        self._rollout_freq, self._rollout_batch_size, \
            self._rollout_length = rollout_setting
        self._dynamics_update_freq = dynamics_update_freq

        self._epoch = epoch
        self._step_per_epoch = step_per_epoch
        self._batch_size = batch_size
        self._real_ratio = real_ratio
        self._eval_episodes = eval_episodes
        self.lr_scheduler = lr_scheduler
        self.gamma = gamma
        self.segment_length = segment_length
        # MPPI parameters
        self.n_samples = n_samples
        self.timesteps = timesteps
        self.noise_mu = noise_mu
        self.noise_std = noise_std
        self.lambda_ = lambda_

    def train(self) -> Dict[str, float]:
        start_time = time.time()

        num_timesteps = 0
        last_10_performance = deque(maxlen=10)
        
        # train loop
        for e in range(1, self._epoch + 1):

            self.policy.train()

            pbar = tqdm(range(self._step_per_epoch), desc=f"Epoch #{e}/{self._epoch}")
            for it in pbar:
                if num_timesteps % self._rollout_freq == 0:
                    init_obss = self.real_buffer.sample(self._rollout_batch_size)["observations"].cpu().numpy()
                    rollout_transitions, rollout_info = self.policy.rollout(init_obss, self._rollout_length)
                    self.fake_buffer.add_batch(**rollout_transitions)
                    self.logger.log(
                        "num rollout transitions: {}, reward mean: {:.4f}".\
                            format(rollout_info["num_transitions"], rollout_info["reward_mean"])
                    )
                    for _key, _value in rollout_info.items():
                        self.logger.logkv_mean("rollout_info/"+_key, _value)

                real_sample_size = int(self._batch_size * self._real_ratio)
                fake_sample_size = self._batch_size - real_sample_size
                real_batch = self.real_buffer.sample(batch_size=real_sample_size)
                # update real_batch reward
                fake_batch = self.fake_buffer.sample(batch_size=fake_sample_size)
                batch = {"real": real_batch, "fake": fake_batch}
                loss = self.policy.learn(batch)
                pbar.set_postfix(**loss)

                for k, v in loss.items():
                    self.logger.logkv_mean(k, v)
                
                # update the dynamics if necessary
                if 0 < self._dynamics_update_freq and (num_timesteps+1)%self._dynamics_update_freq == 0:
                    self._add_online_data(
                        preference_batch_size=self.policy._online_preference_ratio * self.policy._reward_batch_size,
                        segment_length=self.segment_length, 
                        device=self.device)
                    dynamics_update_info = self.policy.update_dynamics_and_reward(self.real_buffer, self.online_buffer, self.offline_preference_dataset, self.online_preference_dataset)
                    for k, v in dynamics_update_info.items():
                        self.logger.logkv_mean(k, v)
                
                num_timesteps += 1

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            
            # evaluate current policy
            eval_info = self._evaluate()
            ep_reward_mean, ep_reward_std = np.mean(eval_info["eval/episode_reward"]), np.std(eval_info["eval/episode_reward"])
            ep_length_mean, ep_length_std = np.mean(eval_info["eval/episode_length"]), np.std(eval_info["eval/episode_length"])
            norm_ep_rew_mean = self.eval_env.get_normalized_score(ep_reward_mean) * 100
            norm_ep_rew_std = self.eval_env.get_normalized_score(ep_reward_std) * 100
            last_10_performance.append(norm_ep_rew_mean)
            self.logger.logkv("eval/normalized_episode_reward", norm_ep_rew_mean)
            self.logger.logkv("eval/normalized_episode_reward_std", norm_ep_rew_std)
            self.logger.logkv("eval/episode_length", ep_length_mean)
            self.logger.logkv("eval/episode_length_std", ep_length_std)
            self.logger.set_timestep(num_timesteps)
            self.logger.dumpkvs(exclude=["dynamics_training_progress"])
        
            # save checkpoint
            torch.save(self.policy.state_dict(), os.path.join(self.logger.checkpoint_dir, "policy.pth"))

        self.logger.log("total time: {:.2f}s".format(time.time() - start_time))
        torch.save(self.policy.state_dict(), os.path.join(self.logger.model_dir, "policy.pth"))
        self.policy.dynamics.save(self.logger.model_dir)
        self.logger.close()
    
        return {"last_10_performance": np.mean(last_10_performance)}

    def _evaluate(self) -> Dict[str, List[float]]:
        self.policy.eval()
        obs = self.eval_env.reset()
        eval_ep_info_buffer = []
        num_episodes = 0
        episode_reward, episode_length = 0, 0

        while num_episodes < self._eval_episodes:
            action = self.get_action(obs.reshape(1, -1))
            next_obs, reward, terminal, _ = self.eval_env.step(action.flatten())
            episode_reward += reward
            episode_length += 1

            obs = next_obs

            if terminal:
                eval_ep_info_buffer.append(
                    {"episode_reward": episode_reward, "episode_length": episode_length}
                )
                num_episodes +=1
                episode_reward, episode_length = 0, 0
                obs = self.eval_env.reset()
        
        return {
            "eval/episode_reward": [ep_info["episode_reward"] for ep_info in eval_ep_info_buffer],
            "eval/episode_length": [ep_info["episode_length"] for ep_info in eval_ep_info_buffer]
        }
    
    def _add_online_data(self, preference_batch_size, segment_length, device) -> Dict[str, torch.tensor]:
        # collect 2 * batch_size trajectories in total. If collected trajectory is smaller than segment size, recollect. 
        # Add all data to self.online_buffer. Return online preference dataset with size of preference_batch_size.
        self.policy.eval()
        obs = self.eval_env.reset()
        preference_batch = {}
        obs_1, act_1, next_obs_1, term1, obs_2, act_2, next_obs_2, term2, label = [], [], [], [], [], [], [], [], []
        current_batch_size = 0

        traj = 0
        reward_1, reward_2 = 0, 0
        while current_batch_size < preference_batch_size:
            obs = self.eval_env.reset()
            obs_lst, act_lst, next_obs_lst, reward_lst, term_lst = [], [], [], [], []
            while True:
                action = self.policy.select_action(obs.reshape(1, -1), deterministic=True)
                next_obs, reward, terminal, _ = self.eval_env.step(action.flatten())
                
                obs_lst.append(obs)
                act_lst.append(action)
                next_obs_lst.append(next_obs)
                reward_lst.append(reward)
                term_lst.append(terminal)

                obs = next_obs

                if terminal:
                    ## convert data to arrays ## 
                    obs_lst = np.vstack(obs_lst).astype(np.float32) # (trajectory_length, obs_dim)
                    act_lst = np.vstack(act_lst).astype(np.float32)
                    next_obs_lst = np.vstack(next_obs_lst).astype(np.float32)
                    reward_lst = np.vstack(reward_lst).astype(np.float32)
                    term_lst = np.vstack(term_lst)
                
                    ## update tp online data buffer ## 
                    self.online_buffer.add_batch(
                        obss=obs_lst,
                        next_obss=next_obs_lst,
                        actions=act_lst,
                        rewards=reward_lst,
                        terminals=term_lst,
                    )

                    ## update to online preference dataset ##
                    # convert data to torch arrays
                    obs_lst = torch.tensor(obs_lst, device=device)
                    act_lst = torch.tensor(act_lst, device=device)
                    next_obs_lst = torch.tensor(next_obs_lst, device=device)
                    reward_lst = torch.tensor(reward_lst, device=device)
                    term_lst = torch.tensor(term_lst, device=device)

                    # if trajectory length is smaller than segment lenght, resample a trajectory
                    if obs_lst.shape[0] < segment_length:
                        break
                    # sample segment from the history and compute the reward
                    start_idx = torch.randint(0, obs_lst.shape[0] - segment_length, (1,)) if obs_lst.shape[0] > segment_length else 0
                    if traj == 0: # collected the first trajectory, need to collect a second one for comparison
                        obs_1.append(obs_lst[start_idx : start_idx + segment_length])
                        act_1.append(act_lst[start_idx : start_idx + segment_length])
                        next_obs_1.append(next_obs_lst[start_idx : start_idx + segment_length])
                        term1.append(term_lst[start_idx : start_idx + segment_length])
                        # calculate discounted reward of the segment
                        discount = (self.gamma ** torch.arange(segment_length))
                        reward_1 = (reward_lst[start_idx : start_idx + segment_length] * discount.to(device)).sum()
                        traj = 1
                    else: # collected 2 trajectories, do comparison and add to preference dataset
                        obs_2.append(obs_lst[start_idx : start_idx + segment_length])
                        act_2.append(act_lst[start_idx : start_idx + segment_length])
                        next_obs_2.append(next_obs_lst[start_idx : start_idx + segment_length])
                        term2.append(term_lst[start_idx : start_idx + segment_length])
                        # calculate discounted reward of the segment
                        discount = (self.gamma ** torch.arange(segment_length))
                        reward_2 = (reward_lst[start_idx : start_idx + segment_length] * discount.to(device)).sum()
                        # add label
                        if reward_1 > reward_2:
                            label.append(torch.tensor(1, device=device))
                        else:
                            label.append(torch.tensor(0, device=device))
                        traj = 0
                        current_batch_size += 1 # finish collecting two trajectories and the label
                    break
        # stack observations into size (online_buffer_size, obs_dim)
        preference_batch["observations1"] = torch.stack(obs_1, dim=0).squeeze()
        preference_batch["actions1"] = torch.stack(act_1, dim=0).squeeze()
        preference_batch["next_observations1"] = torch.stack(next_obs_1, dim=0).squeeze()
        preference_batch["terminals1"] = torch.stack(term1, dim=0).squeeze()
        preference_batch["observations2"] = torch.stack(obs_2, dim=0).squeeze()
        preference_batch["actions2"] = torch.stack(act_2, dim=0).squeeze()
        preference_batch["next_observations2"] = torch.stack(next_obs_2, dim=0).squeeze()
        preference_batch["terminals2"] = torch.stack(term2, dim=0).squeeze()
        preference_batch["label"] = torch.stack(label, dim=0).squeeze()
        self.online_preference_dataset.add_batch(**preference_batch)
        return
    
    def get_action(self, init_obs):
        init_obs = init_obs.astype(np.float32)
        rollout_transitions, _ = self.policy.rollout(init_obs, self.timesteps)
        action_sequence = np.array(rollout_transitions["actions"])
        action_dim = action_sequence.shape[1]
        noise = np.random.normal(
            loc=self.noise_mu, 
            scale=self.noise_std,
            size=(self.n_samples, self.timesteps, action_dim)
        )
        cost_total = np.zeros((self.n_samples))
        action_lst = np.zeros((self.n_samples, action_dim))

        for sample in range(self.n_samples):
            observations = init_obs
            for timestep in range(action_sequence.shape[0]):
                actions = noise[sample, timestep] + action_sequence[timestep]
                actions = actions.reshape(1, actions.shape[0])
                next_observations, _, _, _ = self.policy.dynamics.step(observations, actions)
                rewards = self.policy.reward.get_reward(obs=observations.astype(np.float32), action=actions.astype(np.float32))
                cost_total[sample] += np.squeeze(rewards)
                observations = next_observations.astype(np.float32)
                if timestep == 0:
                    action_lst[sample] = actions

        beta = np.min(cost_total)
        cost_total_non_zero = np.exp(-1 / self.lambda_ * (cost_total - beta))
        eta = np.sum(cost_total_non_zero)
        omega = 1/eta * cost_total_non_zero
        
        final_action = np.sum(omega.reshape(omega.shape[0], 1) * action_lst, axis=0)
        return final_action