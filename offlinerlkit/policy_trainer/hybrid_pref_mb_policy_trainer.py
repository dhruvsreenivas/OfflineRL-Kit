import time
import os

import numpy as np
import torch
import gym

from typing import Optional, Dict, List, Tuple
from tqdm import tqdm
from collections import deque
from offlinerlkit.buffer import ReplayBuffer, PreferenceDataset, TrajectoryBuffer
from offlinerlkit.utils.logger import Logger
from offlinerlkit.policy import HybridRAMBORewardLearningPolicy


# model-based policy trainer with preference-based reward learning
class HybridPrefMBPolicyTrainer:
    def __init__(
        self,
        policy: HybridRAMBORewardLearningPolicy,
        eval_env: gym.Env,
        offline_preference_dataset: PreferenceDataset,
        online_trajectory_buffer: TrajectoryBuffer,
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
        device: str = 'cpu',
        num_online_traj_batch_size: int = 1,
        compare_with_latest_segment: bool = False,
    ) -> None:
        """Hybrid preference-based model-based RL trainer."""
        self.policy = policy
        self.eval_env = eval_env
        self.offline_preference_dataset = offline_preference_dataset
        self.online_trajectory_buffer = online_trajectory_buffer
        self.offline_transition_buffer = real_buffer
        self.fake_buffer = fake_buffer
        self.online_transition_buffer = online_buffer
        self.logger = logger

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
        self.device = device
        self._num_online_traj_batch_size = num_online_traj_batch_size
        self._compare_with_latest_segment = compare_with_latest_segment

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
                    init_obss = self.offline_transition_buffer.sample(self._rollout_batch_size)["observations"].cpu().numpy()
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
                real_batch = self.offline_transition_buffer.sample(batch_size=real_sample_size)
                # update real_batch reward
                fake_batch = self.fake_buffer.sample(batch_size=fake_sample_size)
                batch = {"real": real_batch, "fake": fake_batch}
                loss = self.policy.learn(batch)
                pbar.set_postfix(**loss)

                for k, v in loss.items():
                    self.logger.logkv_mean(k, v)
                
                # update the dynamics if necessary
                if 0 < self._dynamics_update_freq and (num_timesteps+1)%self._dynamics_update_freq == 0:
                    self._add_online_data(num_trajs=self._num_online_traj_batch_size)
                    if self._compare_with_latest_segment:
                        # use the trajectories from the latest policy model
                        # the trajectories is appened to the end of trajectory buffer every time we collect online data
                        # so we sample from the last ${self._num_online_traj_batch_size} idxes, e.g. [-5, -4, -3, -2, -1] if we collect 5 trajectories at a time
                        latest_traj_idx = np.arange(-self._num_online_traj_batch_size, 0) % self.online_trajectory_buffer.num_trajs
                    else:
                        latest_traj_idx = None
                    online_preference_dataset = self.online_trajectory_buffer.sample_snippet_fixed_preference_dataset(
                        num_pairs=self.online_trajectory_buffer.num_trajs,
                        sample_label=False,
                        mandatory_idx=latest_traj_idx
                    )
                    
                    dynamics_update_info = self.policy.update_dynamics_and_reward(
                        offline_transition_buffer=self.offline_transition_buffer,
                        online_transition_buffer=self.online_transition_buffer,
                        offline_preference_buffer=self.offline_preference_dataset,
                        online_preference_buffer=online_preference_dataset
                    )
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
            action = self.policy.select_action(obs.reshape(1, -1), deterministic=True)
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
    
    def _add_online_data(self, num_trajs) -> Dict[str, torch.Tensor]:
        # collect num_trajs trajectories in total. If collected trajectory is smaller than segment size, recollect. 
        # Add all data to self.online_buffer. Add all trajectories into self.online_trajectory_buffer
        
        self.policy.eval()
        obs = self.eval_env.reset()
        collected_num_trajs = 0

        while collected_num_trajs < num_trajs:
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
                
                    ## update tp online transition buffer ## 
                    self.online_transition_buffer.add_batch(
                        obss=obs_lst,
                        next_obss=next_obs_lst,
                        actions=act_lst,
                        rewards=reward_lst,
                        terminals=term_lst,
                    )

                    ## update to online trajectory buffer ##
                    self.online_trajectory_buffer.add_batch(
                        observations=obs_lst,
                        next_observations=next_obs_lst,
                        actions=act_lst,
                        rewards=reward_lst,
                        terminals=term_lst.squeeze(),
                    )

                    collected_num_trajs += 1
                    break
        return