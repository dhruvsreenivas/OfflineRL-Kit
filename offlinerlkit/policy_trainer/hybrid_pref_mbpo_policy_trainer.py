import time
import os

import numpy as np
import torch
import gym

from typing import Optional, Dict, List, Tuple
from tqdm import trange
from collections import deque
from offlinerlkit.buffer import ReplayBuffer, PreferenceDataset
from offlinerlkit.utils.logger import Logger
from offlinerlkit.policy import BasePolicy, SACPolicy


class HybridMBPOPolicyTrainer:
    """MBPO policy trainer, hybrid edition."""
    def __init__(
        self,
        policy: BasePolicy,
        train_env: gym.Env,
        eval_env: gym.Env,
        
        # data sources
        online_buffer: ReplayBuffer,
        online_mb_buffer: ReplayBuffer,
        offline_buffer: ReplayBuffer,
        online_preference_dataset: PreferenceDataset,
        offline_preference_dataset: PreferenceDataset,
        
        # logging
        logger: Logger,
        
        # rollout settings
        rollout_settings: Tuple[int, int, int, int, int],
        
        # training args
        epochs: int = 1000,
        steps_per_epoch: int = 1000,
        model_retain_epochs: int = 1,
        buffer_batch_size: int = 256,
        pref_batch_size: int = 256,
        init_exploration_steps: int = 0, # offline data hopefully helps overcome exploration early so we can set this to be small.
        discount_snippet_returns: bool = False,
        real_to_mb_ratio: float = 0.05,
        online_ratio: float = 0.5,
        add_data_outside_of_model_training: bool = True,
        
        # evaluation args
        eval_episodes: int = 10,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        dynamics_update_freq: int = 0
    ) -> None:
        """Initializes the trainer."""
        self.policy = policy
        self.train_env = train_env
        self.eval_env = eval_env
        
        self.online_buffer = online_buffer
        self.online_mb_buffer = online_mb_buffer
        self.offline_buffer = offline_buffer
        self.online_preference_dataset = online_preference_dataset
        self.offline_preference_dataset = offline_preference_dataset
        self.logger = logger
        
        self._rollout_batch_size, self._rollout_min_length, self._rollout_max_length, self._rollout_min_epoch, self._rollout_max_epoch = rollout_settings
        self._dynamics_update_freq = dynamics_update_freq
        
        self._epochs = epochs
        self._steps_per_epoch = steps_per_epoch
        self._model_retain_epochs = model_retain_epochs
        self._buffer_batch_size = buffer_batch_size
        self._pref_batch_size = pref_batch_size
        
        self._init_exploration_steps = init_exploration_steps
        self._segment_length = self.offline_preference_dataset[0]["observations1"].size(0) # this is of size (seg_length, obs_dim), useful for online dataset collection
        self._real_to_mb_ratio = real_to_mb_ratio
        self._online_ratio = online_ratio
        self._add_data_outside_of_model_training = add_data_outside_of_model_training
        
        assert isinstance(self.policy, SACPolicy)
        self._gamma = self.policy._gamma if discount_snippet_returns else 1
        
        self._eval_episodes = eval_episodes
        self._lr_scheduler = lr_scheduler
        
        
    def set_rollout_length(self, epoch_step):
        rollout_length = min(
            max(self._rollout_min_length + (epoch_step - self._rollout_min_epoch) / (self._rollout_max_epoch - self._rollout_min_epoch) * (self._rollout_max_length - self._rollout_min_length), self._rollout_min_length),
            self._rollout_max_length
        )
        return int(rollout_length)
    
    
    def add_online_data(self, init_exploration=True) -> None:
        """Do initial exploration steps and add to online buffers (both transition + preference)."""
        self.policy.eval()
        device = self.policy.reward.model.device

        preference_batch = {}
        obs_1, act_1, next_obs_1, term1, obs_2, act_2, next_obs_2, term2, label = [], [], [], [], [], [], [], [], []
        current_batch_size = 0
        
        traj = 0
        reward_1, reward_2 = 0, 0
        
        num_steps = self._init_exploration_steps if init_exploration else 1 # this is because at every online data collection phase, we can just sample 1 (tau1, tau2, label) and 1 transition to add online
        while current_batch_size < num_steps // self._segment_length + 1:
            # this is the number of batches that we need to get at least `num_steps` total samples into both buffers
            obs = self.train_env.reset()
            obs_lst, act_lst, next_obs_lst, reward_lst, term_lst = [], [], [], [], []
            
            # start rolling out
            terminal = False
            while not terminal:
                action = self.policy.select_action(obs.reshape(1, -1), deterministic=False)
                next_obs, reward, terminal, _ = self.train_env.step(action.flatten())
                
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
                        rewards=reward_lst, # no reward is actually used here.
                        terminals=term_lst,
                    )

                    ## update to online preference dataset ##
                    # convert data to torch arrays
                    obs_lst = torch.tensor(obs_lst, device=device)
                    act_lst = torch.tensor(act_lst, device=device)
                    next_obs_lst = torch.tensor(next_obs_lst, device=device)
                    reward_lst = torch.tensor(reward_lst, device=device)
                    term_lst = torch.tensor(term_lst, device=device)

                    # if trajectory length is smaller than segment length, resample a trajectory
                    if obs_lst.shape[0] < self._segment_length:
                        break
                    
                    # sample segment from the history and compute the reward
                    start_idx = torch.randint(0, obs_lst.shape[0] - self._segment_length, (1,)) if obs_lst.shape[0] > self._segment_length else 0
                    if traj == 0:
                        # collected the first trajectory, need to collect a second one for comparison
                        obs_1.append(obs_lst[start_idx : start_idx + self._segment_length])
                        act_1.append(act_lst[start_idx : start_idx + self._segment_length])
                        next_obs_1.append(next_obs_lst[start_idx : start_idx + self._segment_length])
                        term1.append(term_lst[start_idx : start_idx + self._segment_length])
                        
                        # calculate discounted reward of the segment
                        discount = (self._gamma ** torch.arange(self._segment_length))
                        reward_1 = (reward_lst[start_idx : start_idx + self._segment_length] * discount.to(device)).sum()
                        traj = 1
                    else:
                        # collected 2 trajectories, do comparison and add to preference dataset
                        obs_2.append(obs_lst[start_idx : start_idx + self._segment_length])
                        act_2.append(act_lst[start_idx : start_idx + self._segment_length])
                        next_obs_2.append(next_obs_lst[start_idx : start_idx + self._segment_length])
                        term2.append(term_lst[start_idx : start_idx + self._segment_length])
                        
                        # calculate discounted reward of the segment
                        discount = (self._gamma ** torch.arange(self._segment_length))
                        reward_2 = (reward_lst[start_idx : start_idx + self._segment_length] * discount.to(device)).sum()
                        
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
        
        
    def resize_mb_buffer(self, rollout_length: int) -> None:
        """Resizes the model's online replay buffer to account for new model length.
        Does something different from the original MBPO implementation in accordance to what MOPO did.
        """
        new_buffer_size = self._rollout_batch_size * rollout_length * self._model_retain_epochs

        all_samples = self.online_mb_buffer.sample_all()
        new_buffer = ReplayBuffer(
            new_buffer_size,
            self.online_mb_buffer.obs_shape,
            self.online_mb_buffer.obs_dtype,
            self.online_mb_buffer.action_dim,
            self.online_mb_buffer.action_dtype,
            self.online_mb_buffer.device
        )
        new_buffer.add_batch(
            obss=all_samples["observations"],
            next_obss=all_samples["next_observations"],
            actions=all_samples["actions"],
            rewards=all_samples["rewards"],
            terminals=all_samples["terminals"]
        )
        self.online_mb_buffer = new_buffer
        
    def train(self) -> Dict[str, float]:
        # fill up buffer with initial random exploration samples
        self.add_online_data(init_exploration=True)
        start_time = time.time()
        
        num_timesteps = 0
        rollout_length = 1
        last_10_performance = deque(maxlen=10)
        
        # now start training
        for e in range(1, self._epochs + 1):
            self.policy.train()
            curr_state = self.train_env.reset()
            
            pbar = trange(self._steps_per_epoch, desc=f"Epoch #{e}/{self._epochs}")
            for it in pbar:
                
                # update the dynamics if necessary
                if 0 < self._dynamics_update_freq and num_timesteps % self._dynamics_update_freq == 0:
                    self.add_online_data(init_exploration=False)
                    dynamics_update_info = self.policy.update_dynamics_and_reward(
                        self.offline_buffer,
                        self.online_buffer,
                        self.offline_preference_dataset,
                        self.online_preference_dataset
                    )
                    for k, v in dynamics_update_info.items():
                        self.logger.logkv_mean(k, v)
                        
                    # set new rollout length like in original MBPO
                    new_rollout_length = self.set_rollout_length(e - 1)
                    if new_rollout_length != rollout_length:
                        rollout_length = new_rollout_length
                        self.resize_mb_buffer(rollout_length)
                        
                    # rollout in new model from real buffer states
                    init_obss = self.online_buffer.sample(self._rollout_batch_size)["observations"].cpu().numpy()
                    rollout_transitions, rollout_info = self.policy.rollout(init_obss, rollout_length)
                    self.online_mb_buffer.add_batch(**rollout_transitions)
                    self.logger.log(
                        "num rollout transitions: {}, reward mean: {:.4f}".\
                            format(rollout_info["num_transitions"], rollout_info["reward_mean"])
                    )
                    for _key, _value in rollout_info.items():
                        self.logger.logkv_mean("rollout_info/"+_key, _value)
                        
                        
                # rollout policy in real env, get reward from learned reward model (as there is nothing else to really do)
                if self._add_data_outside_of_model_training:
                    action = self.policy.select_action(curr_state.reshape(1, -1), deterministic=False).squeeze()
                    next_state, _, terminal, _ = self.train_env.step(action)
                    self.online_buffer.add(curr_state, next_state, action, 0, terminal)
                
                # update the policy with hybrid batch
                real_batch_size = int(self._real_to_mb_ratio * self._buffer_batch_size)
                model_batch_size = self._buffer_batch_size - real_batch_size
                offline_batch_size = self._online_ratio * self._buffer_batch_size
                
                online_batch = self.online_buffer.sample(real_batch_size)
                offline_batch = self.offline_buffer.sample(offline_batch_size)
                try:
                    mb_batch = self.online_mb_buffer.sample(model_batch_size)
                except ValueError:
                    mb_batch = None
                
                hybrid_batch = {
                    "real_online": online_batch,
                    "real_offline": offline_batch,
                    "fake": mb_batch
                }
                loss = self.policy.learn(hybrid_batch)
                pbar.set_postfix(**loss)
                
                # increase number of timesteps where training took place.
                num_timesteps += 1
            
            # step learning rate scheduler if there
            if self._lr_scheduler is not None:
                self._lr_scheduler.step()
                
            
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
        self.policy.reward.save(self.logger.model_dir)
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
                num_episodes += 1
                episode_reward, episode_length = 0, 0
                obs = self.eval_env.reset()
        
        return {
            "eval/episode_reward": [ep_info["episode_reward"] for ep_info in eval_ep_info_buffer],
            "eval/episode_length": [ep_info["episode_length"] for ep_info in eval_ep_info_buffer]
        }