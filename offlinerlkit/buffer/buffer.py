import numpy as np
import torch

from typing import Optional, Union, Tuple, Dict
from collections import defaultdict


class ReplayBuffer:
    def __init__(
        self,
        buffer_size: int,
        obs_shape: Tuple,
        obs_dtype: np.dtype,
        action_dim: int,
        action_dtype: np.dtype,
        device: str = "cpu"
    ) -> None:
        self._max_size = buffer_size
        self.obs_shape = obs_shape
        self.obs_dtype = obs_dtype
        self.action_dim = action_dim
        self.action_dtype = action_dtype

        self._ptr = 0
        self._size = 0

        self.observations = np.zeros((self._max_size,) + self.obs_shape, dtype=obs_dtype)
        self.next_observations = np.zeros((self._max_size,) + self.obs_shape, dtype=obs_dtype)
        self.actions = np.zeros((self._max_size, self.action_dim), dtype=action_dtype)
        self.rewards = np.zeros((self._max_size, 1), dtype=np.float32)
        self.terminals = np.zeros((self._max_size, 1), dtype=np.float32)

        self.device = torch.device(device)

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        terminal: np.ndarray
    ) -> None:
        # Copy to avoid modification by reference
        self.observations[self._ptr] = np.array(obs).copy()
        self.next_observations[self._ptr] = np.array(next_obs).copy()
        self.actions[self._ptr] = np.array(action).copy()
        self.rewards[self._ptr] = np.array(reward).copy()
        self.terminals[self._ptr] = np.array(terminal).copy()

        self._ptr = (self._ptr + 1) % self._max_size
        self._size = min(self._size + 1, self._max_size)
    
    def add_batch(
        self,
        obss: np.ndarray,
        next_obss: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        terminals: np.ndarray
    ) -> None:
        batch_size = len(obss)
        indexes = np.arange(self._ptr, self._ptr + batch_size) % self._max_size

        self.observations[indexes] = np.array(obss).copy()
        self.next_observations[indexes] = np.array(next_obss).copy()
        self.actions[indexes] = np.array(actions).copy()
        self.rewards[indexes] = np.array(rewards).copy()
        self.terminals[indexes] = np.array(terminals).copy()

        self._ptr = (self._ptr + batch_size) % self._max_size
        self._size = min(self._size + batch_size, self._max_size)
    
    def load_dataset(self, dataset: Dict[str, np.ndarray]) -> None:
        observations = np.array(dataset["observations"], dtype=self.obs_dtype)
        next_observations = np.array(dataset["next_observations"], dtype=self.obs_dtype)
        actions = np.array(dataset["actions"], dtype=self.action_dtype)
        rewards = np.array(dataset["rewards"], dtype=np.float32).reshape(-1, 1)
        terminals = np.array(dataset["terminals"], dtype=np.float32).reshape(-1, 1)

        self.observations = observations
        self.next_observations = next_observations
        self.actions = actions
        self.rewards = rewards
        self.terminals = terminals

        self._ptr = len(observations)
        self._size = len(observations)
     
    def normalize_obs(self, eps: float = 1e-3) -> Tuple[np.ndarray, np.ndarray]:
        mean = self.observations.mean(0, keepdims=True)
        std = self.observations.std(0, keepdims=True) + eps
        self.observations = (self.observations - mean) / std
        self.next_observations = (self.next_observations - mean) / std
        obs_mean, obs_std = mean, std
        return obs_mean, obs_std

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:

        batch_indexes = np.random.randint(0, self._size, size=batch_size)
        
        return {
            "observations": torch.tensor(self.observations[batch_indexes]).to(self.device),
            "actions": torch.tensor(self.actions[batch_indexes]).to(self.device),
            "next_observations": torch.tensor(self.next_observations[batch_indexes]).to(self.device),
            "terminals": torch.tensor(self.terminals[batch_indexes]).to(self.device),
            "rewards": torch.tensor(self.rewards[batch_indexes]).to(self.device)
        }
    
    def sample_all(self) -> Dict[str, np.ndarray]:
        return {
            "observations": self.observations[:self._size].copy(),
            "actions": self.actions[:self._size].copy(),
            "next_observations": self.next_observations[:self._size].copy(),
            "terminals": self.terminals[:self._size].copy(),
            "rewards": self.rewards[:self._size].copy()
        }
        

class TrajectoryBuffer:
    """Offline dataset of trajectories as opposed to samples."""
    def __init__(
        self,
        dataset: Dict[str, np.ndarray],
        device: str = "cpu"       
    ) -> None:
        self.device = torch.device(device)
        
        self.obs_shape = dataset["observations"].shape[1]
        self.obs_dtype = dataset["observations"].dtype
        self.action_dim = dataset["actions"].shape[1]
        self.action_dtype = dataset["actions"].dtype
        
        # splitting into trajectories (taken from IQL repo)
        trajs = [[]]
        for i in range(len(dataset["observations"])):
            trajs[-1].append(
                (dataset["observations"][i],
                 dataset["actions"][i],
                 dataset["next_observations"][i],
                 dataset["terminals"][i],
                 dataset["rewards"][i])
            )
            
            if dataset["terminals"][i] == 1.0 and i + 1 < len(dataset["observations"]):
                trajs.append([])
                
        self.trajs = trajs # List[Tuple[np.ndarray]]
        
    @property
    def traj_lengths(self):
        return [len(traj) for traj in self.trajs]
    
    @property
    def traj_rewards(self):
        return [np.sum([traj[i][-1] for i in range(len(traj))]) for traj in self.trajs]
        
    def sample(self, num_trajs: int) -> Dict[str, np.ndarray]:
        """
        Samples a batch of trajectories.
        
        Output should be of size [num_trajs, traj_length, data_size].
        """
        idxs = np.random.randint(0, len(self.trajs), num_trajs)
        length = None
        
        samples = defaultdict(list)
        for idx in idxs:
            # get the trajectory
            traj = self.trajs[idx]
            if length is None:
                length = len(traj)
            else:
                assert len(traj) == length
            
            observations = np.stack(
                [traj[i][0] for i in range(len(traj))]
            )
            actions = np.stack(
                [traj[i][1] for i in range(len(traj))]
            )
            next_observations = np.stack(
                [traj[i][2] for i in range(len(traj))]
            )
            terminals = np.stack(
                [traj[i][3] for i in range(len(traj))]
            )
            rewards = np.stack(
                [traj[i][4] for i in range(len(traj))]
            )
            
            samples["observations"].append(observations)
            samples["actions"].append(actions)
            samples["next_observations"].append(next_observations)
            samples["terminals"].append(terminals)
            samples["rewards"].append(rewards)
            
        return {
            k: np.stack(v)
            for k, v in samples.items()
        }