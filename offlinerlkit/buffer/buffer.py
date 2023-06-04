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
        segment_length: int,
        device: str = "cpu"
    ) -> None:
        self.device = torch.device(device)
        self.segment_length = segment_length
        
        self.obs_shape = dataset["observations"].shape[1]
        self.obs_dtype = dataset["observations"].dtype
        self.action_dim = dataset["actions"].shape[1]
        self.action_dtype = dataset["actions"].dtype
        
        # splitting into trajectories (modified from IQL repo)
        trajs = defaultdict(list)
        
        traj_obs = []
        traj_acts = []
        traj_next_obs = []
        traj_terminals = []
        traj_rewards = []
        for i in range(len(dataset["observations"])):
            traj_obs.append(dataset["observations"][i])
            traj_acts.append(dataset["actions"][i])
            traj_next_obs.append(dataset["next_observations"][i])
            traj_terminals.append(dataset["terminals"][i])
            traj_rewards.append(dataset["rewards"][i])
            
            # in the terminal case, add everything to the dict and reset
            if dataset["terminals"][i] == 1.0 and i + 1 < len(dataset["observations"]):
                trajs["observations"].append(np.stack(traj_obs))
                trajs["actions"].append(np.stack(traj_acts))
                trajs["next_observations"].append(np.stack(traj_next_obs))
                trajs["terminals"].append(np.stack(traj_terminals))
                trajs["rewards"].append(np.stack(traj_rewards))
                
                # reset to collect next trajectory's data
                traj_obs = []
                traj_acts = []
                traj_next_obs = []
                traj_terminals = []
                traj_rewards = []
                
        self.trajs = trajs # Dict[str, List[np.ndarray]]
        
    @property
    def num_trajs(self):
        return len(self.trajs["observations"])
        
    @property
    def traj_lengths(self):
        return [len(traj) for traj in self.trajs["observations"]]
    
    @property
    def traj_rewards(self):
        return [np.sum(traj_r) for traj_r in self.trajs["rewards"]]
        
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Samples a batch of segments of trajectories.
        
        Output should be of size [batch_size, segment_length, input_shape].
        """
        idxs = np.random.randint(0, self.num_trajs, batch_size) # traj indices
        
        samples = defaultdict(list)
        for idx in idxs:
            start_idx = np.random.randint(0, self.traj_lengths[idx] - self.segment_length)
            
            for key in ["observations", "actions", "next_observations", "terminals", "rewards"]:
                value = torch.from_numpy(self.trajs[key][idx][start_idx : start_idx + self.segment_length])
                samples[key].append(value)
                
        samples = {
            k: torch.stack(v).to(self.device)
            for k, v in samples.items()
        }
        return samples
    
    def sample_pairs(self, batch_size: int, return_preference_label: bool = False) -> Dict[str, torch.Tensor]:
        """
        Samples a batch of pairs of segments of trajectories.
        Optionally can return label given from BTL preference model over ground truth rewards.
        
        Output should be a dict, whose elements are of size [batch_size, 2, segment_length, input_shape].
        """
        idxs = np.random.randint(0, self.num_trajs, (batch_size, 2)) # traj indices (can be the same, but then there is no real preference)
        
        samples = defaultdict(list)
        for idx1, idx2 in idxs:
            start_idx1 = np.random.randint(0, self.traj_lengths[idx1] - self.segment_length)
            start_idx2 = np.random.randint(0, self.traj_lengths[idx2] - self.segment_length)
            
            for key in ["observations", "actions", "next_observations", "terminals", "rewards"]:
                value1 = torch.from_numpy(self.trajs[key][idx1][start_idx1 : start_idx1 + self.segment_length])
                value2 = torch.from_numpy(self.trajs[key][idx2][start_idx2 : start_idx2 + self.segment_length])
                
                value = torch.stack([value1, value2], dim=0) # size (2, segment_length, input_shape), first traj is index 0, second is index 1
                samples[key].append(value)
                
                if return_preference_label and key == "rewards":
                    # compute preference label over ground truth rewards in the dataset
                    rew1 = value1.sum()
                    rew2 = value2.sum()
                    one_prob = torch.sigmoid(rew1 - rew2) # this is BTL preference model, whatever comes out is the probability that the first trajectory is better.
                    
                    # labels are consistent across all specific (segment1, segment2) data here
                    probs = torch.tensor([1 - one_prob, one_prob])
                    label_1 = torch.multinomial(probs, num_samples=1).repeat(self.segment_length)
                    label_2 = 1.0 - label_1
                    label = torch.stack([label_1, label_2], dim=0)
                    samples["preference_label"].append(label) # (2, segment_length)
                
        samples = {
            k: torch.stack(v).to(self.device) # (B, 2, segment_length, input_shape)
            for k, v in samples.items()
        }
        return samples