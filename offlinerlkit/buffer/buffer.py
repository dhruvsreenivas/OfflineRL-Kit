import numpy as np
import torch

from typing import Tuple, Dict, List, Union
from collections import defaultdict
from torch.utils.data import Dataset


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

# =================================== PREFERENCE BASED LEARNING BUFFERS ===================================

class PreferenceDataset(Dataset):
    def __init__(self, offline_data: List[Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], torch.Tensor]], device: str):
        super().__init__()
        self.offline_data = offline_data
        self.device = torch.device(device)
        
    def __len__(self):
        """Number of trajectory pairs."""
        return len(self.offline_data)
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        tau1, tau2, label = self.offline_data[idx]
        return {
            "observations1": maybe_to_torch(tau1["observations"], self.device),
            "actions1": maybe_to_torch(tau1["actions"], self.device),
            "next_observations1": maybe_to_torch(tau1["next_observations"], self.device),
            "terminals1": maybe_to_torch(tau1["terminals"], self.device),
            "observations2": maybe_to_torch(tau2["observations"], self.device),
            "actions2": maybe_to_torch(tau2["actions"], self.device),
            "next_observations2": maybe_to_torch(tau2["next_observations"], self.device),
            "terminals2": maybe_to_torch(tau2["terminals"], self.device),
            "label": maybe_to_torch(label, self.device)
        }
        
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Samples a batch of stuff randomly from the dataset."""
        assert batch_size > 0, "batch_size <= 0 is bad"
        idxs = np.random.randint(0, len(self.offline_data), batch_size)
        dps = [self[idx] for idx in idxs]
        batch = {
            k: torch.stack([dp[k] for dp in dps])
            for k in list(dps[0].keys())
        }
        return batch
        
    def statistics(self, eps: float = 1e-3) -> Tuple[np.ndarray, np.ndarray]:
        """Gets mean and std ALL (obs, actions) inputs (from both tau1 and tau2)."""
        obs1 = torch.cat([maybe_to_torch(tau1["observations"], 'cpu') for tau1, _, _ in self.offline_data])
        acts1 = torch.cat([maybe_to_torch(tau1["actions"], 'cpu') for tau1, _, _ in self.offline_data])
        obs2 = torch.cat([maybe_to_torch(tau2["observations"], 'cpu') for _, tau2, _ in self.offline_data])
        acts2 = torch.cat([maybe_to_torch(tau2["actions"], 'cpu') for _, tau2, _ in self.offline_data])
        obs_actions1 = torch.cat([obs1, acts1], dim=-1)
        obs_actions2 = torch.cat([obs2, acts2], dim=-1)
        
        obs_actions = torch.cat([obs_actions1, obs_actions2], dim=0)
        mean = obs_actions.mean(dim=0, keepdim=True)
        std = obs_actions.std(dim=0, keepdim=True) + eps
        
        # delete everything that is not necessary
        del obs1, acts1, obs2, acts2, obs_actions1, obs_actions2, obs_actions
        return mean.cpu().numpy(), std.cpu().numpy()

def filter(dataset: PreferenceDataset, idxs: list) -> PreferenceDataset:
    dps = [dataset[idx] for idx in idxs]
    keys = list(dps[0].keys())
    
    # make a list of the specific segment dicts
    snips1 = [{k[:-1]: dp[k] for k in keys if (k != "label" and k.endswith("1"))} for dp in dps]
    snips2 = [{k[:-1]: dp[k] for k in keys if (k != "label" and k.endswith("2"))} for dp in dps]
    labels = [dp["label"] for dp in dps]
    
    tups = [(snip1, snip2, label) for snip1, snip2, label in zip(snips1, snips2, labels)]
    return PreferenceDataset(tups, device=dataset.device)


def maybe_to_torch(x: Union[np.ndarray, torch.Tensor], device: Union[str, torch.device]) -> torch.Tensor:
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    return x.to(device)


class TrajectoryBuffer:
    """Offline dataset of trajectories as opposed to samples."""
    def __init__(
        self,
        dataset: Dict[str, np.ndarray],
        segment_length: int = 15,
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
            traj_terminals.append(dataset["terminals"][i].astype(np.float32))
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
        
        # in the case we haven't added anything (i.e. there is no terminal signal) or if we've reached the end, we add the stuff in
        if len(traj_obs) > 0:
            trajs["observations"].append(np.stack(traj_obs))
            trajs["actions"].append(np.stack(traj_acts))
            trajs["next_observations"].append(np.stack(traj_next_obs))
            trajs["terminals"].append(np.stack(traj_terminals))
            trajs["rewards"].append(np.stack(traj_rewards))
        
        self.trajs = trajs # Dict[str, List[np.ndarray]]
        self.trajectory_preference_dataset = None
        self.snippet_preference_dataset = None
        
    @classmethod
    def from_trajectories(cls, trajs: Dict[str, List[np.ndarray]], segment_length: int, device: str) -> "TrajectoryBuffer":
        observations = np.concatenate(trajs["observations"])
        actions = np.concatenate(trajs["actions"])
        next_observations = np.concatenate(trajs["next_observations"])
        terminals = np.concatenate(trajs["terminals"])
        rewards = np.concatenate(trajs["rewards"])
        
        dataset = dict(
            observations=observations,
            actions=actions,
            next_observations=next_observations,
            terminals=terminals,
            rewards=rewards
        )
        return cls(
            dataset,
            segment_length,
            device
        )
        
    @property
    def num_trajs(self):
        return len(self.trajs["observations"])
        
    @property
    def traj_lengths(self):
        return [len(traj) for traj in self.trajs["observations"]]
    
    @property
    def size(self):
        return np.sum(self.traj_lengths)
    
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
                    probs = torch.bernoulli(one_prob)
                    label_1 = torch.multinomial(probs, num_samples=1).repeat(self.segment_length) # instead of sampling w prob, we can just say that the trajectory with higher reward automatically gets the good label
                    label_2 = 1.0 - label_1
                    label = torch.stack([label_1, label_2], dim=0)
                    samples["preference_labels"].append(label) # (2, segment_length)
                
        samples = {
            k: torch.stack(v).to(self.device) # (B, 2, segment_length, input_shape)
            for k, v in samples.items()
        }
        return samples
        
    def generate_trajectory_preference_dataset(self, name: str) -> None:
        """Generates offline data of pairs of trajectories (tau_1, tau_2, label). Across all pairs of trajectories."""
        offline_dataset = []
        
        # loop through data
        for i in range(self.num_trajs):
            obs1 = self.trajs["observations"][i]
            act1 = self.trajs["actions"][i]
            next_obs1 = self.trajs["next_observations"][i]
            terminal1 = self.trajs["terminals"][i]
            total_rew1 = np.sum(self.trajs["rewards"][i])
            
            print("=== FIRST TRAJ SHAPES ===")
            print(obs1.shape)
            print(act1.shape)
            print(next_obs1.shape)
            print(terminal1.shape)
            print(total_rew1.shape)
            
            tau1 = dict(
                observations=obs1,
                actions=act1,
                next_observations=next_obs1,
                terminals=terminal1
            )
            
            for j in range(i + 1, self.num_trajs):
                obs2 = self.trajs["observations"][j]
                act2 = self.trajs["actions"][j]
                next_obs2 = self.trajs["next_observations"][j]
                terminal2 = self.trajs["terminals"][j]
                total_rew2 = np.sum(self.trajs["rewards"][j])
                
                print("=== SECOND TRAJ SHAPES ===")
                print(obs2.shape)
                print(act2.shape)
                print(next_obs2.shape)
                print(terminal2.shape)
                print(total_rew2.shape)
                
                tau2 = dict(
                    observations=obs2,
                    actions=act2,
                    next_observations=next_obs2,
                    terminals=terminal2
                )
                
                reward_diff = torch.tensor(total_rew1 - total_rew2)
                label_prob = torch.sigmoid(reward_diff)
                label = torch.bernoulli(label_prob)
                
                # add to dataset
                offline_dataset.append((tau1, tau2, label))
        
        offline_dataset = PreferenceDataset(offline_dataset, self.device)
        
        # save dataset somewhere for future reference so we can load this as fixed later
        torch.save(offline_dataset, f"~/OfflineRL-Kit/offline_data/{name}_trajectory_preference_dataset.pt")
        
        # set as class variable
        self.trajectory_preference_dataset = offline_dataset
        
    def load_trajectory_preference_dataset(self, path: str) -> None:
        dataset = torch.load(path, map_location=self.device)
        self.trajectory_preference_dataset = dataset
        
    def generate_snippet_preference_dataset(self, name: str, sample_label: bool = True) -> None:
        # if there is something hella small in the dataset (smaller than segment length), then we use that
        self.segment_length = min(self.segment_length, min(self.traj_lengths))
        num_pairs = int(self.size) // self.segment_length
        
        datapoints = []
        for _ in range(num_pairs):
            # sample a pair of trajectories
            traj_idx1 = np.random.randint(0, self.num_trajs)
            traj_idx2 = np.random.randint(0, self.num_trajs)
            
            # sample snippets from each trajectory
            start_idx1 = np.random.randint(0, self.traj_lengths[traj_idx1] - self.segment_length) if self.traj_lengths[traj_idx1] > self.segment_length else 0
            start_idx2 = np.random.randint(0, self.traj_lengths[traj_idx2] - self.segment_length) if self.traj_lengths[traj_idx2] > self.segment_length else 0
            
            snip1 = {
                k: v[traj_idx1][start_idx1 : start_idx1 + self.segment_length]
                for k, v in self.trajs.items()
            }
            snip2 = {
                k: v[traj_idx2][start_idx2 : start_idx2 + self.segment_length]
                for k, v in self.trajs.items()
            }
            
            # get label based off of BTL model of reward
            rew1 = np.sum(snip1["rewards"])
            rew2 = np.sum(snip2["rewards"])
            diff = torch.sigmoid(torch.tensor(rew1 - rew2))
            if sample_label:
                label = torch.bernoulli(diff) # this is the label -> 0 if 1 < 2, 1 if not
            else:
                label = torch.as_tensor(int(rew1 > rew2))
            
            datapoints.append((snip1, snip2, label))
        
        offline_dataset = PreferenceDataset(datapoints, self.device)
        
        # save dataset somewhere for future reference so we can load this as fixed later
        torch.save(offline_dataset, f"/home/ds844/OfflineRL-Kit/offline_data/{name}_snippet_preference_dataset_seglen{self.segment_length}_{'deterministic' if sample_label else ''}.pt")
        
        # set as class variable
        self.snippet_preference_dataset = offline_dataset
        
    def load_snippet_preference_dataset(self, path: str) -> None:
        dataset = torch.load(path, map_location=self.device)
        self.snippet_preference_dataset = dataset