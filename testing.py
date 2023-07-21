import torch
import numpy as np
import random
import d4rl
import gym
import matplotlib.pyplot as plt
from typing import Dict

from offlinerlkit.modules import EnsembleRewardModel
from offlinerlkit.rewards import EnsembleReward
from offlinerlkit.buffer import PreferenceDataset
from offlinerlkit.utils.logger import Logger, make_log_dirs
from offlinerlkit.utils.scaler import StandardScaler

"""Various testing utilities."""


@torch.no_grad()
def validate_reward_model(ensemble: EnsembleReward, dataset: PreferenceDataset) -> None:
    num_correct = 0
    for i in range(len(dataset)):
        dp = dataset[i]
        
        # just look at the elite models
        r1 = ensemble.get_reward(dp["observations1"].cpu().numpy(), dp["actions1"].cpu().numpy()).sum()
        r2 = ensemble.get_reward(dp["observations2"].cpu().numpy(), dp["actions2"].cpu().numpy()).sum()
        if np.isnan(r1).any() or np.isnan(r2).any():
            print("Hit a NaN, breaking out...")
            break
        
        lbl = dp["label"]
        if (r1 > r2 and lbl.item() == 1.0) or (r1 < r2 and lbl.item() == 0.0):
            num_correct += 1
    
    print(f"****** fraction of predictions which are correct: {num_correct / len(dataset)} ******")
    
    
@torch.no_grad()
def validate_reward_model_ood(ensemble: EnsembleReward, dataset: Dict[str, np.ndarray]) -> None:
    correct = 0
    n_trajs = dataset["observations"].shape[0] // 1000
    for _ in range(1000000):
        idx1 = np.random.randint(0, n_trajs)
        idx2 = np.random.randint(0, n_trajs)
        
        # verify that they're different
        while idx1 == idx2:
            idx1 = np.random.randint(0, n_trajs)
            idx2 = np.random.randint(0, n_trajs)
        
        s1, a1 = dataset["observations"][1000 * idx1 : 1000 * (idx1 + 1)], dataset["actions"][1000 * idx1 : 1000 * (idx1 + 1)]
        s2, a2 = dataset["observations"][1000 * idx2 : 1000 * (idx2 + 1)], dataset["actions"][1000 * idx2 : 1000 * (idx2 + 1)]
        gt_r1 = dataset["rewards"][1000 * idx1 : 1000 * (idx1 + 1)].sum()
        gt_r2 = dataset["rewards"][1000 * idx2 : 1000 * (idx2 + 1)].sum()
        
        r1 = ensemble.get_reward(s1, a1).sum()
        r2 = ensemble.get_reward(s2, a2).sum()
        if np.isnan(r1).any() or np.isnan(r2).any():
            print("Hit a NaN, breaking out...")
            break
        
        if (gt_r1 > gt_r2 and r1 > r2) or (gt_r1 < gt_r2 and r1 < r2):
            correct += 1
            
    print(f"fraction of times the reward model predicted correctly on this dataset: {correct / 1000000}")


def test_normal_reward_learning(dataset: PreferenceDataset) -> None:
    """Tests whether we can do reward learning with a separate reward model."""
    
    # seed
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    
    # get information from dset such as obs dim, action dim, etc.
    obs_dim = dataset.offline_data[0][0]["observations"].shape[-1]
    action_dim = dataset.offline_data[0][0]["actions"].shape[-1]
    
    # create model + optimizer
    model = EnsembleRewardModel(
        obs_dim, action_dim,
        hidden_dims=[200, 200, 200, 200],
        num_ensemble=7,
        with_action=True,
        weight_decays=None,
        dropout_probs=None
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=3e-4
    )
    ensemble = EnsembleReward(
        model,
        optimizer,
        scaler=None,
        penalty_coef=0
    )
    
    # loop through like 100 epochs or whatever
    log_dirs = make_log_dirs('halfcheetah-random-v2', 'pure_reward_learning', 0, {})
    output_config = {
        "consoleout_backup": "stdout",
        "policy_training_progress": "csv",
        "dynamics_training_progress": "csv",
        "reward_training_progress": "csv",
        "tb": "tensorboard"
    }
    logger = Logger(log_dirs, output_config)
    logger.log_hyperparameters({})
    ensemble.train(
        dataset,
        logger,
        max_epochs=100,
        holdout_ratio=0.1
    )
    return ensemble


def test_reward_model_explained_variance(reward_model_dir: str, normalize_r: bool = False):
    
    # load offline dataset corresponding to the reward model
    if "halfcheetah-medium-replay-v2" in reward_model_dir:
        ind_dataset = d4rl.qlearning_dataset(gym.make("halfcheetah-medium-replay-v2"))
    elif "halfcheetah-medium-v2" in reward_model_dir:
        ind_dataset = d4rl.qlearning_dataset(gym.make("halfcheetah-medium-v2"))
    else:
        ind_dataset = d4rl.qlearning_dataset(gym.make("halfcheetah-random-v2"))
        
    # load OOD dataset
    ood_dataset = d4rl.qlearning_dataset(gym.make("halfcheetah-expert-v2"))
    
    # normalize both datasets
    def normalize_obs(observations, eps=1e-3):
        mean = np.mean(observations, 0, keepdims=True)
        std = np.std(observations, 0, keepdims=True) + eps
        norm_obs = (observations - mean) / std
        return norm_obs
    
    ind_dataset["observations"] = normalize_obs(ind_dataset["observations"])
    ood_dataset["observations"] = normalize_obs(ood_dataset["observations"])
    
    # assert obs + action dims are the same
    for k in ["observations", "actions"]:
        assert ind_dataset[k].shape[-1] == ood_dataset[k].shape[-1]
    
    # create reward model, load in state dict
    reward_model = EnsembleRewardModel(
        obs_dim=ind_dataset["observations"].shape[-1],
        action_dim=ind_dataset["actions"].shape[-1],
        hidden_dims=[200, 200, 200, 200],
        num_ensemble=7,
        num_elites=5,
        with_action=True,
        weight_decays=[0.0, 0.0, 0.0, 0.0, 0.0],
        dropout_probs=[0.0, 0.0, 0.0, 0.0],
        reward_final_activation="none",
        device="cuda"
    )
    
    reward_optim = torch.optim.Adam(
        reward_model.parameters(),
        lr=3e-4
    )
    scaler = StandardScaler()
    scaler.load_scaler(reward_model_dir)
    reward = EnsembleReward(
        reward_model,
        reward_optim,
        scaler,
        penalty_coef=0.0,
        uncertainty_mode="aleatoric"
    )
    reward.load(reward_model_dir)
    print(f"loaded in reward model")
    
    
    def explained_variance(hatr: np.ndarray, r: np.ndarray):
        assert r.ndim == 1 and hatr.ndim == 1 and r.shape[0] == hatr.shape[0]
        true_var = np.var(r)
        return np.nan if true_var == 0 else 1 - np.var(r - hatr) / true_var

    # get learned rewards
    ind_r = []
    for inds, inda in zip(ind_dataset["observations"], ind_dataset["actions"]):
        inds = np.expand_dims(inds, 0)
        inda = np.expand_dims(inda, 0)
        r = reward.get_reward(inds, inda).squeeze()
        ind_r.append(r)
    ind_r = np.stack(ind_r)
    print(f"computed in-distribution rewards!")
    
    ood_r = []
    for oods, ooda in zip(ood_dataset["observations"], ood_dataset["actions"]):
        oods = np.expand_dims(oods, 0)
        ooda = np.expand_dims(ooda, 0)
        r = reward.get_reward(oods, ooda).squeeze()
        ood_r.append(r)
    ood_r = np.stack(ood_r)
    print(f"computed OOD rewards!")
    
    ind_gt_reward = ind_dataset["rewards"].squeeze()
    ood_gt_reward = ood_dataset["rewards"].squeeze()
    
    # look at scaling of the real reward vs. the predicted reward for ind vs. ood
    ind_scale_factor = np.mean(ind_r) / np.mean(ind_gt_reward)
    ind_mse = np.mean((ind_scale_factor * ind_gt_reward - ind_r) ** 2)
    
    ood_scale_factor = np.mean(ood_r) / np.mean(ood_gt_reward)
    ood_mse = np.mean((ood_scale_factor * ood_gt_reward - ood_r) ** 2)
    
    print(f"ind MSE / ood MSE: {ind_mse, ood_mse}")
    
    if normalize_r:
        ind_r = (ind_r - np.mean(ind_r)) / (np.std(ind_r) + 1e-8)
        ood_r = (ood_r - np.mean(ood_r)) / (np.std(ood_r) + 1e-8)
        
        ind_gt_reward = (ind_gt_reward - np.mean(ind_gt_reward)) / (np.std(ind_gt_reward) + 1e-8)
        ood_gt_reward = (ood_gt_reward - np.mean(ood_gt_reward)) / (np.std(ood_gt_reward) + 1e-8)
    
    ind_var = explained_variance(ind_r, ind_gt_reward)
    ood_var = explained_variance(ood_r, ood_gt_reward)
    print(f"Explained variance for in-distribution dataset: {ind_var}")
    print(f"Explained variance for out of distribution dataset: {ood_var}")
    
    # plot just to see whether things are right here
    plt.plot(np.arange(ind_gt_reward.shape[0]), ind_gt_reward, label="ind gt rew")
    plt.plot(np.arange(ood_gt_reward.shape[0]), ood_gt_reward, label="ood gt rew")
    plt.plot(np.arange(ind_r.shape[0]), ind_r, label="ind pred r")
    plt.plot(np.arange(ood_r.shape[0]), ood_r, label="ood pred r")
    
    plt.legend()
    plt.savefig(f"./plots/ev_{'medium' if 'medium' in reward_model_dir else 'random'}.png")
    
    
def test_model_ood(reward_path: str):
    dataset = d4rl.qlearning_dataset(gym.make("halfcheetah-expert-v2"))
    
    reward_model = EnsembleRewardModel(
        obs_dim=dataset["observations"].shape[-1],
        action_dim=dataset["actions"].shape[-1],
        hidden_dims=[200, 200, 200, 200],
        num_ensemble=7,
        num_elites=5,
        with_action=True,
        weight_decays=[0.0, 0.0, 0.0, 0.0, 0.0],
        dropout_probs=[0.0, 0.0, 0.0, 0.0],
        reward_final_activation="none",
        device="cuda"
    )
    
    reward_optim = torch.optim.Adam(
        reward_model.parameters(),
        lr=3e-4
    )
    scaler = StandardScaler()
    scaler.load_scaler(reward_path)
    reward = EnsembleReward(
        reward_model,
        reward_optim,
        scaler,
        penalty_coef=0.0,
        uncertainty_mode="aleatoric"
    )
    reward.load(reward_path)
    print(f"loaded in reward model")
    
    # normalize dataset obs
    def normalize_obs(observations, eps=1e-3):
        mean = np.mean(observations, 0, keepdims=True)
        std = np.std(observations, 0, keepdims=True) + eps
        norm_obs = (observations - mean) / std
        return norm_obs
    
    dataset["observations"] = normalize_obs(dataset["observations"])
    
    # validate model
    validate_reward_model_ood(reward, dataset)


if __name__ == '__main__':
    # dataset = torch.load('./offline_data/halfcheetah-random-v2_snippet_preference_dataset_seglen15_deterministic.pt')
    # ensemble = test_normal_reward_learning(dataset)
    # validate_reward_model(ensemble, dataset)
    
    reward_path = "./log/halfcheetah-random-v2/rambo_relabeled/seed_0&timestamp_23-0713-155029/model"
    test_model_ood(reward_path)