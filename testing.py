import torch
from torch.utils.data import DataLoader
import numpy as np
import random

from offlinerlkit.modules import EnsembleRewardModel
from offlinerlkit.rewards import EnsembleReward
from offlinerlkit.buffer import PreferenceDataset
from offlinerlkit.utils.logger import Logger, make_log_dirs

"""Various testing utilities."""

@torch.no_grad()
def validate_reward_model(ensemble: EnsembleReward, dataset: PreferenceDataset) -> None:
    num_correct = 0
    for i in range(len(dataset)):
        dp = dataset[i]
        
        # just look at the first model like OPRL does
        r1 = ensemble.model(dp["observations1"], dp["actions1"], train=False)[0][0].sum()
        r2 = ensemble.model(dp["observations2"], dp["actions2"], train=False)[0][0].sum()
        lbl = dp["label"]
        
        print('------------------------------------')
        print(f"predicted reward for first traj: {r1}")
        print(f"predicted reward for second traj: {r2}")
        print(f"is first traj actually better than second traj: {'YES' if lbl.item() == 1.0 else 'NO'}")
        
        if (r1 > r2 and lbl.item() == 1.0) or (r1 < r2 and lbl.item() == 0.0):
            num_correct += 1
    
    print(f"fraction of predictions which are correct: {num_correct / len(dataset)}")

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
    
if __name__ == '__main__':
    dataset = torch.load('./offline_data/halfcheetah-random-v2_snippet_preference_dataset_seglen15_deterministic.pt')
    ensemble = test_normal_reward_learning(dataset)
    validate_reward_model(ensemble, dataset)