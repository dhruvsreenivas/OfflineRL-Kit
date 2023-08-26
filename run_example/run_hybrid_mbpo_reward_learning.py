import warnings
warnings.filterwarnings('ignore')
import getpass

import argparse
import os
import sys
import random

import gym
import d4rl

import numpy as np
import torch


from offlinerlkit.nets import MLP
from offlinerlkit.modules import ActorProb, Critic, TanhDiagGaussian, EnsembleDynamicsModel, EnsembleRewardModel
from offlinerlkit.dynamics import EnsembleDynamics
from offlinerlkit.rewards import EnsembleReward
from offlinerlkit.utils.scaler import StandardScaler
from offlinerlkit.utils.termination_fns import get_termination_fn
from offlinerlkit.utils.load_dataset import qlearning_dataset
from offlinerlkit.buffer import ReplayBuffer, PreferenceDataset
from offlinerlkit.utils.logger import Logger, make_log_dirs
from offlinerlkit.policy_trainer import HybridMBPOPolicyTrainer
from offlinerlkit.policy import HybridMBPORewardLearningPolicy


"""
suggested hypers

halfcheetah-medium-v2: rollout-length=5, penalty-coef=0.5
hopper-medium-v2: rollout-length=5, penalty-coef=5.0
walker2d-medium-v2: rollout-length=5, penalty-coef=0.5
halfcheetah-medium-replay-v2: rollout-length=5, penalty-coef=0.5
hopper-medium-replay-v2: rollout-length=5, penalty-coef=2.5
walker2d-medium-replay-v2: rollout-length=1, penalty-coef=2.5
halfcheetah-medium-expert-v2: rollout-length=5, penalty-coef=2.5
hopper-medium-expert-v2: rollout-length=5, penalty-coef=5.0
walker2d-medium-expert-v2: rollout-length=1, penalty-coef=2.5
"""


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--netid", type=str, default=None)
    parser.add_argument("--algo-name", type=str, default="hybrid_pref_mbpo")
    parser.add_argument("--task", type=str, default="halfcheetah-medium-v2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--hidden-dims", type=int, nargs='*', default=[256, 256])
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--auto-alpha", default=True)
    parser.add_argument("--target-entropy", type=int, default=None)
    parser.add_argument("--alpha-lr", type=float, default=1e-4)

    parser.add_argument("--dynamics-lr", type=float, default=1e-3)
    parser.add_argument("--dynamics-hidden-dims", type=int, nargs='*', default=[200, 200, 200, 200])
    parser.add_argument("--dynamics-weight-decay", type=float, nargs='*', default=[2.5e-5, 5e-5, 7.5e-5, 7.5e-5, 1e-4])
    parser.add_argument("--n-ensemble", type=int, default=7)
    parser.add_argument("--n-elites", type=int, default=5)
    parser.add_argument("--rollout-batch-size", type=int, default=50000)
    parser.add_argument("--rollout-min-length", type=int, default=1)
    parser.add_argument("--rollout-max-length", type=int, default=15)
    parser.add_argument("--rollout-min-epoch", type=int, default=20),
    parser.add_argument("--rollout-max-epoch", type=int, default=150)
    parser.add_argument("--penalty-coef", type=float, default=0.0)
    parser.add_argument("--model-retain-epochs", type=int, default=5)
    parser.add_argument("--real-ratio", type=float, default=0.05)
    parser.add_argument("--load-dynamics-path", type=str, default=None)

    parser.add_argument("--epoch", type=int, default=3000)
    parser.add_argument("--step-per-epoch", type=int, default=1000)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    # reward learning args
    parser.add_argument("--segment-length", type=int, default=15)
    parser.add_argument("--use-reward-scaler", type=bool, default=True, help='whether to use dynamics scaler for reward learning or not')
    parser.add_argument("--reward-hidden-dims", type=int, nargs='*', default=[200, 200, 200, 200])
    parser.add_argument("--reward-dropout-probs", type=float, nargs='*', default=[0.0, 0.0, 0.0, 0.0])
    parser.add_argument("--reward-weight-decay", type=float, nargs='*', default=[0.0, 0.0, 0.0, 0.0, 0.0])
    parser.add_argument("--reward-lr", type=float, default=3e-4)
    parser.add_argument("--n-reward_models", type=int, default=7)
    parser.add_argument("--n-reward-elites", type=int, default=5)
    parser.add_argument("--reward-with-action", type=bool, default=True)
    
    parser.add_argument("--reward_batch_size", type=int, default=256)
    parser.add_argument("--reward-uncertainty-mode", type=str, default="aleatoric")
    parser.add_argument("--reward-final-activation", type=str, default="none")
    parser.add_argument("--reward-soft-clamp", type=bool, default=False)
    parser.add_argument("--pred-discounted-return", type=bool, default=False, help='whether to predict discounted return as opposed to full return.')
    parser.add_argument("--normalize-reward-eval", type=bool, default=False)
    parser.add_argument("--reward-penalty-coef", type=float, default=0.0)
    parser.add_argument("--load-reward-path", type=str, default=None)
    
    # online args
    parser.add_argument("--init-exploration-steps", type=int, default=0)
    parser.add_argument("--online-ratio", type=float, default=0.5)
    parser.add_argument("--dynamics-update-freq", type=int, default=1000)
    parser.add_argument("--add-data-outside-of-model-training", type=bool, default=True)

    return parser.parse_args()


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


def train(args=get_args()):
    # create env and dataset
    train_env = gym.make(args.task)
    env = gym.make(args.task)
    dataset = qlearning_dataset(env)
    args.obs_shape = env.observation_space.shape
    args.action_dim = np.prod(env.action_space.shape)
    args.max_action = env.action_space.high[0]
    
    # grab preference datasets
    netid = args.netid if args.netid is not None else getpass.getuser()
    dataset_path = f"/home/{netid}/OfflineRL-Kit/offline_data/{args.task}_snippet_preference_dataset_seglen{args.segment_length}_deterministic.pt" # here, observations are not normalized
    offline_pref_dataset = torch.load(dataset_path)
    offline_pref_dataset.device = args.device
    
    online_pref_dataset = PreferenceDataset(offline_data=[], device=args.device)

    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    env.seed(args.seed)

    # create policy model
    actor_backbone = MLP(input_dim=np.prod(args.obs_shape), hidden_dims=args.hidden_dims)
    critic1_backbone = MLP(input_dim=np.prod(args.obs_shape) + args.action_dim, hidden_dims=args.hidden_dims)
    critic2_backbone = MLP(input_dim=np.prod(args.obs_shape) + args.action_dim, hidden_dims=args.hidden_dims)
    dist = TanhDiagGaussian(
        latent_dim=getattr(actor_backbone, "output_dim"),
        output_dim=args.action_dim,
        unbounded=True,
        conditioned_sigma=True
    )
    actor = ActorProb(actor_backbone, dist, args.device)
    critic1 = Critic(critic1_backbone, args.device)
    critic2 = Critic(critic2_backbone, args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(actor_optim, args.epoch)

    if args.auto_alpha:
        target_entropy = args.target_entropy if args.target_entropy \
            else -np.prod(env.action_space.shape)

        args.target_entropy = target_entropy

        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        alpha = (target_entropy, log_alpha, alpha_optim)
    else:
        alpha = args.alpha

    # create dynamics
    load_dynamics_model = True if args.load_dynamics_path else False
    dynamics_model = EnsembleDynamicsModel(
        obs_dim=np.prod(args.obs_shape),
        action_dim=args.action_dim,
        hidden_dims=args.dynamics_hidden_dims,
        num_ensemble=args.n_ensemble,
        num_elites=args.n_elites,
        weight_decays=args.dynamics_weight_decay,
        with_reward=False,
        device=args.device
    )
    dynamics_optim = torch.optim.Adam(
        dynamics_model.parameters(),
        lr=args.dynamics_lr
    )
    scaler = StandardScaler()
    termination_fn = get_termination_fn(task=args.task)
    dynamics = EnsembleDynamics(
        dynamics_model,
        dynamics_optim,
        scaler,
        termination_fn,
        penalty_coef=args.penalty_coef,
    )

    if args.load_dynamics_path:
        dynamics.load(args.load_dynamics_path)
        
    # create reward learner
    reward_model = EnsembleRewardModel(
        obs_dim=np.prod(args.obs_shape),
        action_dim=args.action_dim,
        hidden_dims=args.reward_hidden_dims,
        num_ensemble=args.n_ensemble,
        num_elites=args.n_elites,
        with_action=args.reward_with_action,
        weight_decays=args.reward_weight_decay,
        dropout_probs=args.reward_dropout_probs,
        soft_clamp_output=args.reward_soft_clamp,
        reward_final_activation=args.reward_final_activation,
        device=args.device
    )
    reward_optim = torch.optim.Adam(
        reward_model.parameters(),
        lr=args.reward_lr
    )
    reward = EnsembleReward(
        reward_model,
        reward_optim,
        dynamics.scaler if args.use_reward_scaler else None,
        args.normalize_reward_eval,
        args.gamma if args.pred_discounted_return else 1.0,
        args.reward_penalty_coef,
        args.reward_uncertainty_mode
    )
    if args.load_reward_path:
        reward.load(args.load_reward_path)

    # create policy
    policy = HybridMBPORewardLearningPolicy(
        dynamics,
        reward,
        actor,
        critic1,
        critic2,
        actor_optim,
        critic1_optim,
        critic2_optim,
        dynamics_optim,
        reward_optim,
        tau=args.tau,
        gamma=args.gamma,
        alpha=alpha,
        online_batch_size=args.batch_size,
        online_to_total_ratio=args.online_ratio
    )

    # create buffer
    offline_buffer = ReplayBuffer(
        buffer_size=len(dataset["observations"]),
        obs_shape=args.obs_shape,
        obs_dtype=np.float32,
        action_dim=args.action_dim,
        action_dtype=np.float32,
        device=args.device
    )
    offline_buffer.load_dataset(dataset)
    
    online_buffer = ReplayBuffer(
        buffer_size=len(dataset["observations"]),
        obs_shape=args.obs_shape,
        obs_dtype=np.float32,
        action_dim=args.action_dim,
        action_dtype=np.float32,
        device=args.device
    )
    
    mb_buffer = ReplayBuffer(
        buffer_size=args.rollout_batch_size * args.rollout_max_length * args.model_retain_epochs,
        obs_shape=args.obs_shape,
        obs_dtype=np.float32,
        action_dim=args.action_dim,
        action_dtype=np.float32,
        device=args.device
    )

    # log
    log_dirs = make_log_dirs(args.task, args.algo_name, args.seed, vars(args))
    # key: output file name, value: output handler type
    output_config = {
        "consoleout_backup": "stdout",
        "policy_training_progress": "csv",
        "dynamics_training_progress": "csv",
        "tb": "tensorboard"
    }
    logger = Logger(log_dirs, output_config)
    logger.log_hyperparameters(vars(args))

    # create policy trainer
    policy_trainer = HybridMBPOPolicyTrainer(
        policy=policy,
        train_env=train_env,
        eval_env=env,
        
        online_buffer=online_buffer,
        online_mb_buffer=mb_buffer,
        offline_buffer=offline_buffer,
        online_preference_dataset=online_pref_dataset,
        offline_preference_dataset=offline_pref_dataset,
        
        logger=logger,
        rollout_settings=(args.rollout_batch_size, args.rollout_min_length, args.rollout_max_length, args.rollout_min_epoch, args.rollout_max_epoch),
        
        epochs=args.epoch,
        steps_per_epoch=args.step_per_epoch,
        model_retain_epochs=args.model_retain_epochs,
        
        buffer_batch_size=args.batch_size,
        pref_batch_size=args.batch_size,
        init_exploration_steps=args.init_exploration_steps,
        discount_snippet_returns=args.pred_discounted_return,
        
        real_to_mb_ratio=args.real_ratio,
        online_ratio=args.online_ratio,
        add_data_outside_of_model_training=args.add_data_outside_of_model_training,
        eval_episodes=args.eval_episodes,
        lr_scheduler=lr_scheduler,
        dynamics_update_freq=args.dynamics_update_freq
    )

    # train
    if not load_dynamics_model:
        dynamics.train(offline_buffer.sample_all(), None, logger, max_epochs_since_update=5)
        
    # train reward model if needed (using dynamics scaler!!!)
    if not args.load_reward_path:
        reward.train(
            offline_pref_dataset,
            logger,
            holdout_ratio=0.1,
            max_epochs_since_update=10,
            batch_size=args.reward_batch_size
        )
    
    # validate_reward_model(reward, offline_pref_dataset)
    
    # start training policy
    policy_trainer.train()


if __name__ == "__main__":
    train()