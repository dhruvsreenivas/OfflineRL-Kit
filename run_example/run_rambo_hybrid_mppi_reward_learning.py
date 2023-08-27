import warnings
warnings.filterwarnings('ignore')
import getpass
import argparse
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
from offlinerlkit.utils.termination_fns import get_termination_fn, obs_unnormalization
from offlinerlkit.buffer import ReplayBuffer, PreferenceDataset
from offlinerlkit.utils.logger import Logger, make_log_dirs
from offlinerlkit.policy_trainer import HybridPrefMBPolicyMppiSacTrainer
from offlinerlkit.policy import HybridRAMBORewardLearningPolicy

"""
suggested hypers

halfcheetah-medium-v2: rollout-length=5, adv-weight=3e-4
hopper-medium-v2: rollout-length=5, adv-weight=3e-4
walker2d-medium-v2: rollout-length=5, adv-weight=0
halfcheetah-medium-replay-v2: rollout-length=5, adv-weight=3e-4
hopper-medium-replay-v2: rollout-length=5, adv-weight=3e-4
walker2d-medium-replay-v2: rollout-length=5, adv-weight=0
halfcheetah-medium-expert-v2: rollout-length=5, adv-weight=0
hopper-medium-expert-v2: rollout-length=5, adv-weight=0
walker2d-medium-expert-v2: rollout-length=2, adv-weight=3e-4
"""


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--netid", type=str, default=None)
    parser.add_argument("--algo-name", type=str, default="rambo_hybrid_mppi_reward_learning")
    parser.add_argument("--task", type=str, default="halfcheetah-random-v2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--dynamics-lr", type=float, default=3e-4)
    parser.add_argument("--dynamics-adv-lr", type=float, default=3e-4)
    parser.add_argument("--hidden-dims", type=int, nargs='*', default=[256, 256])
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--auto-alpha", default=True)
    parser.add_argument("--target-entropy", type=int, default=None)
    parser.add_argument("--alpha-lr", type=float, default=1e-4)

    parser.add_argument("--dynamics-hidden-dims", type=int, nargs='*', default=[200, 200, 200, 200])
    parser.add_argument("--dynamics-dropout-probs", type=float, nargs='*', default=[0.0, 0.0, 0.0, 0.0])
    parser.add_argument("--dynamics-weight-decay", type=float, nargs='*', default=[2.5e-5, 5e-5, 7.5e-5, 7.5e-5, 1e-4])
    parser.add_argument("--n-ensemble", type=int, default=7)
    parser.add_argument("--n-elites", type=int, default=5)
    parser.add_argument("--rollout-freq", type=int, default=250) # set to 1
    parser.add_argument("--dynamics-update-freq", type=int, default=1000) # set to 1
    parser.add_argument("--adv-batch-size", type=int, default=256)
    parser.add_argument("--rollout-batch-size", type=int, default=50000)
    parser.add_argument("--rollout-length", type=int, default=5)
    parser.add_argument("--adv-weight", type=float, default=3e-4)
    parser.add_argument("--model-retain-epochs", type=int, default=5)
    parser.add_argument("--real-ratio", type=float, default=0.5)
    parser.add_argument("--load-dynamics-path", type=str, default=None)
    parser.add_argument("--max-dynamics-pretrain-epochs", type=int, default=500)

    parser.add_argument("--epoch", type=int, default=2000)
    parser.add_argument("--step-per-epoch", type=int, default=1000)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--include-ent-in-adv", type=bool, default=False)
    parser.add_argument("--load-bc-path", type=str, default=None)
    parser.add_argument("--bc-lr", type=float, default=1e-4)
    parser.add_argument("--bc-epoch", type=int, default=50)
    parser.add_argument("--bc-batch-size", type=int, default=256)
    
    # reward learning args
    parser.add_argument("--reward-hidden-dims", type=int, nargs='*', default=[200, 200, 200, 200])
    parser.add_argument("--reward-dropout-probs", type=float, nargs='*', default=[0.0, 0.0, 0.0, 0.0])
    parser.add_argument("--reward-weight-decay", type=float, nargs='*', default=[0.0, 0.0, 0.0, 0.0, 0.0])
    parser.add_argument("--reward-lr", type=float, default=3e-4)
    parser.add_argument("--n-reward_models", type=int, default=7)
    parser.add_argument("--n-reward-elites", type=int, default=5)
    parser.add_argument("--reward-with-action", type=bool, default=True)
    parser.add_argument("--segment-length", type=int, default=15)
    parser.add_argument("--load-reward-path", type=str, default=None)
    parser.add_argument("--reward-penalty-coef", type=float, default=0.0)
    parser.add_argument("--normalize-reward-eval", type=bool, default=False)
    parser.add_argument("--reward_batch_size", type=int, default=256)
    parser.add_argument("--reward-uncertainty-mode", type=str, default="aleatoric")
    parser.add_argument("--reward-final-activation", type=str, default="none")
    parser.add_argument("--reward-soft-clamp", type=bool, default=False)
    parser.add_argument("--sl-dynamics-coef", type=float, default=1.0)
    parser.add_argument("--adv-dynamics-coef", type=float, default=1.0)
    parser.add_argument("--adv-reward-coef", type=float, default=1.0)
    parser.add_argument("--sl-reward-coef", type=float, default=1.0)
    parser.add_argument("--use-reward-scaler", type=bool, default=True, help='whether to use dynamics scaler for reward learning or not')
    parser.add_argument("--pred-discounted-return", type=bool, default=False, help='whether to predict discounted return as opposed to full return.')

    parser.add_argument("--load-std-path", type=str, default=None) # load dynamics std path (for ant env)
    parser.add_argument("--fix-logvar-range", type=bool, default=False) # fixed min and max logvar for dynamics
    parser.add_argument("--online-ratio", type=float, default=0.5)
    parser.add_argument("--online-preference-ratio", type=float, default=0.5)
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
    # get netid
    netid = args.netid if args.netid is not None else getpass.getuser()
    # create env and dataset
    env = gym.make(args.task)
    dataset = d4rl.qlearning_dataset(env)
    args.obs_shape = env.observation_space.shape
    args.action_dim = np.prod(env.action_space.shape)
    args.max_action = env.action_space.high[0]

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
    
    if args.auto_alpha:
        target_entropy = args.target_entropy if args.target_entropy \
            else -np.prod(env.action_space.shape)

        args.target_entropy = target_entropy

        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        alpha = (target_entropy, log_alpha, alpha_optim)
    else:
        alpha = args.alpha

    # create buffers
    real_buffer = ReplayBuffer(
        buffer_size=len(dataset["observations"]),
        obs_shape=args.obs_shape,
        obs_dtype=np.float32,
        action_dim=args.action_dim,
        action_dtype=np.float32,
        device=args.device
    )
    real_buffer.load_dataset(dataset)
    obs_mean, obs_std = real_buffer.normalize_obs() # ONLY OBSERVATIONS ARE NORMALIZED!
    fake_buffer_size = args.step_per_epoch // args.rollout_freq * args.model_retain_epochs * args.rollout_batch_size * args.rollout_length
    fake_buffer = ReplayBuffer(
        buffer_size=fake_buffer_size, 
        obs_shape=args.obs_shape,
        obs_dtype=np.float32,
        action_dim=args.action_dim,
        action_dtype=np.float32,
        device=args.device
    )
    online_buffer = ReplayBuffer(
        buffer_size=len(dataset["observations"]),
        obs_shape=args.obs_shape,
        obs_dtype=np.float32,
        action_dim=args.action_dim,
        action_dtype=np.float32,
        device=args.device
    )
    
    dataset_path = f"/home/{netid}/OfflineRL-Kit/offline_data/{args.task}_snippet_preference_dataset_seglen{args.segment_length}_deterministic.pt" # here, observations are not normalized
    pref_dataset = torch.load(dataset_path)
    pref_dataset.normalize_obs(obs_mean, obs_std) # normalize everything, same scale as regular replay buffer
    pref_dataset.device = args.device

    if args.load_std_path is not None:
        fix_std = torch.load(args.load_std_path)
        fix_std = torch.clamp(fix_std, 1e-5)
        fix_logvar = torch.log(torch.pow(fix_std, 2))
    else:
        fix_logvar = None

    # create dynamics
    dynamics_model = EnsembleDynamicsModel(
        obs_dim=np.prod(args.obs_shape),
        action_dim=args.action_dim,
        hidden_dims=args.dynamics_hidden_dims,
        num_ensemble=args.n_ensemble,
        num_elites=args.n_elites,
        weight_decays=args.dynamics_weight_decay,
        with_reward=False,
        dropout_probs=args.dynamics_dropout_probs,
        device=args.device,
        fix_logvar=fix_logvar,
        fix_logvar_range=args.fix_logvar_range,
    )
    dynamics_optim = torch.optim.Adam(
        dynamics_model.parameters(),
        lr=args.dynamics_lr
    )
    dynamics_adv_optim = torch.optim.Adam(
        dynamics_model.parameters(), 
        lr=args.dynamics_adv_lr
    )
    dynamics_scaler = StandardScaler()
    termination_fn = obs_unnormalization(get_termination_fn(task=args.task), obs_mean, obs_std)
    dynamics = EnsembleDynamics(
        dynamics_model,
        dynamics_optim,
        dynamics_scaler,
        termination_fn,
    )
    
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
    policy_scaler = StandardScaler(mu=obs_mean, std=obs_std)
    policy = HybridRAMBORewardLearningPolicy(
        dynamics,
        reward,
        actor,
        critic1,
        critic2,
        actor_optim, 
        critic1_optim, 
        critic2_optim, 
        dynamics_adv_optim,
        reward_optim,
        tau=args.tau, 
        gamma=args.gamma, 
        alpha=alpha, 
        adv_weight=args.adv_weight, # lambda parameter
        adv_rollout_length=args.rollout_length, 
        adv_rollout_batch_size=args.adv_batch_size,
        sl_dynamics_loss_coef=args.sl_dynamics_coef, # how much do we weight the dynamics SL loss vs. other supervised losses... SUPERVISED LEARNING LOSS
        adv_dynamics_loss_coef=args.adv_dynamics_coef,
        adv_reward_loss_coef=args.adv_reward_coef, # how much to weight the reward adversarial loss (V_pi - V_dataset) vs. the dynamics loss (V^pi_phi)
        sl_reward_loss_coef=args.sl_reward_coef,
        include_ent_in_adv=args.include_ent_in_adv,
        scaler=policy_scaler,
        online_ratio=args.online_ratio,
        online_preference_ratio=args.online_preference_ratio,
        device=args.device
    ).to(args.device)
    
    # log
    log_dirs = make_log_dirs(args.task, args.algo_name, args.seed, vars(args))
    # key: output file name, value: output handler type
    output_config = {
        "consoleout_backup": "stdout",
        "policy_training_progress": "csv",
        "dynamics_training_progress": "csv",
        "reward_training_progress": "csv",
        "tb": "tensorboard"
    }
    logger = Logger(log_dirs, output_config)
    logger.log_hyperparameters(vars(args))
    
    online_preference_dataset = PreferenceDataset(offline_data=[], device=pref_dataset.device)
    # create policy trainer
    policy_trainer = HybridPrefMBPolicyMppiSacTrainer(
        policy=policy,
        eval_env=env,
        offline_preference_dataset=pref_dataset,
        online_preference_dataset=online_preference_dataset,
        real_buffer=real_buffer,
        fake_buffer=fake_buffer,
        online_buffer=online_buffer,
        logger=logger,
        rollout_setting=(args.rollout_freq, args.rollout_batch_size, args.rollout_length),
        dynamics_update_freq=args.dynamics_update_freq,
        epoch=args.epoch,
        step_per_epoch=args.step_per_epoch,
        batch_size=args.batch_size,
        eval_episodes=args.eval_episodes,
        real_ratio=args.real_ratio,
        gamma=args.gamma,
        segment_length=args.segment_length
    )

    # train pure dynamics
    if args.load_dynamics_path:
        dynamics.load(args.load_dynamics_path)
    else:
        dynamics.train(
            real_buffer.sample_all(),
            None,
            logger,
            holdout_ratio=0.1,
            logvar_loss_coef=0.001,
            max_epochs=args.max_dynamics_pretrain_epochs,
            max_epochs_since_update=10
        )
        # sets dynamics scaler here, so have stats for reward
    
    # pretrain policy
    if args.load_bc_path:
        policy.load(args.load_bc_path)
        policy.to(args.device)
    else:
        policy.train()
        real_batch = real_buffer.sample(batch_size=args.batch_size)
        dataset = {"real": real_batch}
        policy.learn(dataset)
        import os
        torch.save(policy.state_dict(), os.path.join(logger.model_dir, "rambo_reward_learn_pretrain.pth"))
    
    # train reward model if needed (using dynamics scaler!!!)
    if not args.load_reward_path:
        reward.train(
            pref_dataset,
            logger,
            holdout_ratio=0.1,
            max_epochs_since_update=10,
            batch_size=args.reward_batch_size
        )
    
    validate_reward_model(reward, pref_dataset)

    # train policy
    policy_trainer.train()


if __name__ == "__main__":
    train()