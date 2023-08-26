import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
import json

def get_plot_args():
    parser = argparse.ArgumentParser(description="plotting")
    parser.add_argument("--task", type=str, default='halfcheetah-random-v2')
    parser.add_argument('--algo', type=str, default='rambo_reward_learning')
    parser.add_argument("--run-id", type=str, default="seed_0&timestamp_23-0703-215834")

    args = parser.parse_args()
    return args

def plot_data(args):
    global_results_dir = f"./log/{args.task}/{args.algo}"
    run_dir = args.run_id
    datapath = os.path.join(global_results_dir, run_dir, "record", "policy_training_progress.csv")
    df = pd.read_csv(datapath)
    
    fig, ax = plt.subplots(8, 1, sharex='all', figsize=(48, 20)) if args.algo != "rambo" else plt.subplots(5, 1, sharex='all', figsize=(40, 16))
    timesteps = df["timestep"]
    
    # get the relevant hparams and add in
    with open(os.path.join(global_results_dir, run_dir, "record", "hyper_param.json")) as f:
        hparams = json.load(f)
        use_scaler = hparams["use_reward_scaler"] if "use_scaler" in hparams else True # default RAMBO just uses it throughout
    
    # set title
    fig.suptitle(f"{'RAMBO with pref-based reward learning' if args.algo == 'rambo_reward_learning' else 'RAMBO GT'} on {args.task} {'with reward scaler' if use_scaler else 'without scaler'}")
    
    # print the reward predictions
    ax[0].plot(timesteps, df["eval/normalized_episode_reward"])
    ax[0].fill_between(timesteps, df["eval/normalized_episode_reward"] - df["eval/normalized_episode_reward_std"], df["eval/normalized_episode_reward"] + df["eval/normalized_episode_reward_std"])
    ax[0].set_ylabel("Normalized episode reward")
    
    if args.algo != "rambo":
        # print the reward BCE loss
        ax[1].plot(timesteps, df["adv_reward_update/reward_bce_loss"])
        ax[1].set_ylabel("Reward BCE loss")
        
        # print the losses that actually matter
        ax[2].plot(timesteps, df["adv_update/adv_dynamics_loss"])
        ax[2].set_ylabel("Adversarial dynamics loss (MLE/MSE)")
        
        ax[3].plot(timesteps, df["adv_update/adv_reward_loss"])
        ax[3].set_ylabel("Adversarial reward loss (v_pi - v_dataset)")
        
        ax[4].plot(timesteps, df["adv_update/v_pi"])
        ax[4].set_ylabel("Value of policy in model")
        
        ax[5].plot(timesteps, df["adv_update/v_dataset"])
        ax[5].set_ylabel("Value of dataset transitions")
        
        ax[6].plot(timesteps, df["adv_dynamics_update/all_loss"])
        ax[6].set_ylabel("Total adversarial objective")
        
        ax[7].plot(timesteps, df["adv_update/v_dataset"])
        ax[7].set_ylabel("Average learned reward on dataset")
    else:
        # print the actor + critic losses
        ax[1].plot(timesteps, df["adv_dynamics_update/v_pi"])
        ax[1].set_ylabel("Policy value (model-based)")
        
        ax[2].plot(timesteps, df["adv_dynamics_update/v_dataset"])
        ax[2].set_ylabel("Dataset value")
        
        ax[3].plot(timesteps, df["adv_dynamics_update/all_loss"])
        ax[3].set_ylabel("Total adversarial loss")
        
        ax[4].plot(timesteps, df["adv_dynamics_update/adv_loss"])
        ax[4].set_ylabel("total dynamics objective (V^pi_phi)")
        
    # set x label
    plt.xlabel("Training timesteps")
    
    # save plot
    plot_dir = f"./plots/{args.task}/{args.algo}/{run_dir}"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    plt.savefig(f"{plot_dir}/progress_plot.jpg")
    
    
def plot_multiple_runs(task, algos, runs):
    assert len(runs) == len(algos), "Need one run per algo"
    
    # set up plots
    fig, ax = plt.subplots(7, 1, sharex='all', figsize=(30, 30)) if "rambo_reward_learning" in algos else plt.subplots(3, 1, sharex='all', figsize=(24, 16))
    # set title
    fig.suptitle(f"Comparison of different RAMBO-like runs on {task}")
    
    for algo, run in zip(algos, runs):
        global_dir = f"./log/{task}/{algo}"
        datapath = os.path.join(global_dir, run, "record", "policy_training_progress.csv")
        df = pd.read_csv(datapath)
        
        timesteps = df["timestep"]
        
        # get the relevant hparams and add in
        with open(os.path.join(global_dir, run, "record", "hyper_param.json")) as f:
            hparams = json.load(f)
            use_scaler = hparams["use_reward_scaler"] if "use_reward_scaler" in hparams.keys() else True # default RAMBO just uses it throughout
        lbl = f"{algo}{'' if use_scaler else '_unscaled'}"
        
        ax[0].plot(timesteps, df["eval/normalized_episode_reward"], label=lbl)
        ax[0].fill_between(timesteps, df["eval/normalized_episode_reward"] - df["eval/normalized_episode_reward_std"], df["eval/normalized_episode_reward"] + df["eval/normalized_episode_reward_std"])
        ax[0].set_ylabel("Normalized episode reward")
        
        if algo not in ["rambo", "rambo_relabeled"]:
            # print the reward BCE loss
            ax[1].plot(timesteps, df["adv_reward_update/reward_bce_loss"], label=lbl)
            ax[1].set_ylabel("Reward BCE loss")
            
            # print the losses that actually matter
            ax[2].plot(timesteps, df["adv_update/adv_dynamics_loss"], label=lbl)
            ax[2].set_ylabel("Adv dyn loss (MLE/MSE)")
            
            ax[3].plot(timesteps, df["adv_update/adv_reward_loss"], label=lbl)
            ax[3].set_ylabel("Adv rew loss (v_pi - v_D)")
            
            ax[4].plot(timesteps, df["adv_update/v_pi"], label=lbl)
            ax[4].set_ylabel("Value of policy in model")
            
            ax[5].plot(timesteps, df["adv_update/v_dataset"], label=lbl)
            ax[5].set_ylabel("Value of dataset transitions")
            
            ax[6].plot(timesteps, df["adv_dynamics_update/all_loss"], label=lbl)
            ax[6].set_ylabel("Total adversarial objective")
        else:
            # print everything corresponding to the above
            ax[3].plot(timesteps, df["adv_dynamics_update/v_pi"] - df["adv_dynamics_update/v_dataset"], label=lbl)
            ax[3].set_ylabel("Adv rew loss (v_pi - v_D)")
            
            ax[4].plot(timesteps, df["adv_dynamics_update/v_pi"], label=lbl)
            ax[4].set_ylabel("Value of policy in model")
            
            ax[5].plot(timesteps, df["adv_dynamics_update/v_dataset"], label=lbl)
            ax[5].set_ylabel("Value of dataset transitions")
            
            ax[6].plot(timesteps, df["adv_dynamics_update/all_loss"], label=lbl)
            ax[6].set_ylabel("Total adversarial objective")
            
        # set x label
        plt.xlabel("Training timesteps")
        plt.legend(loc='lower right')
        
        # save plot
        only_one_algo = len(set(algos)) == 1
        plot_dir = f"./plots/{task}/{algos[0] + '_comparison' if only_one_algo else 'full_comp'}/"
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        plt.savefig(f"{plot_dir}/progress_plot.jpg")
    
if __name__ == "__main__":
    # task = 'halfcheetah-medium-v2'
    # algos = ['rambo', 'rambo_relabeled', 'rambo_reward_learning']
    # runs = ['seed_0&timestamp_23-0715-110036', 'seed_0&timestamp_23-0716-170644', 'seed_0&timestamp_23-0811-202113']
    
    # plot_multiple_runs(task, algos, runs)
    plot_data(get_plot_args())