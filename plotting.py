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
    
    fig, ax = plt.subplots(6, 1, sharex='all', figsize=(36, 16)) if args.algo != "rambo" else plt.subplots(4, 1, sharex='all', figsize=(36, 16))
    timesteps = df["timestep"]
    
    # get the relevant hparams and add in
    with open(os.path.join(global_results_dir, run_dir, "record", "hyper_param.json")) as f:
        hparams = json.load(f)
        use_scaler = hparams["use_reward_scaler"] if "use_scaler" in hparams else True # default RAMBO just uses it throughout
    
    # set title
    fig.suptitle(f"{'RAMBO with pref-based reward learning' if args.algo == 'rambo_reward_learning' else 'RAMBO GT'} on {args.task} {'with reward scaler' if use_scaler else 'without scaler'}")
    
    # print the reward predictions
    ax[0].plot(timesteps, df["eval/normalized_episode_reward"])
    ax[0].set_ylabel("Normalized episode reward")
    
    if args.algo != "rambo":
        # print the reward BCE loss
        ax[1].plot(timesteps, df["adv_reward_update/reward_bce_loss"])
        ax[1].set_ylabel("Reward BCE loss")
        
        # print the actor + critic losses
        ax[2].plot(timesteps, df["loss/actor"])
        ax[2].set_ylabel("Actor loss")
        
        ax[3].plot(timesteps, (df["loss/critic1"] + df["loss/critic2"]) / 2)
        ax[3].set_ylabel("Average critic loss")
        
        ax[4].plot(timesteps, df["adv_dynamics_update/all_loss"])
        ax[4].set_ylabel("Total objective")
        
        ax[5].plot(timesteps, df["adv_update/v_dataset"])
        ax[5].set_ylabel("Average learned reward on dataset")
    else:
        # print the actor + critic losses
        ax[1].plot(timesteps, df["loss/actor"])
        ax[1].set_ylabel("Actor loss")
        
        ax[2].plot(timesteps, (df["loss/critic1"] + df["loss/critic2"]) / 2)
        ax[2].set_ylabel("Average critic loss")
        
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
    fig, ax = plt.subplots(4, 1, sharex='all', figsize=(24, 16))
    # set title
    fig.suptitle("Comparison of different RAMBO runs")
    
    for algo, run in zip(algos, runs):
        global_dir = f"./log/{task}/{algo}"
        datapath = os.path.join(global_dir, run, "record", "policy_training_progress.csv")
        df = pd.read_csv(datapath)
        
        timesteps = df["timestep"]
        
        # get the relevant hparams and add in
        with open(os.path.join(global_dir, run, "record", "hyper_param.json")) as f:
            hparams = json.load(f)
            use_scaler = hparams["use_reward_scaler"] if "use_reward_scaler" in hparams.keys() else True # default RAMBO just uses it throughout
        lbl = f"{algo}_{'scaled' if use_scaler else 'unscaled'}"
        
        ax[0].plot(timesteps, df["eval/normalized_episode_reward"], label=lbl)
        ax[0].set_ylabel("Normalized episode reward")
        
        if algo != "rambo":
            # print the reward BCE loss
            ax[1].plot(timesteps, df["adv_reward_update/reward_bce_loss"], label=lbl)
            ax[1].set_ylabel("Reward BCE loss")
            
            # print the actor + critic losses
            ax[2].plot(timesteps, df["loss/actor"], label=lbl)
            ax[2].set_ylabel("Actor loss")
            
            ax[3].plot(timesteps, (df["loss/critic1"] + df["loss/critic2"]) / 2, label=lbl)
            ax[3].set_ylabel("Average critic loss")
        else:
            # print the actor + critic losses
            ax[2].plot(timesteps, df["loss/actor"], label=lbl)
            ax[2].set_ylabel("Actor loss")
            
            ax[3].plot(timesteps, (df["loss/critic1"] + df["loss/critic2"]) / 2, label=lbl)
            ax[3].set_ylabel("Average critic loss")
            
        # set x label
        plt.xlabel("Training timesteps")
        plt.legend(loc='lower right')
        
        # save plot
        plot_dir = f"./plots/{task}/full_comp/"
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        plt.savefig(f"{plot_dir}/progress_plot.jpg")
    
if __name__ == "__main__":
    # task = 'halfcheetah-random-v2'
    # algos = ['rambo', 'rambo_reward_learning', 'rambo_reward_learning']
    # runs = ['seed_0&timestamp_23-0615-085124', 'seed_0&timestamp_23-0701-174859', 'seed_0&timestamp_23-0701-174735']
    
    # plot_multiple_runs(task, algos, runs)
    plot_data(get_plot_args())