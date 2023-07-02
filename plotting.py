import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
import json

def get_plot_args():
    parser = argparse.ArgumentParser(description="plotting")
    parser.add_argument("--task", type=str, default='halfcheetah-random-v2')
    parser.add_argument('--algo', type=str, default='rambo_reward_learning')
    parser.add_argument("--run-id", type=str, default="seed_0&timestamp_23-0701-174859")

    args = parser.parse_args()
    return args

def plot_data(args):
    global_results_dir = f"./log/{args.task}/{args.algo}"
    run_dir = args.run_id
    datapath = os.path.join(global_results_dir, run_dir, "record", "policy_training_progress.csv")
    df = pd.read_csv(datapath)
    
    fig, ax = plt.subplots(4, 1, sharex='all', figsize=(24, 16))
    timesteps = df["timestep"]
    
    # get the relevant hparams and add in
    with open(os.path.join(global_results_dir, run_dir, "record", "hyper_param.json")) as f:
        hparams = json.load(f)
        use_scaler = hparams["use_reward_scaler"]
    
    # set title
    fig.suptitle(f"{'RAMBO with pref-based reward learning' if args.algo == 'rambo_reward_learning' else 'RAMBO GT'} on {args.task} {'with reward scaler' if use_scaler else 'without scaler'}")
    
    # print the reward predictions
    ax[0].plot(timesteps, df["eval/normalized_episode_reward"])
    ax[0].set_ylabel("Normalized episode reward")
    
    # print the reward BCE loss
    ax[1].plot(timesteps, df["adv_reward_update/reward_bce_loss"])
    ax[1].set_ylabel("Reward BCE loss")
    
    # print the actor + critic losses
    ax[2].plot(timesteps, df["loss/actor"])
    ax[2].set_ylabel("Actor loss")
    
    ax[3].plot(timesteps, (df["loss/critic1"] + df["loss/critic2"]) / 2)
    ax[3].set_ylabel("Average critic loss")
    
    # save plot
    plot_dir = f"./plots/{args.task}/{args.algo}/{run_dir}"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    plt.savefig(f"{plot_dir}/progress_plot.jpg")
    
    
if __name__ == "__main__":
    args = get_plot_args()
    plot_data(args)