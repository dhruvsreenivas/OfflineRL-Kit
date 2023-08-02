import os
from datetime import datetime
import argparse
import uuid

# ==== SWEEP PARAMS ====
ENVS = ["walker2d-medium-v2", "walker2d-medium-expert-v2", "hopper-medium-replay-v2"]
SL_DYNAMICS_COEFS = [0.01, 0.1, 1.0, 10.0] # how much to weight SL dynamics loss vs. BCE loss (keep reward loss at 1, it's scaled I think)
ADV_COEFS = [0, 3e-6, 3e-5, 3e-4] # how much to weight adversarial loss vs. supervised loss
SEGMENT_LENGTHS = [60]


parser = argparse.ArgumentParser()
parser.add_argument('--nhrs', type=int, default=24)
parser.add_argument('--base_save_dir', default=f'{os.path.abspath(os.path.join(os.getcwd()))}')
parser.add_argument('--output-dirname', default='our_method_sweep_out')
parser.add_argument('--error-dirname', default='our_method_sweep_err')
parser.add_argument('--dryrun', action='store_true')
parser.add_argument('--use-sun', action="store_true")
args = parser.parse_args()

# ===== CREATE JOBS ====

jobs = []
params = [
    (task, adv_dynamics_coef, adv_coef, seglen)
    for task in ENVS
    for adv_dynamics_coef in SL_DYNAMICS_COEFS
    for adv_coef in ADV_COEFS
    for seglen in SEGMENT_LENGTHS
]

for param in params:
    # (seed, task, alg, ref_prop, kl, reward) = param
    # (seed, task, alg, reward, clip) = param
    env, adv_dynamics_coef, adv_coef, seglen = param
    
    # create experiment name for logging
    name = f"{env}_dynweight_{adv_dynamics_coef}_advweight_{adv_coef}_seglen_{seglen}"
    
    # create the command
    cmd = 'python run_example/run_rambo_reward_learning.py '
    cmd += f'--task {env} --adv-weight {adv_coef} --use-reward-scaler True '
    cmd += f'--rollout-length 5 --segment-length {seglen} --adv-dynamics-coef {adv_dynamics_coef} '

    jobs.append((cmd, name, param))

# this you can also hardcode
output_dir = os.path.join(args.base_save_dir, args.output_dirname)
error_dir = os.path.join(args.base_save_dir, args.error_dirname)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
print("Output Directory: %s" % output_dir)

if not os.path.exists(error_dir):
    os.makedirs(error_dir)
print("Error Directory: %s" % error_dir)

id_name = uuid.uuid4()
now_name = f'{output_dir}/now_{id_name}.txt'
was_name = f'{output_dir}/was_{id_name}.txt'
log_name = f'{output_dir}/log_{id_name}.txt'
err_name = f'{output_dir}/err_{id_name}.txt'
num_commands = 0
jobs = iter(jobs)
done = False
threshold = 999

while not done:
    for (cmd, name, params) in jobs:

        if os.path.exists(now_name):
            file_logic = 'a'  # append if already exists
        else:
            file_logic = 'w'  # make a new file if not
            print(f'creating new file: {now_name}')

        with open(now_name, 'a') as nowfile,\
             open(was_name, 'a') as wasfile,\
             open(log_name, 'a') as output_namefile,\
             open(err_name, 'a') as error_namefile:

            if nowfile.tell() == 0:
                print(f'a new file or the file was empty: {now_name}')

            now = datetime.now()
            datetimestr = now.strftime("%m%d_%H%M:%S.%f")

            num_commands += 1
            nowfile.write(f'{cmd}\n')
            wasfile.write(f'{cmd}\n')

            output_dir = os.path.join(args.base_save_dir, args.output_dirname)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            output_namefile.write(f'{(os.path.join(output_dir, name))}.log\n')
            error_namefile.write(f'{(os.path.join(error_dir, name))}.error\n')
            if num_commands == threshold:
                break
    
    if num_commands != threshold:
        done = True


    # Make a {name}.slurm file in the {output_dir} which defines this job.
    #slurm_script_path = os.path.join(output_dir, '%s.slurm' % name)
    start=1
    slurm_script_path = os.path.join(output_dir, f'boost_{start}_{num_commands}.slurm')
    slurm_command = "sbatch %s" % slurm_script_path

    # Make the .slurm file
    with open(slurm_script_path, 'w') as slurmfile:
        slurmfile.write("#!/bin/bash\n")
        slurmfile.write(f"#SBATCH --array=1-{num_commands}\n")
        slurmfile.write("#SBATCH -o /home/ds844/OfflineRL-Kit/slurm/multijob_output/adv_pref_rew_learn_sweep_%j.out\n")
        slurmfile.write("#SBATCH -e /home/ds844/OfflineRL-Kit/slurm/multijob_error/adv_pref_rew_learn_sweep_%j.err\n")
        slurmfile.write("#SBATCH --requeue\n")
        slurmfile.write("#SBATCH -t %d:00:00\n" % args.nhrs)
        
        # cores requested
        slurmfile.write("#SBATCH -N 1\n")
        slurmfile.write("#SBATCH -n 1\n")
        slurmfile.write("#SBATCH --gres=gpu:3090:1")
        slurmfile.write("#SBATCH --mem=30G\n")

        # Greene (can use sun here if needed)
        # slurmfile.write("#SBATCH --qos gpu48\n")
        partition = "sun" if args.use_sun else "default_partition"
        slurmfile.write(f"#SBATCH --partition={partition}\n")

        slurmfile.write("\n")
        slurmfile.write("source /share/apps/anaconda3/2021.11/etc/profile.d/conda.sh\n")
        slurmfile.write("cd /home/ds844/OfflineRL-Kit\n")
        slurmfile.write("conda activate offline_rlkit\n")
        slurmfile.write(f"srun --output=$(head -n $SLURM_ARRAY_TASK_ID {log_name} | tail -n 1) --error=$(head -n    $SLURM_ARRAY_TASK_ID {err_name} | tail -n 1)  $(head -n $SLURM_ARRAY_TASK_ID {now_name} | tail -n 1)\n" )
        slurmfile.write("\n")

    if not args.dryrun:
        os.system("%s &" % slurm_command)

    num_commands = 0
    id_name = uuid.uuid4()
    now_name = f'{args.base_save_dir}/output/now_{id_name}.txt'
    was_name = f'{args.base_save_dir}/output/was_{id_name}.txt'
    log_name = f'{args.base_save_dir}/output/log_{id_name}.txt'
    err_name = f'{args.base_save_dir}/output/err_{id_name}.txt'
