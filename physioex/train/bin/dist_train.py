import sys
import subprocess
import os
import portpicker

import uuid

import argparse

def run_distribute():
    
    parser = argparse.ArgumentParser(description='Distributed')
    parser.add_argument('--master', type=str, default='172.16.6.92', help='Master node IP')
    parser.add_argument('--nodes', nargs="+", default=["172.16.2.156"], help='List of nodes IP')   
    parser.add_argument('--n_gpus', type=int, default=1, help='Number of GPUs per node')
    parser.add_argument('--user', type=str, default='guido', help='User')
    parser.add_argument( '--cmd', type=str, default='train', help='Command to execute (train, test, finetune)')
    parser.add_argument( '--cmd_args', type=str, default="", help='Arguments to pass to the command closed by parenteses')
    
    args = parser.parse_args()
        
    
    USER = args.user    
    N_GPUS = args.n_gpus
    MASTER_ADDR = args.master
    NODES_ADDR = args.nodes
    USER = args.user

    CMD = args.cmd
    
    N_NODES = len(NODES_ADDR) + 1
    
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    
    if CMD == "train":
        CMD = os.path.join(current_file_dir, "train.py")
    elif CMD == "test":
        CMD = os.path.join(current_file_dir, "test.py")
    elif CMD == "finetune":
        CMD = os.path.join(current_file_dir, "finetune.py")
    else:
        raise ValueError("Command not recognized")
    
    MASTER_PORT = 3306
    args = args.cmd_args 
    
    if not "-hpc" in args:
        args += " -hpc"
    
    if not f"--num_nodes {N_NODES}" in args and not f"-nn {N_NODES}" in args:
        args += f" --num_nodes {N_NODES}"
    
    if not f"-ck" in args and not "--checkpoint_dir" in args:
        args += f" --checkpoint_dir models/{str(uuid.uuid4())}"
    
    args = f"{CMD} {args}"
    
    # create a tmp file to store the commands
    tmp_file = "master_tmp.sh"
    with open(tmp_file, "w") as f:
        f.write( "echo 'Starting master'\n" )
        f.write(". ../../miniconda3/etc/profile.d/conda.sh\n" )
        f.write("echo 'Activating conda'\n" )
        f.write("conda activate physioex\n")
        f.write( f"echo 'Executing torchrun'\n" )
        f.write(f"torchrun --nproc_per_node={N_GPUS} --nnodes={len(NODES_ADDR) + 1} --node_rank=0 --master_addr={MASTER_ADDR} --master_port={MASTER_PORT} ")
        f.write(args)
        f.write( f" > log_master.log 2>&1" )    
        
    # execute the command on the master node
    subprocess.Popen( f"sh {tmp_file}", shell=True )
    
    # execute the command on the worker nodes
    for i, node in enumerate(NODES_ADDR):
        # create a tmp file to store the commands
        tmp_file = f"worker_tmp_{i}.sh"
        with open(tmp_file, "w") as f:
            f.write( f"echo 'Starting worker {i+1}'\n" )
            f.write( f"cd {os.getcwd()}\n" )
            f.write( f"pwd\n")
            f.write(". ../../miniconda3/etc/profile.d/conda.sh\n" )
            f.write("echo 'Activating conda'\n" )
            f.write("conda activate physioex\n")
            f.write( f"echo 'Executing torchrun'\n" )
            f.write(f"torchrun --nproc_per_node={N_GPUS} --nnodes={len(NODES_ADDR) + 1} --node_rank={i+1} --master_addr={MASTER_ADDR} --master_port={MASTER_PORT} ")
            f.write(args)
            f.write( f" > log_worker.log 2>&1" )
        
        # get the absolute path of tmp file
        tmp_file = os.path.abspath(tmp_file)
        
        # add the ssh command to execute the command on the worker node
        cmd = f"ssh {USER}@{node} 'sh {tmp_file}'"
        
        # execute the command on the worker node
        subprocess.Popen( cmd, shell=True )
    
    
if __name__ == "__main__":
    run_distribute()