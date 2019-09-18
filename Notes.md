
## Python virtual environment

Using created virtual environment:

```shell
module load python3/intel/3.6.3 cudnn/9.0v7.0.5 cuda/9.0.176
source ~/ms/py3.6.3/bin/activate
```


## Problems to be solved

Use tensorflow-gpu 1.12.0 + cuda 9 + cudnn 7:

- batch size 4: OOM error due to allocating tensors
- batch size 1: failed to get convolution algorithm; load runtime cudnn library: 7.0.5, but source was compiled with: 7.1.4

==> should check the package from terminal provided by GPU server (use jupyternotebook)
==> should use tenforlow-gpu 1.12.0

---

Use tensorflow-gpu 1.13.0 + cuda 10 + cudnn 7.4:

```
2019-03-21 19:51:54.604769: E tensorflow/stream_executor/cuda/cuda_driver.cc:981] failed to synchronize the stop event: CUDA_ERROR_ILLEGAL_ADDRESS: an illegal memory access was encountered
2019-03-21 19:51:54.605342: E tensorflow/stream_executor/cuda/cuda_timer.cc:55] Internal: error destroying CUDA event in context 0x2b98146d3e30: CUDA_ERROR_ILLEGAL_ADDRESS: an illegal memory access was encountered
2019-03-21 19:51:54.605384: E tensorflow/stream_executor/cuda/cuda_timer.cc:60] Internal: error destroying CUDA event in context 0x2b98146d3e30: CUDA_ERROR_ILLEGAL_ADDRESS: an illegal memory access was encountered
2019-03-21 19:51:54.605457: F tensorflow/stream_executor/cuda/cuda_dnn.cc:194] Check failed: status == CUDNN_STATUS_SUCCESS (7 vs. 0)Failed to set cuDNN stream.
[I 19:52:17.486 NotebookApp] KernelRestarter: restarting kernel (1/5), keep random ports
```

Illegal access: something wrong with the model to cause illegal access? (https://github.com/JuliaGPU/CUDAnative.jl/issues/160#issuecomment-364199100)




## Jupyter Notebook

- Customize Jupyter themes: https://github.com/dunovank/jupyter-themes
- Jupyter theme: ``` jt -t onedork -fs 95 -tfs 11 -nfs 115 -cellw 80% -T```
- Restart kernel: ```ctrl + z```
- Runn all cells: ```cmd + i```
- Check GPU status: new - Terminal - ```nvidia-smi```

## HPC login

[prince tutorials](https://devwikis.nyu.edu/display/NYUHPC/PrinceTutorials)

```shell
$ ssh [net_id]@prince.hpc.nyu.edu # in nyu network
# or
$ ssh [net_id]@gw.hpc.nyu.edu && ssh prince.hpc.nyu.edu
```

sbatch submit job & cancel job:

```shell
$ sbatch jupyter.sbatch # under /scratch/[net_id]
# submit job; create [job-number].out file

$ cat slurm-[job_number].out file
# find out the commands to open connect the jupyter notebook with ssh

$ squeue -u [net_id]
# check running jobs

$ scancel [job_number]
```

---

Using anaconda to install virtual environment:

```shell
$ module load anaconda3/5.3.0 cuda/9.0.176 cudnn/9.0v7.0.5
# $ conda create -n ve python=3.6
$ source activate ve

(ve) $ conda install nb_conda
(ve) $ conda install nbconverter
(ve) $ jupyter notebook
```

Using python to install virtual environment:

```shell
$ mkdir [nameofenv]
$ cd [nameofenv]/
$ module load python3/intel/3.6.3
$ virtualenv --system-site-packages py3.6.3
$ source py3.6.3/bin/activate
$ pip3 install -I http://download.pytorch.org/whl/cpu/torch-0.4.1-cp36-cp36m-linux_x86_64.whl
$ pip3 install -I torchvision
$ pip3 install -I jupyter

# # then inside the SBATCH script:
# module purge
# module load python3/intel/3.6.3
# source ~/[nameofenv]/py3.6.3/bin/activate
```

## Sbatch setup

### Original sbatch files

```
#!/bin/bash

#SBATCH --job-name=jupyterTest2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1

module purge
#module load jupyter-kernels/py2.7 (load other modules)
module load jupyter-kernels/py3.5

port=$(shuf -i 6000-9999 -n 1)

/usr/bin/ssh -N -f -R $port:localhost:$port log-0
/usr/bin/ssh -N -f -R $port:localhost:$port log-1

cat<<EOF

Jupyter server is running on: $(hostname)
Job starts at: $(date)

Step 1 :

If you are working in NYU campus, please open an iTerm window, run command

ssh -L $port:localhost:$port $USER@prince.hpc.nyu.edu

If you are working off campus, you should already have ssh tunneling setup through HPC bastion host,
that you can directly login to prince with command

ssh $USER@prince

Please open an iTerm window, run command

ssh -L $port:localhost:$port $USER@prince

Step 2:

Keep the iTerm windows in the previouse step open. Now open browser, find the line with

The Jupyter Notebook is running at: $(hostname)

the URL is something: http://localhost:${port}/?token=XXXXXXXX (see your token below)

you should be able to connect to jupyter notebook running remotly on prince compute node with above url

EOF

unset XDG_RUNTIME_DIR
if [ "$SLURM_JOBTMP" != "" ]; then
    export XDG_RUNTIME_DIR=$SLURM_JOBTMP
fi

jupyter notebook --no-browser --port $port --notebook-dir=$(pwd)


```

### Current sbatch setup

```
#!/bin/bash

#SBATCH --job-name=jupyterTest2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64GB
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:p40:1

source ~/.bashrc
module purge
module load cuda/9.0.176 cudnn/9.0v7.0.5
conda activate mscond

```
