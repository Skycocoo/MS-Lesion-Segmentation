## HPC login

``` shell
$ ssh <net_id>@prince.hpc.nyu.edu # in nyu network
# or
$ ssh <net_id>@gw.hpc.nyu.edu && ssh prince.hpc.nyu.edu


$ sbatch run-jupyter.sbatch # under ~/
# submit job; create [job-number].out file

$ cat [job-number].out file
# find out the commands to open connect the jupyter notebook with ssh

$ squeue -u <net_id>
# check running jobs
```

## Jupyter Notebook

- Restart kernel: ```ctrl + z```
- Runn all cells: ```cmd + i```
- Check GPU status: open new - Terminal - ```nvidia-smi```


## Python virtual environment

Using created virtual environment:

```shell
module load python3/intel/3.6.3 cudnn/9.0v7.0.5
source ~/pyenv/py3.6.3/bin/activate
```



Using anaconda to install:

```shell
$ module load anaconda3/5.3.0 cuda/9.0.176 cudnn/9.0v7.0.5
# $ conda create -n ve python=3.6
$ source activate ve

(ve) $ conda install nb_conda
(ve) $ conda install nbconverter
(ve) $ jupyter notebook
```

Using python to install:

```shell
$ mkdir pytorch-cpu
$ cd pytorch-cpu/
$ module load python3/intel/3.6.3
$ virtualenv --system-site-packages py3.6.3
$ source py3.6.3/bin/activate
$ pip3 install -I http://download.pytorch.org/whl/cpu/torch-0.4.1-cp36-cp36m-linux_x86_64.whl
$ pip3 install -I torchvision
$ pip3 install -I jupyter

# # then inside the SBATCH script:
# module purge
# module load python3/intel/3.6.3
# source ~/pytorch-cpu/py3.6.3/bin/activate
```










## original sbatch files

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
