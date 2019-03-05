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
