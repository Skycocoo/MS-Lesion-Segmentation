## Todo

- V-net (3D) / U-net (2D)
- Display data: [itk-SNAP](http://www.itksnap.org/pmwiki/pmwiki.php?n=Main.HomePage)

- classify / cluster lesion (subtypes)
- cross-validation: 3:1

---

- load FLAIR as input and Consensus as training?
- deal with different input shape
- modify existing unet

---

- soft dice loss / hard dice loss
- find more data to train

## Notes

- [input with variable shape](https://github.com/keras-team/keras/issues/1920)

- [train on batches](https://github.com/keras-team/keras/issues/68)

```py
# each batch would be expected to have different sizes for MSSEG
# Alternatively, without data augmentation / normalization:
for e in range(nb_epoch):
    print("epoch %d" % e)
    for X_train, Y_train in ImageNet(): # these are chunks of ~10k pictures
        model.fit(X_batch, Y_batch, batch_size=32, nb_epoch=1)
```


## HPC setting

syncing code: [atom sftp](https://atom.io/packages/atom-sftp-sync)

[jupyter notebook on hpc](https://wikis.nyu.edu/display/NYUHPC/Running+Jupyter+on+Prince)

[prince tutorials](https://devwikis.nyu.edu/display/NYUHPC/PrinceTutorials)

```shell
$ cd /scratch/<net_id>
$ cp /share/apps/examples/jupyter/run-jupyter.sbatch ./
$ vim run-jupyter.sbatch
# #SBATCH: set up configuration of GPU for jobs
# module: could load existing models so far: module load jupyter-kernels/py3.5

$ sbatch run-jupyter.sbatch
# submit job; create [job-number].out file

$ cat [job-number].out file
# find out the commands to open connect the jupyter notebook with ssh

$ squeue -u yl4217
# check running jobs

```
