import os
os.chdir("/scratch/yl4217/MS-Lesion-Segmentation/")
weight_path = ['/model/weight/fold0_weights-03-0.38.hdf5']
weight_name = ['dice']

from model.data import *
from model.model import *
from model.recon import *

import matplotlib.pyplot as plt
import h5py
import numpy as np
import nibabel as nib
from collections import defaultdict

config = {}
config["weights_file"] = os.getcwd() + '/model/weight'
config["patch_size"] = (64, 64, 64)  # switch to None to train on the whole image
config["patch_gap"] = 16
config["batch_size"] = 10
config["kfold"] = 5

config["input_shape"] = (1, None, None, None)
config["depth"] = 4 # depth of layers for V/Unet
config["n_base_filters"] = 32
config["pool_size"] = (2, 2, 2)  # pool size for the max pooling operations
config["deconvolution"] = True  # if False, will use upsampling instead of deconvolution

config["patience"] = 10  # learning rate will be reduced after this many epochs if the validation loss is not improving
config["early_stop"] = 10  # training will be stopped after this many epochs without the validation loss improving
config["initial_learning_rate"] = 0.00001
config["learning_rate_drop"] = 0.5  # factor by which the learning rate will be reduced
config["n_epochs"] = 1

d = Data()
d.load_data(config["patch_size"])
# set up valid index
train_num, valid_num = d.prekfold(config["patch_size"], config["patch_gap"], config["batch_size"], config["kfold"])

for i_weight in range(len(weight_path)):
    print("loading weight: ", weight_name[i_weight])
    model = unet_model_3d(input_shape=config["input_shape"],
                                  pool_size=config["pool_size"],
                                  initial_learning_rate=config["initial_learning_rate"],
                                  deconvolution=config["deconvolution"],
                                  depth=config["depth"],
                                  n_base_filters=config["n_base_filters"])
    model.load_weights(os.getcwd() + weight_path[i_weight]) 
    
    fold_index = 0
    for i in d.valid_index:
        j = d.valid_index[i][fold_index]
        normal = Reconstruct(j, d.data[i][j][0].shape, config["patch_size"], False)
        image = Reconstruct(j, d.data[i][j][0].shape, config["patch_size"], False)
        target = Reconstruct(j, d.data[i][j][0].shape, config["patch_size"], False)
        for ind in range(d.patch_index[i][j].shape[0]):
            index = d.patch_index[i][j][ind]
            image_i = np.expand_dims(d.data[i][j][0][
                             index[0]:index[0]+d.patch_size[0], 
                             index[1]:index[1]+d.patch_size[1], 
                             index[2]:index[2]+d.patch_size[2]], axis=0)
            target_i = np.expand_dims(d.data[i][j][1][
                             index[0]:index[0]+d.patch_size[0], 
                             index[1]:index[1]+d.patch_size[1], 
                             index[2]:index[2]+d.patch_size[2]], axis=0)
            result = model.predict([image_i[None, :]])
            normal.add(result, index)
            image.add(image_i, index)
            target.add(target_i, index)
        dir_name = './model/h5df_data/recon/' + weight_name[i_weight] + '/'
        os.makedirs(os.path.dirname(dir_name), exist_ok=True)
        file_name = '/recon/' + weight_name[i_weight] + '/'+ str(d.data[i][j][0].shape)
        normal.store(file_name + "_uniform_output")
        image.store(file_name + "_input")
        target.store(file_name + "_target")

print("finish reconstructing image")


def fetch_file():
    path = os.getcwd() + '/model/h5df_data/recon/'
    root, sub_dir, _ = next(os.walk(path))
    total = {}
    for sub in sub_dir:
        _, _, sub_files = next(os.walk(root + sub))
        uniform = {}
        weight = {}
        target = {}
        for file in sub_files:
            if "nii" not in file and "threshold" not in file:
                cur_file = h5py.File(root + sub + '/' + file, 'r')
                if "weight" in file:
                    weight[str(cur_file["shape"][()])]= cur_file
                if "uniform" in file:
                    uniform[str(cur_file["shape"][()])]= cur_file
                if "target" in file:
                    target[str(cur_file["shape"][()])]= cur_file
        total[sub] = [target, uniform, weight]
    return total

def fetch_threshold_file():
    path = os.getcwd() + '/model/h5df_data/recon/'
    root, sub_dir, _ = next(os.walk(path))
    total = []
    for sub in sub_dir:
        _, _, sub_files = next(os.walk(root + sub))
        for file in sub_files:
            if "nii.gz" in file and "threshold" in file:
                total.append(path + sub + '/' + file)
    return total

def dice(y_true, y_pred, smooth=1.):
    y_true_f = np.array(y_true).flatten()
    y_pred_f = np.array(y_pred).flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    # tensorflow computation graph: will not configure print as one of the graph, unless using tf.Print()
    return (2.*intersection+smooth) / (np.sum(y_true_f)+np.sum(y_pred_f)+smooth)

def calc_thres(output, target):
    it = np.arange(0, 1.01, 0.01)
    dice_thre = []
    for i in it:
        dice_thre.append(dice(target, output>i))
    return dice_thre


total = fetch_file()
table = {0: "target", 1: "uniform", 2: "weight"}

for i in total:
    print(i)
    for j in range(len(total[i])):
        print(table[j])
        for k in total[i][j]:
            print(total[i][j][k])

path = os.getcwd() + '/model/h5df_data/recon/'

for i in total:
    # all, binary, normal
    # target: total[i][0]
    # uniform: total[i][1]
    # weight: total[i][2]
    print(i)
    # for j in range(1, 3):
    j = 1
    threshold = np.zeros(101)
    for k in total[i][0]:
        threshold += calc_thres(total[i][j][k]["data"][:], total[i][0][k]["data"][:])
    threshold /= len(total[i][0])
    plt.plot(np.arange(0, 1.01, 0.01), threshold)
    opt_dice = threshold[np.argmax(threshold)]
    opt_thre = np.arange(0, 1.01, 0.01)[np.argmax(threshold)]
    plt.savefig(path + i + '/' + table[j] + '_dice_' + str(opt_dice) + '_opt_' + str(opt_thre) + '_threshold.png')
    plt.clf()
    plt.cla()
    
    for k in total[i][0]:
        nib.save(nib.Nifti1Image(np.array(total[i][j][k]["data"][:]>opt_thre).astype(int), np.eye(4)), path + i + '/' + os.path.basename(total[i][j][k].filename) + "_" + table[j] + "_" + str(opt_thre) + "_threshold.nii.gz")

        
print("finish determining the optimal threshold")

threshold_total = fetch_threshold_file()
for i in threshold_total:
    image = nib.load(i)
    nib.save(nib.Nifti1Image(image.get_fdata(), np.eye(4)), i)

print("finish restoring thresholded image")
