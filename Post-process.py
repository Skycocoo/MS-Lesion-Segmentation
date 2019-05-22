import os

os.chdir("/scratch/yl4217/MS-Lesion-Segmentation/")

import matplotlib.pyplot as plt
import h5py
import numpy as np
import nibabel as nib
from collections import defaultdict

def fetch_file():
    path = os.getcwd() + '/model/h5df_data/recon/'
    root, sub_dir, _ = next(os.walk(path))
    total = {}
#     uniform = defaultdict(list)
#     weight = defaultdict(list)
#     target = defaultdict(list)
    for sub in sub_dir:
        _, _, sub_files = next(os.walk(root + sub))
        uniform = {}
        weight = {}
        target = {}
        for file in sub_files:
#             print(file)
            if "nii" not in file and "threshold" not in file:
                cur_file = h5py.File(root + sub + '/' + file, 'r')
                if "weight" in file:
                    # binary
                    weight[str(cur_file["shape"][()])]= cur_file
#                     weight[sub].append(cur_file)
#                     weight[sub + str(cur_file["shape"][()])]= cur_file
                if "uniform" in file:
                    uniform[str(cur_file["shape"][()])]= cur_file
#                     uniform[sub].append(cur_file)
#                     uniform[sub + str(cur_file["shape"][()])] = cur_file
                if "target" in file:
                    target[str(cur_file["shape"][()])]= cur_file
#                     target[sub].append(cur_file)
#                     target[sub + str(cur_file["shape"][()])] = cur_file
        total[sub] = [target, uniform, weight]
    return total

def dice(y_true, y_pred, smooth=1.):
    y_true_f = np.array(y_true).flatten()
    y_pred_f = np.array(y_pred).flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    # tensorflow computation graph: will not configure print as one of the graph, unless using tf.Print()
    return (2.*intersection+smooth) / (np.sum(y_true_f)+np.sum(y_pred_f)+smooth)
# print(dice(merge_target, merge_result>0.02))
            
total = fetch_file()
table = {0: "target", 1: "uniform", 2: "weight"}

for i in total:
    print(i)
    for j in range(len(total[i])):
        print(table[j])
        for k in total[i][j]:
            print(total[i][j][k])



def calc_thres(output, target):
    it = np.arange(0, 1.01, 0.01)
    dice_thre = []
    for i in it:
        dice_thre.append(dice(target, output>i))
#     print(dice_thre)
    return dice_thre
#     plt.plot(it, dice_thre)
#     plt.savefig()
#     threshold = dice_thre[np.argmax(dice_thre)]
        
# first uniform:

path = os.getcwd() + '/model/h5df_data/recon/'

for i in total:
    # all, binary, normal
    # target: total[i][0]
    # uniform: total[i][1]
    # weight: total[i][2]
    print(i)
    for j in range(1, 3):
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
#         plt.close()
        
        for k in total[i][0]:
            nib.save(nib.Nifti1Image(np.array(total[i][j][k]["data"][:]>opt_thre).astype(int), np.eye(4)), path + i + '/' + os.path.basename(total[i][j][k].filename) + "_" + table[j] + "_" + str(opt_thre) + "_threshold.nii.gz")

