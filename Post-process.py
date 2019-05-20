import os

os.chdir("/scratch/yl4217/MS-Lesion-Segmentation/")

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

from model.data import *
from model.model import *

d = Data()
d.load_data(config["patch_size"])

# prepare data for training
train_num, valid_num = d.prekfold(config["patch_size"], config["patch_gap"], config["batch_size"], config["kfold"])

from model.recon import *

weight_path = ['/model/weight/fold0_all_patch_weights-05-0.40.hdf5',
               '/model/weight/fold0_weights-03-0.39.hdf5',
               '/model/weight/fold_binary0weights-05-0.06.hdf5',
              ]
weight_name = ['all',
               'normal',
               'binary',
              ]

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
        recons = Reconstruct(j, d.data[i][j][0].shape, config["patch_size"], True)
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
            recons.add(result, index)
            normal.add(result, index)
            image.add(image_i, index)
            target.add(target_i, index)
#             break
        dir_name = './model/h5df_data/recon/' + weight_name[i_weight] + '/'
        os.makedirs(os.path.dirname(dir_name), exist_ok=True)
        file_name = '/recon/' + weight_name[i_weight] + '/'+ str(d.data[i][j][0].shape)
        recons.store(file_name + '_weighted_output')
        normal.store(file_name + "_uniform_output")
        image.store(file_name + "_input")
        target.store(file_name + "_target")
#         break
#         recon.append(recons)
#         normal.append(orig)


import os
import h5py
from collections import defaultdict

def fetch_file():
    path = os.getcwd() + '/model/h5df_data/recon/'
    root, sub_dir, _ = next(os.walk(path))
    uniform = defaultdict(list)
    weight = defaultdict(list)
    target = defaultdict(list)
    for sub in sub_dir:
        _, _, sub_files = next(os.walk(root + sub))
        for file in sub_files:
            if "nii" not in file:
                cur_file = h5py.File(root + sub + '/' + file, 'r')
                if "weight" in file:
                    # binary
                    weight[sub].append(cur_file)
#                     weight[sub + str(cur_file["shape"][()])]= cur_file
                if "uniform" in file:
                    uniform[sub].append(cur_file)
#                     uniform[sub + str(cur_file["shape"][()])] = cur_file
                if "target" in file:
                    target[sub].append(cur_file)
#                     target[sub + str(cur_file["shape"][()])] = cur_file
    return weight, uniform, target

def dice(y_true, y_pred, smooth=1.):
    y_true_f = np.array(y_true).flatten()
    y_pred_f = np.array(y_pred).flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    # tensorflow computation graph: will not configure print as one of the graph, unless using tf.Print()
    return (2.*intersection+smooth) / (np.sum(y_true_f)+np.sum(y_pred_f)+smooth)
# print(dice(merge_target, merge_result>0.02))
            
weight, uniform, target = fetch_file()
# print(weight)

def calc_thres(output, target):
    it = np.arange(0, 1.01, 0.01)
    dice_thre = []
    for i in it:
        dice_thre.append(dice(target, output>i))
    print(dice_thre)
    return dice_thre
#     plt.plot(it, dice_thre)
#     plt.savefig()
#     threshold = dice_thre[np.argmax(dice_thre)]
        
# first uniform:

path = os.getcwd() + '/model/h5df_data/recon/'

for i in target:
    # i: binary
    threshold = np.zeros(101)
    for j in range(len(target[i])):
        # target[i][j]: binary, (shape), --
        threshold += calc_thres(weight[i][j]["data"][:], target[i][j]["data"][:])
    threshold /= len(target[i])
    plt.plot(np.arange(0, 1.01, 0.01), threshold)
    plt.savefig(path + i + '/weight_threashold.png')
    
    opt = threshold[np.argmax(threshold)]
    for j in range(len(target[i])):
        nib.save(nib.Nifti1Image(np.array(weight[i][j]["data"][:]>opt).astype(int), np.eye(4)), 
                 path + i + '/' + os.path.basename(target[i][j].filename) + "_weight_threashold.nii.gz")
for i in target:
    # i: binary
    threshold = np.zeros(101)
    for j in range(len(target[i])):
        # target[i][j]: binary, (shape), --
        threshold += calc_thres(uniform[i][j]["data"][:], target[i][j]["data"][:])
    threshold /= len(target[i])
    plt.plot(np.arange(0, 1.01, 0.01), threshold)
    plt.savefig(path + i + '/uniform_threashold.png')
    
    opt = threshold[np.argmax(threshold)]
    for j in range(len(target[i])):
        nib.save(nib.Nifti1Image(np.array(uniform[i][j]["data"][:]>opt).astype(int), np.eye(4)), 
                 path + i + '/' + os.path.basename(target[i][j].filename) + "_uniform_threashold.nii.gz")
    