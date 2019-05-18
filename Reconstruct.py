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
from model.recon import *

print("Reconstruct:")

d = Data()
d.load_data(config["patch_size"])

# prepare data for training
train_num, valid_num = d.prekfold(config["patch_size"], config["patch_gap"], config["batch_size"], config["kfold"])


model = unet_model_3d(input_shape=config["input_shape"],
                              pool_size=config["pool_size"],
                              initial_learning_rate=config["initial_learning_rate"],
                              deconvolution=config["deconvolution"],
                              depth=config["depth"],
                              n_base_filters=config["n_base_filters"])

model.load_weights(os.getcwd() + '/model/weight/fold-0-weights-04-0.34.hdf5')

fold_index = 0

for i in d.valid_index:
    j = d.valid_index[i][fold_index]
    recons = Reconstruct(j, d.data[i][j][0].shape, config["patch_size"])
    print(i, d.patch_index[i][j].shape[0])
    for ind in range(d.patch_index[i][j].shape[0]):
        index = d.patch_index[i][j][ind]
        image_i = np.expand_dims(d.data[i][j][0][
                         index[0]:index[0]+d.patch_size[0], 
                         index[1]:index[1]+d.patch_size[1], 
                         index[2]:index[2]+d.patch_size[2]], axis=0)
        recons.add(model.predict([image_i[None, :]]), index)
    recons.store("dice_softmax_circle")

# model.load_weights(os.getcwd() + '/model/weight/fold-binary0-weights-06-0.02.hdf5')
#for i in d.valid_index:
#    j = d.valid_index[i][fold_index]
#    recons = Reconstruct(j, d.data[i][j][0].shape, config["patch_size"])
#    print(i, d.patch_index[i][j].shape[0])
#    for ind in range(d.patch_index[i][j].shape[0]):
#        index = d.patch_index[i][j][ind]
#        image_i = np.expand_dims(d.data[i][j][0][
#                         index[0]:index[0]+d.patch_size[0], 
#                         index[1]:index[1]+d.patch_size[1], 
#                         index[2]:index[2]+d.patch_size[2]], axis=0)
#        recons.add(model.predict([image_i[None, :]]), index)
#    recons.store("binary_softmax_circle")
