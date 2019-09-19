import os

os.chdir("/scratch/yl4217/MS-Lesion-Segmentation/")

config = {}
config["weights_file"] = os.getcwd() + '/model/weight'
config["patch_size"] = (64, 64, 64)  # switch to None to train on the whole image
config["patch_gap"] = 16
config["batch_size"] = 2
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
config["n_epochs"] = 5

from model.data import *
from model.model import *

                    
import keras
import numpy as np

# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
# https://www.tensorflow.org/api_docs/python/tf/keras/utils/Sequence
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, pad_data, pad_ind, kfold, batch_size, 
                 patch_size, patch_gap, valid_ind, is_train = False):
        self.data = pad_data
        self.patch_index = pad_ind
        self.kfold = kfold
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.patch_gap = patch_gap
        self.valid_index = valid_ind
        self.isTrain = is_train
        self.fold_index = None
        self.len = None
        
    def set_index(self, index):
        self.fold_index = index
        
    def __len__(self):
        if self.len == None:
            # count the total number of patches
            count = 0
            for i in self.patch_index:
                if i == "count" or i == "patch_size" or i == "patch_gap":
                    continue
                unit = self.patch_index[i].shape[0] // self.kfold
                iter_start = self.valid_index[i][self.fold_index] * unit
                iter_end = (self.valid_index[i][self.fold_index]+1) * unit
                for j in range(self.patch_index[i].shape[0]):
                    if j >= iter_start and j < iter_end:
                        if self.isTrain:
                            continue
                        else:
                            count += self.patch_index[i][j].shape[0]
                    else:
                        if self.isTrain:
                            count += self.patch_index[i][j].shape[0]
                        else:
                            continue
            self.len = count // self.batch_size
        return self.len
    
            
    def __getitem__(self, batch_index):
        start = batch_index * self.batch_size
        num = self.batch_size
        first_iter = True
        img = []
        tar = []
        
        for i in self.patch_index:
            if i == "count" or i == "patch_size" or i == "patch_gap":
                continue
            unit = self.data[i].shape[0] // self.kfold
            iter_start = self.valid_index[i][self.fold_index] * unit
            iter_end = (self.valid_index[i][self.fold_index]+1) * unit
            for j in range(self.patch_index[i].shape[0]):
                # validation dataset: should validate on all patches
                if j >= iter_start and j < iter_end:
                    if self.isTrain:
                        continue
                    else:
                        if start > self.patch_index[i][j].shape[0]:
                            start -= self.patch_index[i][j].shape[0]
                        else:
                            # generate
                            if first_iter:
                                # first iter: append from start
                                start_iter = start
                                first_iter = False
                                for k in range(start_iter, self.patch_index[i][j].shape[0]):
                                    patch = self.patch_index[i][j][k]
#                                     if (patch[3] == 1):
                                    image = self.data[i][j][0]
                                    target = self.data[i][j][1]
                                    img.append(image[patch[0]:patch[0]+self.patch_size[0], 
                                                     patch[1]:patch[1]+self.patch_size[1], 
                                                     patch[2]:patch[2]+self.patch_size[2]])
                                    tar.append(target[patch[0]:patch[0]+self.patch_size[0], 
                                                     patch[1]:patch[1]+self.patch_size[1], 
                                                     patch[2]:patch[2]+self.patch_size[2]])
                                    num -= 1
                                    if (num == 0):
                                        return np.expand_dims(img, axis=1), np.expand_dims(tar, axis=1)
                            else:
                                # append from first patch for current image
                                for k in range(self.patch_index[i][j].shape[0]):
                                    patch = self.patch_index[i][j][k]
#                                     if (patch[3] == 1):
                                    image = self.data[i][j][0]
                                    target = self.data[i][j][1]
                                    img.append(image[patch[0]:patch[0]+self.patch_size[0], 
                                                     patch[1]:patch[1]+self.patch_size[1], 
                                                     patch[2]:patch[2]+self.patch_size[2]])
                                    tar.append(target[patch[0]:patch[0]+self.patch_size[0], 
                                                     patch[1]:patch[1]+self.patch_size[1], 
                                                     patch[2]:patch[2]+self.patch_size[2]])
                                    num -= 1
                                    if (num == 0):
                                        return np.expand_dims(img, axis=1), np.expand_dims(tar, axis=1)

                else:
                    # training dataset
                    if self.isTrain:
                        if start > self.patch_index[i][j].shape[0]:
                            start -= self.patch_index[i][j].shape[0]
                        else:
                            # generate
                            if first_iter:
                                # first iter: append from start
                                start_iter = start
                                first_iter = False
                                # feed some number of patches that does not have lesion
                                for k in range(start_iter, self.patch_index[i][j].shape[0]):
                                    patch = self.patch_index[i][j][k]
#                                     if (patch[3] == 1):
                                    image = self.data[i][j][0]
                                    target = self.data[i][j][1]
                                    img.append(image[patch[0]:patch[0]+self.patch_size[0], 
                                                     patch[1]:patch[1]+self.patch_size[1], 
                                                     patch[2]:patch[2]+self.patch_size[2]])
                                    tar.append(target[patch[0]:patch[0]+self.patch_size[0], 
                                                     patch[1]:patch[1]+self.patch_size[1], 
                                                     patch[2]:patch[2]+self.patch_size[2]])
                                    num -= 1
                                    if (num == 0):
                                        return np.expand_dims(img, axis=1), np.expand_dims(tar, axis=1)
                            else:
                                # append from first patch for current image
                                for k in range(self.patch_index[i][j].shape[0]):
                                    patch = self.patch_index[i][j][k]
#                                     if (patch[3] == 1):
                                    image = self.data[i][j][0]
                                    target = self.data[i][j][1]
                                    img.append(image[patch[0]:patch[0]+self.patch_size[0], 
                                                     patch[1]:patch[1]+self.patch_size[1], 
                                                     patch[2]:patch[2]+self.patch_size[2]])
                                    tar.append(target[patch[0]:patch[0]+self.patch_size[0], 
                                                     patch[1]:patch[1]+self.patch_size[1], 
                                                     patch[2]:patch[2]+self.patch_size[2]])
                                    num -= 1
                                    if (num == 0):
                                        return np.expand_dims(img, axis=1), np.expand_dims(tar, axis=1)
                    else:
                        continue





d = Data()
d.load_data(config["patch_size"])

# prepare data for training
train_num, valid_num = d.prekfold(config["patch_size"], config["patch_gap"], config["batch_size"], config["kfold"])
train_generator = DataGenerator(d.data, d.patch_index, d.kfold, d.batch_size, 
                                d.patch_size, d.patch_gap, d.valid_index, True)
valid_generator = DataGenerator(d.data, d.patch_index, d.kfold, d.batch_size, 
                                d.patch_size, d.patch_gap, d.valid_index, False)

result = []
target = []
image = []

print("training all patch; dice score loss")

def train(config, data, train_generator, valid_generator, train_num, valid_num):
#     models = []
    print(train_num, valid_num)
    for i in range(data.kfold):
        print ('-'*100)
        print ("Fold:", i)
        
        train_generator.set_index(i)
        valid_generator.set_index(i)
        
        model = unet_model_3d(input_shape=config["input_shape"],
                              pool_size=config["pool_size"],
                              initial_learning_rate=config["initial_learning_rate"],
                              deconvolution=config["deconvolution"],
                              depth=config["depth"],
                              n_base_filters=config["n_base_filters"])
        
        # model.load_weights(os.getcwd() + '/model/weight/weights-01-0.02-0428-binary-patch.hdf5')
        
        # print(model.summary())
        
        callbacks = get_callbacks(config["weights_file"], str(i)+'_all_patch_',
                                initial_learning_rate=config["initial_learning_rate"],
                                learning_rate_drop=config["learning_rate_drop"],
                                learning_rate_patience=config["patience"],
                                early_stopping_patience=config["early_stop"])

        model.fit_generator(generator=train_generator,
                            steps_per_epoch=train_num,
                            epochs=config["n_epochs"],
                            validation_data=valid_generator,
                            validation_steps=valid_num,
                            callbacks=callbacks,
                            workers=2,
                            verbose=1)
        break
train(config, d, train_generator, valid_generator, train_num, valid_num)
