import keras
import numpy as np

# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
# https://www.tensorflow.org/api_docs/python/tf/keras/utils/Sequence
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, moda, pad_input, pad_target, pad_ind, kfold, batch_size, 
                 patch_size, patch_gap, valid_ind, is_train = False):
        self.moda = moda
        self.input = pad_input
        self.target = pad_target

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
        cur_image = []
        cur_target = []
        
        for i in self.patch_index:
            if i == "count" or i == "patch_size" or i == "patch_gap":
                continue
            unit = self.target[i].shape[0] // self.kfold
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
                                    # if (patch[3] == 1): # should use all images for validation
                                    patch = self.patch_index[i][j][k]
                                    img = []
                                    for m in self.moda:
                                        image = self.input[m][i][j]
                                        img.append(image[patch[0]:patch[0]+self.patch_size[0], 
                                                         patch[1]:patch[1]+self.patch_size[1], 
                                                         patch[2]:patch[2]+self.patch_size[2]])
                                    target = self.target[i][j]
                                    cur_image.append(img)
                                    cur_target.append(target[patch[0]:patch[0]+self.patch_size[0], 
                                                             patch[1]:patch[1]+self.patch_size[1], 
                                                             patch[2]:patch[2]+self.patch_size[2]])
                                    num -= 1
                                    if (num == 0):
                                        return np.array(cur_image), np.expand_dims(cur_target, axis=1)
#                                         return np.expand_dims(cur_image, axis=1), np.expand_dims(cur_target, axis=1)
                            else:
                                # append from first patch for current image
                                for k in range(self.patch_index[i][j].shape[0]):
                                    # if (patch[3] == 1): # should use all images for validation
                                    patch = self.patch_index[i][j][k]
                                    img = []
                                    for m in self.moda:
                                        image = self.input[m][i][j]
                                        img.append(image[patch[0]:patch[0]+self.patch_size[0], 
                                                         patch[1]:patch[1]+self.patch_size[1], 
                                                         patch[2]:patch[2]+self.patch_size[2]])
                                    target = self.target[i][j]
                                    cur_image.append(img)
                                    cur_target.append(target[patch[0]:patch[0]+self.patch_size[0], 
                                                             patch[1]:patch[1]+self.patch_size[1], 
                                                             patch[2]:patch[2]+self.patch_size[2]])
                                    num -= 1
                                    if (num == 0):
                                        return np.array(cur_image), np.expand_dims(cur_target, axis=1)
#                                         return np.expand_dims(cur_image, axis=1), np.expand_dims(cur_target, axis=1)

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
                                    if (patch[3] == 1):
                                        img = []
                                        for m in self.moda:
                                            image = self.input[m][i][j]
                                            img.append(image[patch[0]:patch[0]+self.patch_size[0], 
                                                             patch[1]:patch[1]+self.patch_size[1], 
                                                             patch[2]:patch[2]+self.patch_size[2]])
                                        target = self.target[i][j]
                                        cur_image.append(img)
                                        cur_target.append(target[patch[0]:patch[0]+self.patch_size[0], 
                                                                 patch[1]:patch[1]+self.patch_size[1], 
                                                                 patch[2]:patch[2]+self.patch_size[2]])
                                        num -= 1
                                        if (num == 0):
                                            return np.array(cur_image), np.expand_dims(cur_target, axis=1)
#                                             return np.expand_dims(cur_image, axis=1), np.expand_dims(cur_target, axis=1)
                            else:
                                # append from first patch for current image
                                for k in range(self.patch_index[i][j].shape[0]):
                                    patch = self.patch_index[i][j][k]
                                    if (patch[3] == 1):
                                        img = []
                                        for m in self.moda:
                                            image = self.input[m][i][j]
                                            img.append(image[patch[0]:patch[0]+self.patch_size[0], 
                                                             patch[1]:patch[1]+self.patch_size[1], 
                                                             patch[2]:patch[2]+self.patch_size[2]])
                                        target = self.target[i][j]
                                        cur_image.append(img)
                                        cur_target.append(target[patch[0]:patch[0]+self.patch_size[0], 
                                                                 patch[1]:patch[1]+self.patch_size[1], 
                                                                 patch[2]:patch[2]+self.patch_size[2]])
                                        num -= 1
                                        if (num == 0):
                                            return np.array(cur_image), np.expand_dims(cur_target, axis=1)
#                                             return np.expand_dims(cur_image, axis=1), np.expand_dims(cur_target, axis=1)
                    else:
                        continue
