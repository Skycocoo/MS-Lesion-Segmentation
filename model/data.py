import glob, os, random
import nibabel as nib
import numpy as np
import h5py

import matplotlib.pyplot as plt
from collections import defaultdict
from datetime import datetime

import ipywidgets as widgets
from ipywidgets import interact, interactive

from scipy import ndimage, misc

# directory: ./data/*/*.nii.gz
# there are different modalities that should be taken care of


class Data:
    def __init__(self):
        # self.data[image.shape][i][0]: image
        # self.data[image.shape][i][1]: segment
        
        ########## maybe shape doesnt matter => just store in a list? ##########
        self.data = defaultdict(list)
        self.kfold = None
        self.patch_index = None
        self.valid_index = {}
        random.seed(datetime.now())
        
          
    def zero_pad(self, image, div=(32, 32, 32)):
        pad_size = [0, 0, 0]
        pad = False
        for i in range(len(image.shape)):
            remain = image.shape[i] % div[i]
            if remain != 0:
                pad = True
                pad_size[i] = (image.shape[i] // div[i] + 1) * div[i] - image.shape[i]
        if pad:
            # deal with odd number of padding
            pad0 = (pad_size[0]//2, pad_size[0] - pad_size[0]//2)
            pad1 = (pad_size[1]//2, pad_size[1] - pad_size[1]//2)
            pad2 = (pad_size[2]//2, pad_size[2] - pad_size[2]//2)
            # https://stackoverflow.com/questions/50008587/zero-padding-a-3d-numpy-array
            return np.pad(image, (pad0, pad1, pad2), 'constant')
        else:
            return image

    def fetch_raw_data(self, raw):
        def fetch_file():
            model = []
            seg = []
            root, sub_dir, _ = next(os.walk(os.getcwd() + '/data/'))
            for sub in sub_dir:
                model.append(os.path.join(root, sub + '/FLAIR_preprocessed.nii.gz'))
                seg.append(os.path.join(root, sub + '/Consensus.nii.gz'))
            return model, seg
        
        model, seg = fetch_file()
        raw_data = defaultdict(list)
        for i in range(len(model)):
            image = nib.load(model[i])
            segment = nib.load(seg[i])
            raw_data[image.shape].append([image.get_fdata(), segment.get_fdata()])
        with h5py.File(raw, 'w') as f:
            for i in raw_data:
                f.create_dataset(str(i), data=raw_data[i])
        return self.load_raw_data(raw)
    
    def load_raw_data(self, raw):
        raw_file = h5py.File(raw, 'r') # should not close it immediately
        raw_data = defaultdict(list)
        for i in raw_file.keys():
            # to get the matrix: self.data[i][:]
            # d.data[i][j][0], d.data[i][j][1]
            raw_data[i] = raw_file[i]
        return raw_data, raw_file
    
    def pad_raw_data(self, patch_size, pad, raw):
        raw_data = None
        raw_file = None
        if os.path.isfile(raw):
            raw_data, raw_file = self.load_raw_data(raw)
        else:
            raw_data, raw_file = self.fetch_raw_data(raw)
        pad_data = defaultdict(list)
        for i in raw_data:
            for j in range(raw_data[i].shape[0]):
                img = self.zero_pad(raw_data[i][j][0], patch_size)
                tar = self.zero_pad(raw_data[i][j][1], patch_size)
                pad_data[img.shape].append([img, tar])
        raw_file.close()
        with h5py.File(pad, 'w') as f:
            f.create_dataset("patch_size", data=patch_size)    
            for i in pad_data:
                f.create_dataset(str(i), data=pad_data[i])
        pad_file = h5py.File(pad, 'r')
        for i in pad_file.keys():
            self.data[i] = pad_file[i]
    
    def load_data(self, patch_size=(32, 32, 32), 
                  pad="./model/h5df_data/pad_data.h5", raw="./model/h5df_data/raw_data.h5"):
        # self.data[image.shape][i][0]: image
        # self.data[image.shape][i][1]: segment
        if os.path.isfile(pad):
            pad_file = h5py.File(pad, 'r')
            if np.all(pad_file["patch_size"][:] == list(patch_size)):
                for i in pad_file.keys():
                    self.data[i] = pad_file[i]
            else:
                pad_file.close()
                self.pad_raw_data(patch_size, pad, raw)
        else:
            self.pad_raw_data(patch_size, pad, raw)
    
    def show_image(self, images):
        # show image with [None, None, : ,: ,:] dimension
        def show_frame(id):
            length = len(images)
            for i in range(length):
                plt.subplot(1, length, i+1)
                plt.imshow(images[i][0, 0, id, :, :], cmap='gray')
        interact(show_frame, 
                 id=widgets.IntSlider(min=0, max=images[0].shape[2]-1, step=1, value=images[0].shape[2]/2))
        
        
        
    def gen_patch_index(self, patch_size, patch_gap):
        count = 0
        self.patch_index = defaultdict(list)
        for i in self.data:
            if i == "patch_size":
                continue
            shape = self.data[i][0][0].shape
            patch_ind = []
            patch_num = [int((shape[i]-patch_size[i]) / patch_gap) for i in range(len(shape))]

            # assume this is a 3d image
            for a in range(patch_num[0]):
                for b in range(patch_num[1]):
                    for c in range(patch_num[2]):
                        patch_ind.append([a * patch_gap, b * patch_gap, c * patch_gap])
            self.patch_index[i] = patch_ind
            # total number of patches for this shape
            count += len(patch_ind) * self.data[i].shape[0]
            print("count: ", count)
        print("count: ", count)
        return count
#         for i in self.patch_index:
#             print(np.array(self.patch_index[i]).shape)

    def prekfold(self, patch_size, patch_gap, batch_size, kfold=5):
        if (self.kfold == kfold):
            return
        self.kfold = kfold

        # initialize validation index for training
        # K-fold LOOCV: leave one out cross validation
        for i in self.data:
            if i == "patch_size":
                continue
            self.valid_index[i] = np.random.choice(range(self.kfold), self.kfold)

        num = self.gen_patch_index(patch_size, patch_gap)
        
        ###### need to recalculate training and validation number ######
        
        train_num = num // self.kfold * (self.kfold - 1)
        valid_num = num - train_num
        
        print(self.valid_index)

        return train_num // batch_size, valid_num
    
#     def prepatch(self, patch_size=(32, 32, 32), gap=10):
#         for i in self.data:
            
    # batch_size: 2 or 4
    def train_generator(self, fold_index, batch_size=2):
#         while True:
#             print("new epoch")
            for i in self.data:
#                 print(i)
                input = [] # input
                output = [] # target
                unit = len(self.data[i]) // self.kfold
                for j in range(len(self.data[i])):
                    # skip validation data
                    if j >= self.valid_index[i][fold_index] * unit and j < (self.valid_index[i][fold_index]+1) * unit:
#                         print("skipping {0}".format(j))
                        continue
                    if len(input) < batch_size:
                        input.append(self.data[i][j][0])
                        output.append(self.data[i][j][1])
#                         input.append(np.expand_dims(self.data[i][j][0], axis=0))
#                         output.append(np.expand_dims(self.data[i][j][1], axis=0))

#                         shape = self.data[i][j][0].shape
# #                         x = shape[0] // 2
# #                         y = shape[1] // 2
# #                         z = shape[2] // 2
# #                         input.append(np.expand_dims(self.data[i][j][0][x : x + patch_size, y : y + patch_size, z : z + patch_size], axis=0))
# #                         output.append(np.expand_dims(self.data[i][j][1][x : x + patch_size, y : y + patch_size, z : z + patch_size], axis=0))
#                         input.append(np.expand_dims(ndimage.zoom(self.data[i][j][0], (32/shape[0], 32/shape[1], 32/shape[2])), axis=0))
#                         output.append(np.expand_dims(ndimage.zoom(self.data[i][j][1], (32/shape[0], 32/shape[1], 32/shape[2])), axis=0))
            
                    else:
                        # print(np.array(input).shape, np.array(output).shape)
                        yield np.array(input), np.array(output)
                        # reinitialize input and output
                        input = [self.data[i][j][0]]
                        output = [self.data[i][j][1]]
#                         input.append(np.expand_dims(self.data[i][j][0], axis=0))
#                         output.append(np.expand_dims(self.data[i][j][1], axis=0))
                                      
#                         shape = self.data[i][j][0].shape
# #                         x = shape[0] // 2
# #                         y = shape[1] // 2
# #                         z = shape[2] // 2
# #                         input.append(np.expand_dims(self.data[i][j][0][x : x + patch_size, y : y + patch_size, z : z + patch_size], axis=0))
# #                         output.append(np.expand_dims(self.data[i][j][1][x : x + patch_size, y : y + patch_size, z : z + patch_size], axis=0))
#                         input.append(np.expand_dims(ndimage.zoom(self.data[i][j][0], (32/shape[0], 32/shape[1], 32/shape[2])), axis=0))
#                         output.append(np.expand_dims(ndimage.zoom(self.data[i][j][1], (32/shape[0], 32/shape[1], 32/shape[2])), axis=0))
                if (len(input) == batch_size): 
                    yield np.array(input), np.array(output)
            
                    

    # each scanner yield a simple validation sample
    def valid_generator(self, fold_index):
#         while True:
            for i in self.valid_index:
                input = [self.data[i][fold_index][0]]
                output = [self.data[i][fold_index][1]]
#                 valid = self.data[i][fold_index]
#                 input.append(np.expand_dims(valid[0], axis=0))
#                 output.append(np.expand_dims(valid[1], axis=0))

#                 shape = valid[0].shape
# #                 x = shape[0] // 2
# #                 y = shape[1] // 2
# #                 z = shape[2] // 2
# #                 input.append(np.expand_dims(valid[0][x : x + patch_size, y : y + patch_size, z : z + patch_size], axis=0))
# #                 output.append(np.expand_dims(valid[1][x : x + patch_size, y : y + patch_size, z : z + patch_size], axis=0))
#                 input.append(np.expand_dims(ndimage.zoom(valid[0], (32/shape[0], 32/shape[1], 32/shape[2])), axis=0))
#                 output.append(np.expand_dims(ndimage.zoom(valid[1], (32/shape[0], 32/shape[1], 32/shape[2])), axis=0))
                yield np.array(input), np.array(output)
            
#             unit = len(self.data[i]) // self.kfold
#             for j in range(self.valid_index[i][fold_index] * unit, self.valid_index[i][fold_index] * (unit+1)):
#                 valid = self.data[i][j]
#                 yield valid[0], valid[1]
