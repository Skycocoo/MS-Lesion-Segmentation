import glob, os, random
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from datetime import datetime

import ipywidgets as widgets
from ipywidgets import interact, interactive

from scipy import ndimage, misc

# directory: ./data/*/*.nii.gz
# there are different modalities that should be taken care of


class Data:
    def __init__(self):
        self.model = []
        self.seg = []
        # self.data[image.shape][i][0]: image
        # self.data[image.shape][i][1]: segment
        self.data = defaultdict(list)
        self.kfold = None
        self.valid_index = {}
        random.seed(datetime.now())

    def fetch_file(self):
        root, sub_dir, _ = next(os.walk(os.getcwd() + '/data/'))
        for sub in sub_dir:
            self.model.append(os.path.join(root, sub + '/FLAIR_preprocessed.nii.gz'))
            self.seg.append(os.path.join(root, sub + '/Consensus.nii.gz'))

    def load_data(self):
        self.fetch_file()
        for i in range(len(self.model)):
            image = nib.load(self.model[i])
            segment = nib.load(self.seg[i])
            # self.data[image.shape][i][0]: image
            # self.data[image.shape][i][1]: segment
            self.data[image.shape].append([image.get_fdata(),
                                           segment.get_fdata()])

    def show_image(images):
        # show image with [None, None, : ,: ,:] dimension
        def show_frame(id):
            length = len(images)
            for i in range(length):
                plt.subplot(1, length, i+1)
                plt.imshow(images[i][0, 0, id, :, :], cmap='gray')
        interact(show_frame, id=widgets.IntSlider(min=0, max=images[0].shape[2]-1, step=1, value=images[0].shape[2]/2))

    # need to zeropad image: shape divisible by pool ^ depth
    def zero_pad(self, image, pool_size=(2, 2, 2), depth=4):
        pad_size = [0, 0, 0]
        pad = False
        for i in range(len(image.shape)):
            divident = pool_size[i] ** depth
            remain = image.shape[i] % divident
            if remain != 0:
                pad = True
                div = image.shape[i] // divident
                pad_size[i] = (div+1) * divident - image.shape[i]
        if pad:
            # deal with odd number of padding
            pad0 = (pad_size[0]//2, pad_size[0] - pad_size[0]//2)
            pad1 = (pad_size[1]//2, pad_size[1] - pad_size[1]//2)
            pad2 = (pad_size[2]//2, pad_size[2] - pad_size[2]//2)
            # https://stackoverflow.com/questions/50008587/zero-padding-a-3d-numpy-array
            image = np.pad(image, (pad0, pad1, pad2), 'constant')

    def data_num(self):
        num = 0
        for i in self.data:
            num += len(self.data[i])
        return num

    def prekfold(self, kfold=5, batch_size=2):
        if (self.kfold == kfold):
            return
        
        self.kfold = kfold

        # initialize validation index for training
        # K-fold LOOCV: leave one out cross validation
        for i in self.data:
            self.valid_index[i] = random.sample(range(self.kfold), self.kfold)

        num = self.data_num()
        train_num = num // self.kfold * (self.kfold - 1)
        valid_num = num - train_num

        return train_num // batch_size, valid_num
    
    def preprocess(self, batch_size=2, pool_size=(2, 2, 2), depth=4):
        # # pad data to be divisible by pool_size ^ depth
        # for i in self.data:
        #     for j in range(len(self.data[i])):
        #         self.zero_pad(self.data[i][j][0], pool_size, depth)
        #         self.zero_pad(self.data[i][j][1], pool_size, depth)
        #         # print(d.data[i][j][0].shape, d.data[i][j][1].shape)
        
        # # resize the image to be shape (32, 32, 32) to test the model
        for i in self.data:
            for j in range(len(self.data[i])):
                for k in range(2):
                    shape = self.data[i][j][k].shape
                    self.data[i][j][k] = np.expand_dims(ndimage.zoom(self.data[i][j][k], 
                                                                     (32/shape[0], 32/shape[1], 32/shape[2])), axis=0)

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
