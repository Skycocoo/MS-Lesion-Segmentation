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
        ########## maybe shape doesnt matter => just store in a list? ##########
        self.data = defaultdict(list)
        # self.data = [] 
        # https://stackoverflow.com/questions/37214482/saving-with-h5py-arrays-of-different-sizes
        # cannot use h5df to store array with different sizes
        self.kfold = None
        self.patch_size = None
        self.patch_index = defaultdict(list)
        self.valid_index = {}
        random.seed(datetime.now())
        
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
        # raw_data = []
        # raw_data[i][0]: image, raw_data[i][1]: target
        for i in range(len(model)):
            image = nib.load(model[i])
            segment = nib.load(seg[i])
            raw_data[image.shape].append([image.get_fdata(), segment.get_fdata()])
            # raw_data.append([image.get_fdata(), segment.get_fdata()])
        with h5py.File(raw, 'w') as f:
            # f.create_dataset("raw_data", data=raw_data)
            for i in raw_data:
                f.create_dataset(str(i), data=raw_data[i])
        return self.load_raw_data(raw)
    
    def load_raw_data(self, raw):
        raw_file = h5py.File(raw, 'r') # should not close it immediately
        # raw_data = raw_file["raw_data"]
        raw_data = defaultdict(list)
        for i in raw_file.keys():
            # to get the matrix: self.data[i][:]
            # d.data[i][j][0], d.data[i][j][1]
            raw_data[i] = raw_file[i]
        return raw_data, raw_file
    
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

        
    def pad_raw_data(self, patch_size, pad, raw):
        raw_data = None
        raw_file = None
        if os.path.isfile(raw):
            raw_data, raw_file = self.load_raw_data(raw)
        else:
            raw_data, raw_file = self.fetch_raw_data(raw)
        
        # pad_data = []
        pad_data = defaultdict(list)
        for i in raw_data:
            for j in range(raw_data[i].shape[0]):
                img = self.zero_pad(raw_data[i][j][0], patch_size)
                tar = self.zero_pad(raw_data[i][j][1], patch_size)
                pad_data[img.shape].append([img, tar])
                # pad_data.append([img, tar])
        raw_file.close()
        with h5py.File(pad, 'w') as f:
            f.create_dataset("patch_size", data=patch_size)
            # f.create_dataset("pad_data", data=pad_data)
            for i in pad_data:
                f.create_dataset(str(i), data=pad_data[i])

        pad_file = h5py.File(pad, 'r')
        # self.data = pad_file["pad_data"]
        for i in pad_file.keys():
            if i == "patch_size":
                continue
            self.data[i] = pad_file[i]
    
    def load_data(self, patch_size=(32, 32, 32), 
                  pad="./model/h5df_data/pad_data.h5", raw="./model/h5df_data/raw_data.h5"):
        # self.data[image.shape][i][0]: image
        # self.data[image.shape][i][1]: segment
        if os.path.isfile(pad):
            pad_file = h5py.File(pad, 'r')
            if np.all(pad_file["patch_size"][:] == list(patch_size)):
                # self.data = pad_file["pad_data"]
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
        
        
        
    def gen_patch_index(self, patch_size, patch_gap, pat):
        count = 0
        patch_index = defaultdict(list)
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
            # self.patch_index[i] = patch_ind
            patch_index[i] = [np.copy(patch_ind) for _ in range(self.data[i].shape[0])]
            for c in range(len(patch_index[i])):
                # in-place shuffle
                np.random.shuffle(patch_index[i][c])
            # total number of patches for this shape
            count += len(patch_ind) * self.data[i].shape[0]
        
        with h5py.File(pat, 'w') as f:
            f.create_dataset("count", data=count)
            f.create_dataset("patch_size", data=patch_size)
            f.create_dataset("patch_gap", data=patch_gap)
            for i in patch_index:
                f.create_dataset(str(i), data=patch_index[i])
        
        pat_ind = h5py.File(pat, 'r')
        for i in pat_ind.keys():
            if i == "count" or i == "patch_size" or i == "patch_gap":
                continue
            self.patch_index[i] = pat_ind[i]
        # return the total number of patches
        return pat_ind["count"][()]

    def load_patch_index(self, patch_size, patch_gap, pat):
        if os.path.isfile(pat):
            pat_ind = h5py.File(pat, 'r')
            # print(list(pat_ind.keys()))
            if (np.all(pat_ind["patch_size"][:] == list(patch_size))) and (pat_ind["patch_gap"][()] == patch_gap):
                for i in pat_ind.keys():
                    self.patch_index[i] = pat_ind[i]
                return pat_ind["count"][()]
            else:
                pat_ind.close()
                return self.gen_patch_index(patch_size, patch_gap, pat)
        else:
            return self.gen_patch_index(patch_size, patch_gap, pat)
            
    
    def prekfold(self, patch_size, patch_gap, batch_size, pat='./model/h5df_data/pat_ind.h5', kfold=5):
        self.kfold = kfold
        self.patch_size = patch_size

        # initialize validation index for training
        # K-fold LOOCV: leave one out cross validation
        for i in self.data:
            if i == "patch_size":
                continue
            self.valid_index[i] = random.sample(range(self.kfold), self.kfold)

        num = self.load_patch_index(patch_size, patch_gap, pat)
        train_num = num // self.kfold * (self.kfold - 1)
        valid_num = num - train_num
        
        return train_num // batch_size, valid_num
    

    # batch_size: 2 or 4
    def train_generator(self, fold_index, batch_size=10):
        img = []
        tar = []
        for i in self.data:
            num_img = self.data[i].shape[0]
            unit = num_img // self.kfold
            # self.patch_index[i].shape: # img, # patches, 3d index
            for ind in range(self.patch_index[i].shape[1]):
                # self.data[i].shape: # img, img/tar, (3d image)
                for j in range(num_img):
                    # skip validation data
                    if j >= self.valid_index[i][fold_index] * unit and j < (self.valid_index[i][fold_index]+1) * unit:
                        continue
                    if len(img) == batch_size:
                        yield np.array(img), np.array(tar)
                        img = []
                        tar = []
                    patch = self.patch_index[i][j][ind]
                    image = self.data[i][j][0]
                    target = self.data[i][j][1]
                    img.append(image[patch[0]:patch[0]+self.patch_size[0], 
                                     patch[1]:patch[1]+self.patch_size[1], 
                                     patch[2]:patch[2]+self.patch_size[2]])
                    tar.append(target[patch[0]:patch[0]+self.patch_size[0], 
                                     patch[1]:patch[1]+self.patch_size[1], 
                                     patch[2]:patch[2]+self.patch_size[2]])
        if len(img) == batch_size:
            yield np.array(img), np.array(tar)

            
    # each scanner yield a simple validation sample
    def valid_generator(self, fold_index):
        img = []
        tar = []
        for i in self.valid_index:
            unit = self.data[i].shape[0] // self.kfold
            for j in range(self.valid_index[i][fold_index] * unit, (self.valid_index[i][fold_index]+1) * unit):
                # self.data[i][j][0]: training image
                for ind in range(self.patch_index[i].shape[1]):
                    patch = self.patch_index[i][j][ind]
                    image = self.data[i][j][0]
                    target = self.data[i][j][1]
                    img.append(image[patch[0]:patch[0]+self.patch_size[0], 
                                     patch[1]:patch[1]+self.patch_size[1], 
                                     patch[2]:patch[2]+self.patch_size[2]])
                    tar.append(target[patch[0]:patch[0]+self.patch_size[0], 
                                     patch[1]:patch[1]+self.patch_size[1], 
                                     patch[2]:patch[2]+self.patch_size[2]])
                    yield np.array(img), np.array(tar)
                    img = []
                    tar = []
