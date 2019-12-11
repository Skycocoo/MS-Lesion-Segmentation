
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
    def __init__(self, moda):
        self.moda = moda
        # ["FLAIR_preprocessed", "T1_preprocessed", "T2_preprocessed"]
        self.input = {m : defaultdict(list) for m in self.moda}
        self.target = defaultdict(list)
        self.kfold = None
        self.batch_size = None
        self.patch_size = None
        self.patch_gap = None
        self.patch_index = defaultdict(list)
        self.valid_index = {}
        # pre-set seed so that reconstruct can retrieve valid_index for final result
        random.seed(3000)
        
    def fetch_raw_data_from_file(self, modality, file_name):
        def fetch_file(modality):
            # could be "Consensus", "FLAIR_preprocessed"
            data = []
            root, sub_dir, _ = next(os.walk(os.getcwd() + '/data/'))
            for sub in sub_dir:
                data.append(os.path.join(root, sub + '/' + modality + '.nii.gz'))
            return data
        file = fetch_file(modality)
        raw_data = defaultdict(list)
        for i in range(len(file)):
            image = nib.load(file[i])
            raw_data[image.shape].append(image.get_fdata())
        with h5py.File(file_name, 'w') as f:
            for i in raw_data:
                f.create_dataset(str(i), data=raw_data[i])
        return raw_data
    
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
    
    def check_file_exist(self, path, pattern):
        for m in self.moda:
            if not os.path.isfile(path + pattern + m + '.h5'):
                return False
        return True
    
    
    def fetch_target_data(self, target_path):
        self.fetch_raw_data_from_file('Consensus', target_path + 'target_data.h5')
        return self.load_target_data(target_path)
    
    def fetch_raw_data(self, raw_path):
        for m in self.moda:
            # different modality: same order of images
            self.fetch_raw_data_from_file(m, raw_path + 'raw_data_' + m + '.h5')
        return self.load_raw_data(raw_path)
    
    def load_target_data(self, target_path):
        target_file = h5py.File(target_path + 'target_data.h5', 'r')
        target_data = defaultdict(list)
        for i in target_file.keys():
            target_data[i] = target_file[i]
        return target_data, target_file
    
    def load_raw_data(self, raw_path):
        raw_file = []
        raw_data = {m : defaultdict(list) for m in self.moda}
        for m in self.moda:
            data = defaultdict(list)
            file = h5py.File(raw_path + 'raw_data_' + m + '.h5', 'r') 
            raw_file.append(file)
            for i in file.keys():
                # to get the matrix: self.data[i][:]
                # d.data[i][j][0], d.data[i][j][1]
                data[i] = file[i]
            raw_data[m] = data
        return raw_data, raw_file
    
        # raw_data:
        # {'FLAIR_preprocessed': defaultdict(list,
        #               {'(128, 224, 256)': <HDF5 dataset "(128, 224, 256)": shape (5, 128, 224, 256), type "<f8">,
        #                '(144, 512, 512)': <HDF5 dataset "(144, 512, 512)": shape (5, 144, 512, 512), type "<f8">,
        #                '(261, 336, 336)': <HDF5 dataset "(261, 336, 336)": shape (5, 261, 336, 336), type "<f8">}),
        #   'T1_preprocessed': defaultdict(list,
        #               {'(128, 224, 256)': <HDF5 dataset "(128, 224, 256)": shape (5, 128, 224, 256), type "<f8">,
        #                '(144, 512, 512)': <HDF5 dataset "(144, 512, 512)": shape (5, 144, 512, 512), type "<f8">,
        #                '(261, 336, 336)': <HDF5 dataset "(261, 336, 336)": shape (5, 261, 336, 336), type "<f8">}),
        #   'T2_preprocessed': defaultdict(list,
        #               {'(128, 224, 256)': <HDF5 dataset "(128, 224, 256)": shape (5, 128, 224, 256), type "<f8">,
        #                '(144, 512, 512)': <HDF5 dataset "(144, 512, 512)": shape (5, 144, 512, 512), type "<f8">,
        #                '(261, 336, 336)': <HDF5 dataset "(261, 336, 336)": shape (5, 261, 336, 336), type "<f8">})}
        # raw_file:
        #  [<HDF5 file "raw_data_FLAIR_preprocessed.h5" (mode r)>,
        #   <HDF5 file "raw_data_T1_preprocessed.h5" (mode r)>,
        #   <HDF5 file "raw_data_T2_preprocessed.h5" (mode r)>]

    def pad_target_data(self, patch_size, pad_path, target_path):
        # input
        target_data = None
        target_file = None
        if os.path.isfile(target_path + "target_data.h5"):
            target_data, target_file = self.load_target_data(target_path)
        else:
            target_data, target_file = self.fetch_target_data(target_path)
        
        pad_target_data = defaultdict(list)
        for i in target_data.keys():
            for j in range(target_data[i].shape[0]):
                img = self.zero_pad(target_data[i][j], patch_size)
                pad_target_data[img.shape].append(img)

        target_file.close()

        with h5py.File(pad_path + "pad_target_data.h5" , 'w') as f:
            f.create_dataset("patch_size", data=patch_size)
            for i in pad_target_data:
                f.create_dataset(str(i), data=pad_target_data[i])
        
        pad_file = h5py.File(pad_path + "pad_target_data.h5" , 'r')
        for i in pad_file.keys():
            if i == "patch_size":
                continue
            self.target[i] = pad_file[i][:]
        
    def pad_raw_data(self, patch_size, pad_path, raw_path):
        # input
        raw_data = None
        raw_file = None
        if self.check_file_exist(raw_path, "raw_data_"):
            raw_data, raw_file = self.load_raw_data(raw_path)
        else:
            raw_data, raw_file = self.fetch_raw_data(raw_path)
        
        pad_data = {m : defaultdict(list) for m in self.moda}
        for m in raw_data.keys():
            data = defaultdict(list)
            for i in raw_data[m].keys():
                for j in range(raw_data[m][i].shape[0]):
                    img = self.zero_pad(raw_data[m][i][j], patch_size)
                    data[img.shape].append(img)
            pad_data[m] = data
            
        for f in raw_file:
            f.close()

        for m in pad_data.keys():
            with h5py.File(pad_path + "pad_data_" + m + ".h5" , 'w') as f:
                f.create_dataset("patch_size", data=patch_size)
                for i in pad_data[m]:
                    f.create_dataset(str(i), data=pad_data[m][i])
        
        for m in pad_data.keys():
            pad_file = h5py.File(pad_path + "pad_data_" + m + ".h5" , 'r')
            for i in pad_file.keys():
                if i == "patch_size":
                    continue
                self.input[m][i] = pad_file[i][:]
        
    
    def load_data(self, patch_size=(32, 32, 32), 
                  pad_path="./model/h5df_data/", raw_path="./model/h5df_data/", target_path="./model/h5df_data/"):
                  # pad_path="./model/h5df_data/pad_data.h5", raw_path="./model/h5df_data/raw_data.h5"):
        padded = True
        if self.check_file_exist(raw_path, "pad_data_"):
            for m in self.moda:
                pad_file = h5py.File(pad_path + "pad_data_" + m + ".h5", 'r')
                if np.all(pad_file["patch_size"][:] == list(patch_size)):
                    # self.data = pad_file["pad_data"]
                    for i in pad_file.keys():
                        self.input[m][i] = pad_file[i][:]
                else:
                    padded = False
                    break
        else:
            padded = False
        if padded == False:
            self.pad_raw_data(patch_size, pad_path, raw_path)
        
        if os.path.isfile(target_path + "pad_target_data.h5"):
            pad_file = h5py.File(target_path + "pad_target_data.h5", 'r')
            if np.all(pad_file["patch_size"][:] == list(patch_size)):
                # self.data = pad_file["pad_data"]
                for i in pad_file.keys():
                    self.target[i] = pad_file[i][:]
            else:
                self.pad_target_data(patch_size, pad_path, target_path)
        else:
            self.pad_target_data(patch_size, pad_path, target_path)
    
    ################################################################################
    
    def gen_patch_index(self, patch_size, patch_gap, patch_path):
        count = 0
        patch_index = defaultdict(list)
        # https://arxiv.org/pdf/1710.02316.pdf  at least 0.01% voxels contain lesions
        voxel = int(patch_size[0]*patch_size[1]*patch_size[2]*0.0001)
        
        # patches of segmentation
        for i in self.target:
            if i == "patch_size":
                continue
            shape = self.target[i][0].shape
            patch_num = [int((shape[i]-patch_size[i]) / patch_gap) for i in range(len(shape))]
            for j in range(self.target[i].shape[0]):
                patch_ind = []
                # assume this is a 3d image
                for a in range(patch_num[0]):
                    for b in range(patch_num[1]):
                        for c in range(patch_num[2]):
                            patch_iter = [a * patch_gap, b * patch_gap, c * patch_gap, 1]
                            if (np.sum(self.target[i][j][
                                patch_iter[0]:patch_iter[0] + patch_size[0],
                                patch_iter[1]:patch_iter[1] + patch_size[1],
                                patch_iter[2]:patch_iter[2] + patch_size[2]]) <= voxel):
                                # 0: does not satisfy, need to skip when generating
                                patch_iter[3] = 0
                            patch_ind.append(patch_iter)
                count += len(patch_ind)
                patch_index[i].append(patch_ind)
            for c in range(len(patch_index[i])):
                np.random.shuffle(patch_index[i][c]) # in-place shuffule
        
        with h5py.File(patch_path + "pad_patch_index.h5", 'w') as f:
            f.create_dataset("count", data=count)
            f.create_dataset("patch_size", data=patch_size)
            f.create_dataset("patch_gap", data=patch_gap)
            for i in patch_index:
                f.create_dataset(str(i), data=patch_index[i])
        
        index_file = h5py.File(patch_path + "pad_patch_index.h5", 'r')
        for i in index_file.keys():
            if i == "count" or i == "patch_size" or i == "patch_gap":
                continue
            self.patch_index[i] = index_file[i][:]
        # return the total number of patches
        return index_file["count"][()]

    
    def load_patch_index(self, patch_size, patch_gap, patch_path):
        if os.path.isfile(patch_path + "pad_patch_index.h5"):
            index_file = h5py.File(patch_path + "pad_patch_index.h5", 'r')
            # print(list(pat_ind.keys()))
            if (np.all(index_file["patch_size"][:] == list(patch_size))) and (index_file["patch_gap"][()] == patch_gap):
                for i in index_file.keys():
                    if i == "count" or i == "patch_size" or i == "patch_gap":
                        continue
                    self.patch_index[i] = index_file[i][:]
                return index_file["count"][()]
            else:
                index_file.close()
                return self.gen_patch_index(patch_size, patch_gap, patch_path)
        else:
            return self.gen_patch_index(patch_size, patch_gap, patch_path)
            
    def prekfold(self, patch_size, patch_gap, batch_size, kfold=5, patch_path='./model/h5df_data/'):
        self.kfold = kfold
        self.patch_size = patch_size
        self.patch_gap = patch_gap
        self.batch_size = batch_size

        # initialize validation index for training
        # K-fold LOOCV: leave one out cross validation
        for i in self.target:
            if i == "patch_size":
                continue
            self.valid_index[i] = random.sample(range(self.kfold), self.kfold)

        num = self.load_patch_index(patch_size, patch_gap, patch_path)
        train_num = num // self.kfold * (self.kfold - 1)
        valid_num = num - train_num
        
        return train_num // batch_size, valid_num // batch_size