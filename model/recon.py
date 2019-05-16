import numpy as np
import h5py

class Reconstruct:
    def __init__(self, ind, shape):
        # find its original image: d.data[str(shape)][ind][0]
        # find its target image: d.data[str(shape)][ind][1]
        self.ind = ind
        self.shape = shape
        self.data = np.zeros(shape)
        self.count = np.zeros(shape, dtype=np.int)
        
    def add(self, patch, index):
        # get patch data
        patch_index = np.zeros(self.shape, dtype=np.bool)
        patch_index[...,
                    index[0]:index[0]+patch.shape[-3],
                    index[1]:index[1]+patch.shape[-2],
                    index[2]:index[2]+patch.shape[-1]] = True
        patch_data = np.zeros(self.shape)
        patch_data[patch_index] = patch.flatten()
        
        # store patch data in self.data
        new_data_index = np.logical_and(patch_index, np.logical_not(self.count > 0))
        self.data[new_data_index] = patch_data[new_data_index]
        
        # average overlapped region
        averaged_data_index = np.logical_and(patch_index, self.count > 0)
        if np.any(averaged_data_index):
            self.data[averaged_data_index] = (self.data[averaged_data_index] * self.count[averaged_data_index] + 
                                              patch_data[averaged_data_index]) / (self.count[averaged_data_index] + 1)
        self.count[patch_index] += 1
        
    def store(self):
        with h5py.File("./model/h5df_data/reconstruct" + str(self.shape) + ".h5", 'w') as f:
            f.create_dataset("index", data=self.ind)
            f.create_dataset("shape", data=self.shape)
            f.create_dataset("data", data=self.data)
            
        
        