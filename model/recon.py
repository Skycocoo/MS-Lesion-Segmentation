import numpy as np
import h5py

class Reconstruct:
    def __init__(self, ind, shape, patch_size):
        # find its original image: d.data[str(shape)][ind][0]
        # find its target image: d.data[str(shape)][ind][1]
        self.ind = ind
        self.shape = shape
        self.patch_size = patch_size
        self.data = np.zeros(shape)
        self.count = np.zeros(shape, dtype=np.int)
        
#         construct softmax map for distance from the boundary
#         self.dist_map = np.zeros(patch_size)
#         mini = 0
#         minj = 0
#         mink = 0
#         for i in range(patch_size[0]):
#             mini = min(i+1, patch_size[0]-i)
#             for j in range(patch_size[1]):
#                 minj = min(j+1, patch_size[1]-j)
#                 for k in range(patch_size[2]):
#                     mink = min(k+1, patch_size[2]-k)
# #                     print(i, j, k, mini, minj, mink)
#                     self.dist_map[i, j, k] = min(mini, minj, mink)
# #         print(self.dist_map)
#         self.dist_map = np.exp(self.dist_map)/np.sum(np.exp(self.dist_map))

        self.dist_map = np.zeros(patch_size)
        center = (np.array(patch_size)-1) / 2
        center_dist = np.linalg.norm(center)
        for i in range(patch_size[0]):
            for j in range(patch_size[1]):
                for k in range(patch_size[2]):
#                     print([i, j, k], np.array([i, j, k]) - center)
                    self.dist_map[i, j, k] = center_dist - np.linalg.norm(np.array([i, j, k]) - center)
#         print(self.dist_map)
        self.dist_map[self.dist_map < 0] = 0
        self.dist_map = np.exp(self.dist_map)/np.sum(np.exp(self.dist_map))
#         print(self.dist_map)

        
    def add(self, patch, index):
        patch = patch * self.dist_map
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
        
    def store(self, name):
        with h5py.File("./model/h5df_data/reconstruct_" + name + "_" +  str(self.shape) + ".h5", 'w') as f:
            f.create_dataset("index", data=self.ind)
            f.create_dataset("shape", data=self.shape)
            f.create_dataset("data", data=self.data)