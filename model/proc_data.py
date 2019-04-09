from model.data import *
import h5py
import numpy as np

d = Data()
d.load_data()

def data_to_h5df(data):
    with h5py.File("./model/h5df_data/raw_data.h5", 'w') as f:
        for i in d.data:
            f.create_dataset(str(i), data=d.data[i])

    # check loaded data
    with h5py.File("./model/h5df_data/raw_data.h5", 'r') as f:
        print(list(f.keys()))
        for i in f.keys():
            print(f[i])
            
def pad_