import h5py
import numpy as np


def save(variables, names, file_path):
    file = h5py.File(file_path, 'w')
    for var, name in zip(variables, names):
        file.create_dataset(name, data=var)
    file.close()


def load(file_path):
    file = h5py.File(file_path, 'r')
    for name in file.keys():
        data = file[name]
        globals()[name] = np.array(data)
    file.close()




