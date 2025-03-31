import numpy as np
from enum import Enum
from dynamical_systems.DummyModel import DummyModel
from utilities.datasetloader import DatasetLoadUtility
from dynamical_systems.LinearComplexSystem import LinearSystemComplexStable
from dynamical_systems.TwoTanks import TwoTanks
from dynamical_systems.GearSystem import GearSystem

class SystemSelectorEnum(Enum):
    @staticmethod
    def load_from_dataset(filename, non_linear_input=False):
        dynamic_model = DummyModel()
        ds_loading = DatasetLoadUtility()
        u, y, u_v, y_v = ds_loading.load_dataset_from_mat_file(filename)
        size_t = u.shape[0]
        size_v = u_v.shape[0]
        u_n = np.reshape(u.T[0], (size_t, 1))
        y_n = np.reshape(y.T[0], (size_t, 1))
        u_vn = np.reshape(u_v.T[0], (size_v, 1))
        y_vn = np.reshape(y_v.T[0], (size_v, 1))
        mean_y = np.mean(y_n)
        mean_u = np.mean(u_n)
        std_y = np.std(y_n)
        std_u = np.std(u_n)
        y_n = (y_n - mean_y) / std_y
        y_vn = (y_vn - mean_y) / std_y
        u_n = (u_n - mean_u) / std_u
        u_vn = (u_vn - mean_u) / std_u
        return dynamic_model, u_n, y_n, u_vn, y_vn
    
    @staticmethod
    def linear():
        dynamic_model = LinearSystemComplexStable()
        u, y, u_val, y_val = dynamic_model.prepare_dataset(20000, 2000)
        return dynamic_model, u, y, u_val, y_val
    
    @staticmethod
    def pwa():
        dynamic_model = GearSystem()
        u, y, u_val, y_val = dynamic_model.prepare_dataset(20000, 2000)
        return dynamic_model, u, y, u_val, y_val
    
    @staticmethod
    def nonlinear():
        dynamic_model = TwoTanks()
        u, y, u_val, y_val = dynamic_model.prepare_dataset(20000, 2000)
        return dynamic_model, u, y, u_val, y_val

    

