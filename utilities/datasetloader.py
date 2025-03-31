import scipy.io.matlab

class DatasetLoadUtility:
    def __init__(self):
        pass

    @staticmethod
    def load_dataset_from_mat_file(filename='-1'):
        dataset = scipy.io.loadmat(filename)
        u, y, u_val, y_val = [dataset.get(x) for x in ['U', 'Y', 'U_val', 'Y_val']]
        return u, y, u_val, y_val

    @staticmethod
    def load_field_from_mat_file(filename, fields):
        dataset = scipy.io.loadmat(filename)
        return [dataset.get(x) for x in fields]
