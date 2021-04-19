import os


class TianchiConsts:
    def __init__(self):
        # 目录
        self._root_folder = '..'
        self._data_folder = 'tcdata'
        self._output_folder = 'result'
        self._user_folder = 'user_data'
        self._logs_folder = 'logs'
        self._tmp_folder = 'tmp_data'
        # 文件
        self._seed_matrix = 'seed_matrix.npy'

    @property
    def root_folder(self):
        return self._root_folder

    @property
    def data_folder(self):
        return os.path.join(self.root_folder, self._data_folder, '')

    @property
    def output_folder(self):
        return os.path.join(self.root_folder, self._output_folder, '')

    @property
    def user_folder(self):
        return os.path.join(self.root_folder, self._user_folder, '')

    @property
    def logs_folder(self):
        return os.path.join(self.root_folder, self._logs_folder, '')

    @property
    def tmp_folder(self):
        return os.path.join(self.root_folder, self._tmp_folder, '')

    @property
    def seed_matrix(self):
        return os.path.join(self.user_folder, self._seed_matrix)

    @property
    def seed_idx(self) -> str:
        return 'seed_idx'
