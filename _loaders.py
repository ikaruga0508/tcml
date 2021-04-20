from ._consts import TianchiConsts
from ._loader_configs import KFoldCrossValidationConfig
from ._utilities import def_log, reduce_memory
import abc
import numpy as np
import pandas as pd
from typing import List, Callable, Tuple, Any
from sklearn.model_selection import StratifiedKFold, KFold

__all__ = [
    'DataLoaderBase',
    'DataFrameLoaderBase',
    'DataGeneratorBase',
]


class DataLoaderBase(abc.ABC):
    def __init__(self, consts=TianchiConsts(), log: Callable = def_log) -> None:
        """初始化
        Args:
            consts: 天池竞赛项目目录结构常量
            log: 日志输出函数
        """
        super().__init__()
        self.log = log
        self.consts = consts
        self.loaded = False
        self._train_samples = None
        self._test_samples = None
        # 交叉验证相关
        self.cv_config = None

    def load(self, testset_from='A', dataset_from=None):
        """数据加载
        Args:
            testset_from: 测试集来源，任意形式，通常为'A'或者'B'。使用方法由派生类的重载函数所决定。
            dataset_from: 训练集来源，任意形式。使用方法由派生类的重载函数所决定。
        """
        self._train_samples, self._test_samples = self.load_train_test_dataset(testset_from, dataset_from)
        self.log('训练集件数: {}, 来自: {}'.format(self.train_count, dataset_from if dataset_from is not None else '默认'))
        self.log('测试集件数: {}, 来自: {}'.format(self.test_count, testset_from))
        self.loaded = True

    @abc.abstractmethod
    def load_train_test_dataset(self, testset_from='A', dataset_from=None) -> (Any, Any):
        """加载训练集和测试集
        Args:
            testset_from: 测试集来源，任意形式，通常为'A'或者'B'。使用方法由派生类的重载函数所决定。
            dataset_from: 训练集来源，任意形式。使用方法由派生类的重载函数所决定。

        Returns:
            (训练集, 测试集)
        """
        pass

    @property
    def train_samples(self):
        """获得训练集"""
        return self._train_samples

    @train_samples.setter
    def train_samples(self, train_samples):
        """设置训练集"""
        self._train_samples = train_samples

    @property
    def test_samples(self):
        """获得测试集"""
        return self._test_samples

    @test_samples.setter
    def test_samples(self, test_samples):
        """设置测试集"""
        self._test_samples = test_samples

    @abc.abstractmethod
    def get_train_labels(self):
        """获得训练集的标签"""
        pass

    @property
    def train_count(self):
        """训练集数量"""
        return len(self._train_samples) if self._train_samples is not None else None

    @property
    def test_count(self):
        """测试集数量"""
        return len(self._test_samples) if self._test_samples is not None else None

    def preprocessing(self):
        """数据预处理
        self.load()执行完毕后调用，可以对self._train_samples和self._test_samples进行一些预处理
        """
        self.log('对数据进行预处理')

    def get_train_sample_indices(self, sample_count: int = -1, shuffle: bool = True):
        """获得训练集的乱序索引列表
        Args:
            sample_count: 样本数量，-1则为全部索引
            shuffle: 是否打乱。若为False，则返回[0:sample_count]前sample_count个样本数据

        Returns:
            索引列表
        """
        assert (sample_count <= self.train_count)
        return self._get_sample_indices(np.arange(self.train_count), sample_count, shuffle)

    def get_test_sample_indices(self, sample_count=-1, shuffle=False):
        """获得测试集的索引列表(默认不打乱)
        Args:
            sample_count: 样本数量，-1则为全部索引
            shuffle: 是否打乱。若为False，则返回[0:sample_count]前sample_count个样本数据

        Returns:
            索引列表
        """
        assert (sample_count <= self.test_count)
        return self._get_sample_indices(np.arange(self.test_count), sample_count, shuffle)

    def split_train_indices(self, *rates) -> Tuple:
        """切分训练集索引
        Args:
            rates: 除了第一份以外的切分比例

        Returns
            (剩下比例的索引, rates[0]比例的索引, ...)
        """
        assert (np.sum(rates) < 1.)
        # 计算参数rates每一部分的索引数量
        sample_counts = [int(rate * self.train_count) for rate in rates]
        # 剩下的部分作为训练集
        X_count = self.train_count - np.sum(sample_counts)
        # 取出所有索引并打乱
        indices = self.get_train_sample_indices()
        # 使用打乱后的索引，顺序分割训练集
        indices_list = []
        idx = 0
        for count in [X_count, *sample_counts]:
            indices_list.append(indices[idx:idx+count])
            idx += count

        return tuple(indices_list)

    @staticmethod
    def _get_sample_indices(idx_list, sample_count: int, shuffle: bool):
        """获得索引列表
        Args:
            idx_list: 升序的索引列表
            sample_count: 样本数量，-1则为全部索引
            shuffle: 是否打乱。若为False，则返回[0:sample_count]前sample_count个样本数据

        Returns:
            索引列表
        """
        if shuffle:
            if sample_count < 0:
                return np.random.choice(idx_list, len(idx_list), replace=False)
            else:
                return np.random.choice(idx_list, sample_count, replace=False)
        else:
            if sample_count < 0:
                return idx_list
            else:
                return idx_list[0:sample_count]

    @staticmethod
    def _enumerate_with_rows(samples, indices=None, fast_mode=True):
        """按行枚举
        Args:
            samples: 样本，类型可以为list, numpy.ndarray或者pandas.DataFrame
            indices: 索引列表
            fast_mode: 当使用DataFrame时，可以选择是否使用快速模式。高速模式下使用getattr(v, <列名>)来获得数据，否则使用v[<列名>]来获得数据。

        Returns:
            可使用enumerate按行来枚举的对象
        """
        if isinstance(samples, pd.DataFrame):
            _samples = samples if indices is None else samples.loc[indices]
            return _samples.itertuples() if fast_mode else _samples.iterrows()
        else:
            _samples = samples if indices is None else samples[indices]
            return _samples

    @staticmethod
    def calc_batch_count(batch_size: int, sample_count: int) -> int:
        """计算批次数量
        Args:
            batch_size: 批次大小
            sample_count: 样本数量

        Returns:
            批次数量
        """
        return ((sample_count - 1) // batch_size) + 1

    def get_train_sample_indices_for_kfold(self, n_splits=5, stratified=True, shuffle=True, shuffle_in_indices=True):
        """获得K折后的训练集的索引矩阵
        Args:
            n_splits: 折数
            stratified: 是否分层(保持标签的分布相同)
            shuffle: 在K折处理中是否打乱顺序(该参数传递给KFold或者StratifiedKFold)
            shuffle_in_indices: KFold或者StratifiedKFold返回的索引顺序是从小到大的，是否打乱索引顺序

        Returns:
            (训练集的索引矩阵, 测试集的索引矩阵)
            其中，索引矩阵的形状为(n_splits, 训练集/测试集的样本数)
        """
        # 按顺序获得所有的标签
        labels = self.get_train_labels()
        assert (len(labels) > 0)

        # 若为One-Hot向量则自动转化为单值标签
        if isinstance(labels[0], list) and len(labels[0]) > 1:
            labels = np.argmax(labels, axis=-1)

        # K折无需真实数据，为了节省内存，制造假数据
        X_dummy = np.zeros_like(labels)

        if stratified:
            kfold = StratifiedKFold(n_splits, shuffle=shuffle)
            gen = kfold.split(X_dummy, labels)
        else:
            kfold = KFold(n_splits, shuffle=shuffle)
            gen = kfold.split(X_dummy, labels)

        X_indices_groups = []
        X_test_indices_groups = []
        for X_indices, X_test_indices in gen:
            if shuffle_in_indices:
                np.random.shuffle(X_indices)
                np.random.shuffle(X_test_indices)

            X_indices_groups.append(X_indices)
            X_test_indices_groups.append(X_test_indices)

        return X_indices_groups, X_test_indices_groups


class DataFrameLoaderBase(DataLoaderBase):
    @abc.abstractmethod
    def get_label_column(self) -> str:
        """获得标签列名"""
        pass

    @property
    def train_samples(self) -> pd.DataFrame:
        """获得训练集"""
        return self._train_samples

    @train_samples.setter
    def train_samples(self, train_samples: pd.DataFrame):
        """设置训练集"""
        self._train_samples = train_samples

    @property
    def test_samples(self) -> pd.DataFrame:
        """获得测试集"""
        return self._test_samples

    @test_samples.setter
    def test_samples(self, test_samples: pd.DataFrame):
        """设置测试集"""
        self._test_samples = test_samples

    def get_train_labels(self):
        """获得训练集的标签"""
        return self.train_samples[self.get_label_column()]

    @abc.abstractmethod
    def get_test_ids(self) -> List:
        """获得测试集的标识"""
        pass

    def apply_reduce_memory(self) -> None:
        """申请减少训练集和测试集的内存使用量"""
        self.train_samples = reduce_memory(self.train_samples, self.log, '训练集')
        self.test_samples = reduce_memory(self.test_samples, self.log, '测试集')

    def get_union_samples(self):
        """获得训练集和测试集合并后的结果
        Returns:
            (合并后样本集, 训练集索引, 测试集索引, 训练集标签)
        """
        train_idx = self.train_samples.index
        test_idx = self.test_samples.index
        labels = self.get_train_labels()
        union_df = pd.concat([self.train_samples[self.test_samples.columns], self.test_samples])
        return union_df, train_idx, test_idx, labels


class DataGeneratorBase(DataLoaderBase):
    def get_train_samples(self, indices, fast_mode: bool = True, contain_id: bool = False):
        """获得训练样本数据
        Args:
            indices: 索引列表
            fast_mode: 当使用DataFrame时，可以选择是否使用快速模式。高速模式下使用getattr(v, <列名>)来获得数据，否则使用v[<列名>]来获得数据。
            contain_id: 返回值中是否包含标识

        Returns:
            contain_id为False时: (索引指向的样本数据数组, 索引指向的样本标签数组)
            contain_id为True时: (索引数组, 索引指向的样本数据数组, 索引指向的样本标签数组)
        """
        items = list(zip(indices, self._enumerate_with_rows(self._train_samples, indices, fast_mode)))

        if contain_id:
            return np.array([self._map_train_id(v, i) for i, v in items]), \
                   np.array([self._map_train_data(v, i) for i, v in items]), \
                   np.array([self._map_train_label(v, i) for i, v in items])
        else:
            return np.array([self._map_train_data(v, i) for i, v in items]), \
                   np.array([self._map_train_label(v, i) for i, v in items])

    def get_test_samples(self, indices: List = None, fast_mode=True):
        """获得测试样本数据
        Args:
            indices: 索引列表
            fast_mode: 当使用DataFrame时，可以选择是否使用快速模式。高速模式下使用getattr(v, <列名>)来获得数据，否则使用v[<列名>]来获得数据。

        Returns:
            索引指向的样本数据数组
        """
        indices = indices if indices is not None else self.get_test_sample_indices()
        rows = self._enumerate_with_rows(self._test_samples, indices, fast_mode)
        return np.array([self._map_test_data(v, i) for i, v in enumerate(rows)])

    def get_test_ids(self, indices: List = None, fast_mode=True):
        """获得测试样本标识
        Args:
            indices: 索引列表
            fast_mode: 当使用DataFrame时，可以选择是否使用快速模式。高速模式下使用getattr(v, <列名>)来获得数据，否则使用v[<列名>]来获得数据。

        Returns:
            索引指向的样本标识数组
        """
        indices = indices if indices is not None else self.get_test_sample_indices()
        rows = self._enumerate_with_rows(self._test_samples, indices, fast_mode)
        return np.array([self._map_test_id(v, i) for i, v in enumerate(rows)])

    def get_train_generator(self, batch_size: int, indices: List, fast_mode: bool = True, contain_id: bool = False):
        """获得训练集上的迭代器
        Args:
            batch_size: 批次大小
            indices: 索引列表
            fast_mode: 当使用DataFrame时，可以选择是否使用快速模式。高速模式下使用getattr(v, <列名>)来获得数据，否则使用v[<列名>]来获得数据。
            contain_id: 返回值中是否包含标识

        Returns:
            迭代器
        """
        while True:
            start = 0
            while start < len(indices):
                one_batch_indices = indices[start:start + batch_size]
                start += batch_size
                yield self.get_train_samples(one_batch_indices, fast_mode=fast_mode, contain_id=contain_id)

    def get_test_generator(self, batch_size: int, indices: List = None):
        """获得测试集上的迭代器
        Args:
            batch_size: 批次大小
            indices: 索引列表

        Returns:
            迭代器
        """
        indices = indices if indices is not None else np.arange(0, len(self._test_samples))
        while True:
            start = 0
            while start < len(indices):
                one_batch_indices = indices[start:start + batch_size]
                start += batch_size
                yield self.get_test_samples(one_batch_indices)

    @abc.abstractmethod
    def _map_train_label(self, value, index):
        """映射训练集标签
        Args:
            value: 单件样本
            index: 样本索引

        Returns:
            标签值
        """
        pass

    @abc.abstractmethod
    def _map_train_data(self, value, index):
        """映射训练集训练数据
        Args:
            value: 单件样本
            index: 样本索引

        Returns:
            训练数据
        """
        pass

    def _map_train_id(self, value, index):
        """映射训练集标识
        Args:
            value: 单件样本
            index: 样本索引

        Returns:
            标识
        """
        return index

    def _map_test_id(self, value, index):
        """映射测试集标识
        Args:
            value: 单件样本
            index: 样本索引

        Returns:
            标识
        """
        return index

    def _map_test_data(self, value, index):
        """映射测试集数据
        Args:
            value: 单件样本
            index: 样本索引

        Returns:
            测试数据
        """
        return value

    def get_train_labels(self):
        """获得训练集标签"""
        rows = self._enumerate_with_rows(self._train_samples)
        return [self._map_train_label(v, i) for i, v in enumerate(rows)]

    def _set_cv_config(self, config: KFoldCrossValidationConfig):
        """设置交叉验证设定
        Args:
            config: 交叉验证设定
        """
        self.cv_config = config

    def apply_default_cv(self, n_splits):
        """启用默认的交叉验证
        Args:
            n_splits: 折数
        """
        cv_config = KFoldCrossValidationConfig(*self.get_train_sample_indices_for_kfold(n_splits))
        self._set_cv_config(cv_config)
