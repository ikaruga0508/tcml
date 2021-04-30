import pandas as pd

from ._loaders import DataLoaderBase
from ._consts import TianchiConsts
from ._utilities import def_log
import abc
import os
import random
import numpy as np
from pandas import DataFrame, Series
from typing import List, Callable
import pickle

__all__ = [
    'MainBase',
    'DataFrameDataMakerBase',
]


class MainBase(abc.ABC):
    def __init__(self, data_loader: DataLoaderBase, consts=TianchiConsts(), log: Callable = def_log) -> None:
        """初始化
        Args:
            data_loader: 数据加载器
            consts: 天池竞赛项目目录结构常量
            log: 日志输出函数
        """
        super().__init__()
        self.data_loader = data_loader
        self.consts = consts
        self.log = log
        self.seed_idx = 0

        # 读取随机数种子矩阵
        self.seed_matrix = self.load_seed_matrix()

        if self.consts.seed_idx in os.environ:
            self.seed_idx = int(os.environ[self.consts.seed_idx])
            self.log('从环境变量中读取`{}`的值为: {}'.format(self.consts.seed_idx, self.seed_idx))
        else:
            self.log('[警告] 环境变量中不存在`{}`，因此默认该值为{}'.format(self.consts.seed_idx, self.seed_idx))

    def load_seed_matrix(self):
        """读取随机数种子矩阵"""
        if os.path.exists(self.consts.seed_matrix):
            return np.load(self.consts.seed_matrix)
        else:
            self.log('[警告] 随机数种子矩阵文件`{}`不存在，可以执行`init`命令创建'.format(self.consts.seed_matrix))
            self.log('[警告] 由于随机数种子矩阵文件`{}`不存在，将被随机创建于内存中'.format(self.consts.seed_matrix))
            return np.random.randint(65535, size=(65535, 100))

    def create_seed_matrix(self):
        """创建随机数种子矩阵"""
        os.makedirs(self.consts.data_folder, exist_ok=True)
        seed_matrix = np.random.randint(65535, size=(65535, 100))
        np.save(self.consts.seed_matrix, seed_matrix)
        self.log('随机数种子矩阵{}已被创建在{}'.format(seed_matrix.shape, self.consts.seed_matrix))

    def set_fixed_seed(self, seed1: int, seed2: int, *objs, **kwargs) -> None:
        """固化随机种子
        Args:
            seed1: 随机种子1
            seed2: 随机种子2
        """
        os.environ['PYTHONHASHSEED'] = '0'
        random.seed(seed1)
        np.random.seed(seed2)
        self.log('设置PYTHONHASHSEED=0, random.seed({}), np.random.seed({})'.format(seed1, seed2))

    def get_seeds(self):
        """获得随机数数组"""
        return self.seed_matrix[self.seed_idx]

    def load_data(self, *objs, **kwargs) -> DataLoaderBase:
        """加载数据
        Args:
            *objs: 参数数组
            **kwargs: 参数字典

        Returns:
            数据加载器
            @rtype: object
        """
        self.log('开始加载数据')
        self.data_loader.load(*objs, **kwargs)
        return self.data_loader

    def get_default_entry(self) -> Callable[[List], None]:
        """默认入口"""
        return None

    def handle(self, argv: List) -> None:
        """处理函数
        Args:
            argv: 命令行参数
        """
        if len(argv) > 1:
            cmd = argv[1]
            method_name = 'self.run_{}'.format(cmd)
            self.log('执行 {}([{}])'.format(method_name, ', '.join(map(lambda x: "'{}'".format(x), argv[2:]))))
            eval(method_name)(argv[2:])
        else:
            default_entry = self.get_default_entry()
            if default_entry is not None:
                default_entry([])
            else:
                self.log('参数错误(至少有一个)')

    def run_init(self, params):
        """初始化"""
        # 创建所有的目录
        folders = [
            self.consts.user_folder,
            self.consts.tmp_folder,
            self.consts.output_folder,
            self.consts.data_folder,
            self.consts.logs_folder,
        ]
        for folder in folders:
            os.makedirs(folder, exist_ok=True)

        self.log('工程目录已被创建')
        self.create_seed_matrix()


class DataFrameDataMakerBase(MainBase):
    @staticmethod
    def get_train_data_filename() -> str:
        """处理后训练集的文件名"""
        return 'train.pkl'

    @staticmethod
    def get_test_data_filename() -> str:
        """处理后测试集的文件名"""
        return 'test.pkl'

    @abc.abstractmethod
    def _pre_process(self, df: DataFrame, train_idx, test_idx, labels, label_name) -> DataFrame:
        """预处理"""
        pass

    @abc.abstractmethod
    def _drop_features(self, df: DataFrame, train_idx, test_idx, labels, label_name) -> DataFrame:
        """删除不要的特征"""
        pass

    @abc.abstractmethod
    def _fillna(self, df: DataFrame, train_idx, test_idx, labels, label_name) -> DataFrame:
        """填充必要的NA"""
        pass

    @abc.abstractmethod
    def _fillna_others(self, df: DataFrame, train_idx, test_idx, labels, label_name) -> DataFrame:
        """当模型不允许传入NA时，填充剩余所有NA"""
        pass

    @abc.abstractmethod
    def _convert_features(self, df: DataFrame, train_idx, test_idx, labels, label_name) -> DataFrame:
        """处理或转换现有的特征"""
        pass

    @abc.abstractmethod
    def _new_features(self, df: DataFrame, train_idx, test_idx, labels, label_name) -> DataFrame:
        """使用现有的特征生成新的特征"""
        pass

    @abc.abstractmethod
    def _drop_features_handled(self, df: DataFrame, train_idx, test_idx, labels, label_name) -> DataFrame:
        """其他处理完毕后，删除不要的特征"""
        pass

    @abc.abstractmethod
    def _post_process(self, df: DataFrame, train_idx, test_idx, labels, label_name) -> DataFrame:
        """后处理"""
        pass

    def make(self, udf: DataFrame, train_idx, test_idx, labels, label_name, folder, description, pipeline,
             contain_labels: bool = True):
        """生成处理过后的训练集和测试集
        Args:
            udf: 合并后的数据集，使用DataFrameLoaderBase.get_union_samples()可以获得
            train_idx: 训练集索引，获得方法同上
            test_idx: 测试集索引，获得方法同上
            labels: 标签，获得方法同上
            label_name: 标签的列名
            folder: 数据生成在哪个子目录下
            description: 表述语句，用于日志输出
            pipeline: 处理的流水线
            contain_labels: 数据在传入流水线前是否拼接上标签(为了TargetEncoding)
        """
        self.log(description)
        # 为了不影响后续其他的生成处理，创建拷贝
        udf_cloned = udf.copy()

        # 是否拼接上标签列
        if contain_labels:
            udf_cloned[label_name] = labels

        # 执行流水线
        for f in pipeline:
            udf_cloned = f(udf_cloned, train_idx, test_idx, labels, label_name)

        # 去除标签
        if contain_labels:
            udf_cloned.drop(columns=[label_name], inplace=True)

        os.makedirs(os.path.join(self.consts.tmp_folder, folder), exist_ok=True)

        train_df = udf_cloned.loc[train_idx]
        train_df[label_name] = labels
        train_filepath = os.path.join(self.consts.tmp_folder, folder, self.get_train_data_filename())
        train_df.to_pickle(train_filepath, protocol=pickle.DEFAULT_PROTOCOL)
        self.log('共计{}行，{}列的数据保存至`{}`文件中'.format(len(train_df), len(train_df.columns), train_filepath))

        test_df = udf_cloned.loc[test_idx]
        test_filepath = os.path.join(self.consts.tmp_folder, folder, self.get_test_data_filename())
        test_df.to_pickle(test_filepath, protocol=pickle.DEFAULT_PROTOCOL)
        self.log('共计{}行，{}列的数据保存至`{}`文件中'.format(len(test_df), len(test_df.columns), test_filepath))

    def freq_encoding(self, targets: List, udf: DataFrame, train_idx, aggregate_on_all: bool = False,
                      new_column_names: List = None) -> DataFrame:
        """频度编码
        Args:
            targets: 目标列数组
            udf: 合并后的数据集
            train_idx: 训练集索引
            aggregate_on_all: 是否在合并后的数据上做聚合计算(False的话只在训练集上)
            new_column_names: 新列的名字数组
                              如果为None的话，全部默认为`{target}_freq`
                              如果不为None，数组长度需要和targets相同，并且其中如果存在None元素，则设置为`{target}_freq`
        """
        if new_column_names is None:
            new_column_names = [None] * len(targets)

        udf_cloned = udf.copy()

        for target, new_column_name in zip(targets, new_column_names):
            if aggregate_on_all:
                aggr_data = udf[target]
            else:
                aggr_data = udf.loc[train_idx, target]

            if new_column_name is None:
                new_column_name = '{}_freq'.format(target)

            values = aggr_data.value_counts() / len(aggr_data)
            values.name = new_column_name
            udf_cloned = udf_cloned.merge(values, how='left', left_on=target, right_index=True)

        return udf_cloned

    def target_encoding(self, targets, udf: DataFrame, train_idx, test_idx, labels, label_name,
                        aggregate_func: Callable[[DataFrame, str, str], Series] = None,
                        new_column_names: List = None, aggregate_name: str = 'mean',
                        n_splits=5, stratified=True, shuffle=True, shuffle_in_indices=True, verbose=True,
                        random_state: int = None
                        ) -> DataFrame:
        """目标编码
        Args:
            targets: 目标列数组
            udf: 合并后的数据集
            train_idx: 训练集索引
            test_idx: 测试集索引
            labels: 标签
            label_name: 标签的列名
            aggregate_func: 聚合函数
            new_column_names: 新列的名字数组
                              如果为None的话，全部默认为`{target}_{aggregate_name}`
                              如果不为None，数组长度需要和targets相同，并且其中如果存在None元素，则设置为`{target}_{aggregate_name}`
            aggregate_name: 聚合名称

            # 以下参数用于get_train_sample_indices_for_kfold()
            n_splits: 折数
            stratified: 是否分层(保持标签的分布相同)
            shuffle: 在K折处理中是否打乱顺序(该参数传递给KFold或者StratifiedKFold)
            shuffle_in_indices: KFold或者StratifiedKFold返回的索引顺序是从小到大的，是否打乱索引顺序
            verbose: 打印Debug信息
            random_state: 随机状态
        """
        if new_column_names is None:
            new_column_names = [None] * len(targets)

        # 把特征全部转换成数组形式
        targets = [[target] if not isinstance(target, (list, tuple)) else list(target) for target in targets]

        def _make_new_column_name(ori_col, new_col):
            if new_col is None:
                return '{}_{}'.format('_'.join(ori_col), aggregate_name)
            else:
                return new_col

        new_column_names = list(map(lambda x: _make_new_column_name(x[0], x[1]), zip(targets, new_column_names)))

        if aggregate_func is None:
            aggregate_func = self.aggr_mean

        udf_cloned = udf.copy()
        train_df = udf.loc[train_idx].copy()
        test_df = udf.loc[test_idx].copy()

        if label_name not in train_df:
            train_df[label_name] = labels

        X_indices_list, X_val_indices_list = self.data_loader.get_train_sample_indices_for_kfold(
            n_splits, stratified, shuffle, shuffle_in_indices, verbose, random_state)

        new_columns_dict = {}
        # 对训练集数据进行填充
        for X_indices, X_val_indices in zip(X_indices_list, X_val_indices_list):
            # 通过get_train_sample_indices_for_kfold()得到的索引都是基于0开始的[0, N-1]，真实索引可能乱序也不一定基于0
            # 所以需要通过以下代码获得真正的索引
            X_indices = train_idx[X_indices]
            X_val_indices = train_idx[X_val_indices]

            # X_val_indices上数据的编码来源于X_indices上数据的聚合运算
            for target, new_column_name in zip(targets, new_column_names):
                new_columns_dict.setdefault(new_column_name, [])

                # 获得用于聚合运算的数据
                raw_values = train_df.loc[X_indices, [*target, label_name]]
                agg_values = aggregate_func(raw_values, target, label_name)
                agg_values.name = new_column_name
                values = train_df.loc[X_val_indices].merge(
                    agg_values, how='left', left_on=target, right_index=True)[[*target, new_column_name]]
                # 对仅在X_val_indices上存在的个别值，使用X_indices上计算得到的全局平均值进行填充
                values.loc[~values[target].isna().any(axis=1) & values[new_column_name].isna(), new_column_name]\
                    = raw_values[label_name].mean()
                new_columns_dict[new_column_name].append(values)

        # 对测试集数据进行填充
        # 如果某原始列的某值在训练集中存在，则使用该值对应新列所有值的平均值
        # 如果不存在(连接后会导致新值为NA)，则使用新列所有值的平均值来填充NA
        for target, new_column_name in zip(targets, new_column_names):
            values_on_train = pd.concat(new_columns_dict[new_column_name])
            agg_on_train = values_on_train.groupby(target).mean(new_column_name)
            values = test_df.merge(
                agg_on_train, how='left', left_on=target, right_index=True)[[*target, new_column_name]]
            # 原始列本来就是NA的不进行填充
            values.loc[~values[target].isna().any(axis=1) & values[new_column_name].isna(), new_column_name]\
                = values_on_train[new_column_name].mean()
            new_columns_dict[new_column_name].append(values)

        for new_column_name, values in new_columns_dict.items():
            udf_cloned[new_column_name] = pd.concat(values)[new_column_name]

        return udf_cloned

    @staticmethod
    def aggr_mean(df: DataFrame, target, label_name) -> Series:
        """均值编码
        Args:
            df: 聚合运算的数据集
            target: 目标列
            label_name: 标签的列名

        Returns:
            均值编码后的Series对象
        """
        return df.groupby(target).agg({label_name: 'mean'})[label_name]
