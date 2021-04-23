from ._loaders import DataLoaderBase
from ._consts import TianchiConsts
from ._utilities import def_log
import abc
import os
import random
import numpy as np
from pandas import DataFrame
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
    def _pre_process(self, df: DataFrame) -> DataFrame:
        """预处理"""
        pass

    @abc.abstractmethod
    def _drop_features(self, df: DataFrame) -> DataFrame:
        """删除不要的特征"""
        pass

    @abc.abstractmethod
    def _fillna(self, df: DataFrame) -> DataFrame:
        """填充必要的NA"""
        pass

    @abc.abstractmethod
    def _fillna_others(self, df: DataFrame) -> DataFrame:
        """当模型不允许传入NA时，填充剩余所有NA"""
        pass

    @abc.abstractmethod
    def _convert_features(self, df: DataFrame) -> DataFrame:
        """处理或转换现有的特征"""
        pass

    @abc.abstractmethod
    def _new_features(self, df: DataFrame) -> DataFrame:
        """使用现有的特征生成新的特征"""
        pass

    @abc.abstractmethod
    def _drop_features_handled(self, df: DataFrame) -> DataFrame:
        """其他处理完毕后，删除不要的特征"""
        pass

    @abc.abstractmethod
    def _post_process(self, df: DataFrame) -> DataFrame:
        """后处理"""
        pass

    def make(self, udf: DataFrame, train_idx, test_idx, labels, label_name, folder, description, pipeline):
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
        """
        self.log(description)
        udf_cloned = udf.copy()
        for f in pipeline:
            f(udf_cloned)

        os.makedirs(os.path.join(self.consts.tmp_folder, folder), exist_ok=True)

        train_df = udf_cloned.loc[train_idx]
        train_df[label_name] = labels
        train_df.to_pickle(os.path.join(
            self.consts.tmp_folder, folder, self.get_train_data_filename()), protocol=pickle.DEFAULT_PROTOCOL)

        test_df = udf_cloned.loc[test_idx]
        test_df.to_pickle(os.path.join(
            self.consts.tmp_folder, folder, self.get_test_data_filename()), protocol=pickle.DEFAULT_PROTOCOL)
