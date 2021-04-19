from ._loaders import DataLoaderBase
from ._consts import TianchiConsts
from ._utilities import def_log
import abc
import os
import random
import numpy as np
from typing import List, Callable

__all__ = [
    'MainBase',
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
