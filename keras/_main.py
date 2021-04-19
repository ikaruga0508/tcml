from .._main import MainBase
from .._loaders import DataLoaderBase
from .._consts import TianchiConsts
from .._utilities import def_log
from ._config import KerasConfig, DEFAULT_MODEL_NAME
from typing import List, Dict, Callable
import abc
import os
import tensorflow as tf
from tensorflow.keras.models import load_model, Model

__all__ = [
    'KerasMainBase',
]


class KerasMainBase(MainBase):
    def __init__(self, data_loader: DataLoaderBase, consts=TianchiConsts(), log: Callable = def_log,
                 use_memory_growth=True) -> None:
        """初始化
        Args:
            data_loader: 数据加载器
            consts: 天池竞赛项目目录结构常量
            log: 日志输出函数
            use_memory_growth: 是否采用显存按需增长
        """
        super().__init__(data_loader, consts=consts, log=log)
        if use_memory_growth:
            self.set_memory_growth()

    def set_fixed_seed(self, seed1: int, seed2: int, *objs, **kwargs) -> None:
        """固化随机种子
        Args:
            seed1: 随机种子1
            seed2: 随机种子2
        """
        super().set_fixed_seed(seed1, seed2)
        seed3 = objs[0]
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        tf.random.set_seed(seed3)
        self.log("设置TF_DETERMINISTIC_OPS='1', tf.random.set_seed({})".format(seed3))

    def set_memory_growth(self) -> None:
        """设置显存按需增长"""
        self.log('设置所有GPU显存按需增长')
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    def load_model_from_local(self, filename: str = DEFAULT_MODEL_NAME, folder: str = None) -> Model:
        """从本地加载模型
        Args:
            filename: 模型文件名
            folder: 模型所在目录路径

        Returns:
            Tensorflow模型
        """
        folder = folder if folder is not None else self.consts.user_folder
        filepath = os.path.join(folder, filename)
        return load_model(filepath, custom_objects=self.get_model_custom_objects())

    @abc.abstractmethod
    def create_model(self) -> Model:
        """从代码创建模型"""
        pass

    @abc.abstractmethod
    def save_results(self, y_pred, ids: List, filepath: str) -> None:
        """保存预测结果
        Args:
            y_pred: 模型预测的输出结果
            ids: 测试集的标识
            filepath: 预测结果保存的路径
        """
        pass

    def get_model_custom_objects(self) -> Dict:
        """模型自定义对象"""
        return {}

    def train_simple_mode(self, config: KerasConfig, val_rate: float = 0.2, folder: str = None) -> None:
        """执行训练(简单模式)
        Args:
            config: 训练参数
            val_rate: 验证集比例
            folder: 模型保存的目录，不设置将会保存在默认目录
        """
        if self.data_loader.loaded:
            dl = self.data_loader
        else:
            dl = self.load_data()
        self.log('开始训练(simple_mode)')
        X_indices, X_val_indices = dl.split_train_indices(val_rate)
        self.log('按比例({})随机切分训练集={}件, 验证集={}件'.format(val_rate, len(X_indices), len(X_val_indices)))

        X_batch_count = dl.calc_batch_count(config.batch_size, len(X_indices))
        X_gen = dl.get_train_generator(config.batch_size, X_indices)
        X_val_batch_count = dl.calc_batch_count(config.batch_size, len(X_val_indices))
        X_val_gen = dl.get_train_generator(config.batch_size, X_val_indices)

        model = self.create_model()
        if config.print_summary:
            model.summary()

        self.log('训练参数: epoch={}, batch_size={}'.format(config.epoch, config.batch_size))
        model.fit(
            X_gen,
            epochs=config.epoch,
            steps_per_epoch=X_batch_count,
            validation_data=X_val_gen,
            validation_steps=X_val_batch_count,
            callbacks=config.callbacks
        )

        if config.save_model:
            folder = folder if folder is not None else self.consts.user_folder
            filepath = os.path.join(folder, config.model_name)
            model.save(filepath)
            self.log('模型已经保存至{}'.format(filepath))

    def predict(self, filename: str, batch_size: int = 32, folder: str = None) -> None:
        """执行预测
        Args:
            filename: 结果文件名
            batch_size: 批次大小
            folder: 结果保存的目录，不设置将会保存在默认目录
        """
        if self.data_loader.loaded:
            dl = self.data_loader
        else:
            dl = self.load_data()
        self.log('开始对测试集预测并生成结果文件')
        model = self.load_model_from_local()

        batch_count = dl.calc_batch_count(batch_size, dl.test_count)
        test_gen = dl.get_test_generator(batch_size)
        y_pred = model.predict(test_gen, steps=batch_count)
        ids = dl.get_test_ids()

        folder = folder if folder is not None else self.consts.output_folder
        filepath = os.path.join(folder, filename)
        self.save_results(y_pred, ids, filepath)
        self.log('预测结果{}件已经保存至{}'.format(len(y_pred), filepath))

    def run_model(self, params):
        """打印模型概要"""
        model = self.create_model()
        model.summary()
