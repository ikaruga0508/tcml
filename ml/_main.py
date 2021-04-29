import abc
import os.path

import pandas as pd
import pickle

from .._main import MainBase
from .._loaders import DataFrameLoaderBase
from typing import Callable, List
from pandas import DataFrame, Series
import numpy as np
import glob

__all__ = [
    'MachineLearningMainBase',
]


class MachineLearningMainBase(MainBase):
    def train_predict_simple_mode(
            self,
            dl: DataFrameLoaderBase,
            fit_func: Callable[[DataFrame, Series, DataFrame, Series], None],
            predict_func: Callable[[DataFrame], List],
            complete_func: Callable[[], None] = None,
            val_rate: float = 0.2,
            result_filepath: str = None
    ) -> None:
        """执行训练与预测(简单模式)
        Args:
            dl: 用于DataFrame的数据加载器
            fit_func: 训练回调函数，需在里面完成fit()
                      参数: 训练集数据, 训练集标签, 验证集数据, 验证集标签
                      返回值: 无
            predict_func: 预测回调函数，需在里面完成predict()或者predict_proba()
                      参数: 测试集数据
                      返回值: 预测结果
            complete_func: 完成时回调函数，其中可包含任意操作
                      参数: 无
                      返回值: 无
            val_rate: 验证集比例
            result_filepath: 预测结果保存的路径
        """
        self.log('开始训练(simple_mode)')
        X_indices, X_val_indices = dl.split_train_indices(val_rate)
        labels = dl.get_train_labels()
        ids = dl.get_test_ids()
        self.log('划分比例: {:.2f}, 训练集: {}件, 验证集: {}件'.format(val_rate, len(X_indices), len(X_val_indices)))

        X = dl.train_samples.loc[X_indices].drop(columns=[dl.get_label_column()])
        y = labels.loc[X_indices]
        X_val = dl.train_samples.loc[X_val_indices].drop(columns=[dl.get_label_column()])
        y_val = labels.loc[X_val_indices]
        X_test = dl.test_samples

        fit_func(X, y, X_val, y_val)
        if result_filepath is not None:
            folder = os.path.dirname(result_filepath)
            if os.path.exists(folder):
                y_pred = predict_func(X_test)
                self.save_results(y_pred, ids, result_filepath)
                self.log('预测结果保存至`{}`'.format(result_filepath))
            else:
                self.log('预测结果无法生成，上层目录不`{}`存在'.format(folder))

        if complete_func is not None:
            complete_func()

    def train_predict_with_cv(
            self,
            dl: DataFrameLoaderBase,
            fit_func: Callable[[DataFrame, Series, DataFrame, Series], None],
            predict_func: Callable[[DataFrame], List],
            complete_func: Callable[[DataFrame], None] = None,
            n_splits: int = 5,
            stratified: bool = True,
            result_filepath: str = None,
            result_folder_for_ensemble: str = None,
            random_state=None
    ) -> None:
        """执行训练与预测(交叉验证模式)
        Args:
            dl: 用于DataFrame的数据加载器
            fit_func: 训练回调函数，需在里面完成fit()
                      参数: 训练集数据, 训练集标签, 验证集数据, 验证集标签
                      返回值: 无
            predict_func: 预测回调函数，需在里面完成predict()或者predict_proba()
                      参数: 测试集数据
                      返回值: 预测结果
            complete_func: 完成时回调函数，其中可包含任意操作
                      参数: 在所有测试集上的预测结果，包含两列['y_pred', 'y_true']
                      返回值: 无
            n_splits: 折数
            stratified: 是否分层(保持标签的分布相同)
            result_filepath: 预测结果保存的路径(折数将自动拼接在文件名后，后缀名前)
            result_folder_for_ensemble: 用于模型融合的预测结果保存目录
                                        训练集的预测结果将携带索引和真实结果保存成 X_result_{第N折}.pkl
                                        验证集的预测结果将携带索引和真实结果保存成 X_val_result_{第N折}.pkl
                                        测试集的预测结果将携带id保存成 X_test_result_{第N折}.pkl
            random_state: 随机状态，该值将被传入get_train_sample_indices_for_kfold()以保证K折结果确定化
        """
        self.log('开始训练({}折交叉验证)'.format(n_splits))
        X_indices_list, X_val_indices_list = dl.get_train_sample_indices_for_kfold(
            n_splits, stratified, random_state=random_state)
        labels = dl.get_train_labels()
        ids = dl.get_test_ids()

        # 预测
        y_preds_on_X_val = []

        for i, (X_indices, X_val_indices) in enumerate(zip(X_indices_list, X_val_indices_list)):
            self.log('执行第{}折'.format(i + 1))
            # 通过get_train_sample_indices_for_kfold()得到的索引都是基于0开始的[0, N-1]，真实索引可能乱序也不一定基于0
            # 所以需要通过以下代码获得真正的索引
            X_indices = dl.train_samples.index[X_indices]
            X_val_indices = dl.train_samples.index[X_val_indices]

            X = dl.train_samples.loc[X_indices].drop(columns=[dl.get_label_column()])
            y = labels.loc[X_indices]
            X_val = dl.train_samples.loc[X_val_indices].drop(columns=[dl.get_label_column()])
            y_val = labels.loc[X_val_indices]
            X_test = dl.test_samples

            # 训练
            fit_func(X, y, X_val, y_val)

            if result_filepath is not None:
                # 将第几折信息拼接在文件名最后，后缀名前
                path_root, ext = os.path.splitext(result_filepath)
                filepath = '{}_{}{}'.format(path_root, i + 1, ext)
                folder = os.path.dirname(filepath)
                if os.path.exists(folder):
                    y_pred_on_X_test = predict_func(X_test)
                    self.save_results(y_pred_on_X_test, ids, filepath)
                else:
                    self.log('预测结果无法生成，上层目录不`{}`存在'.format(folder))

            if complete_func is not None or result_folder_for_ensemble is not None:
                # 需要验证集预测结果
                y_pred_on_X_val = predict_func(X_val)
                df_val = pd.DataFrame({'y_pred': y_pred_on_X_val, 'y_true': y_val}, index=X_val.index)
                y_preds_on_X_val.append(df_val)

            if result_folder_for_ensemble is not None:
                os.makedirs(result_folder_for_ensemble, exist_ok=True)
                # 训练集预测结果
                y_pred_on_X = predict_func(X)
                df = pd.DataFrame({'y_pred': y_pred_on_X, 'y_true': y}, index=X.index)
                df.to_pickle(os.path.join(result_folder_for_ensemble, 'X_result_{}.pkl').format(i + 1),
                             protocol=pickle.DEFAULT_PROTOCOL)
                # 验证集预测结果
                df_val.to_pickle(os.path.join(result_folder_for_ensemble, 'X_val_result_{}.pkl').format(i + 1),
                                 protocol=pickle.DEFAULT_PROTOCOL)
                # 测试集预测结果
                y_pred_on_X_test = predict_func(X_test)
                df = pd.DataFrame(y_pred_on_X_test, columns=['y_pred'], index=ids)
                df.to_pickle(os.path.join(result_folder_for_ensemble, 'X_test_result_{}.pkl').format(i + 1),
                             protocol=pickle.DEFAULT_PROTOCOL)

        if complete_func is not None:
            complete_func(pd.concat(y_preds_on_X_val))

    def get_default_output_folder_for_ensemble(self, name):
        """获得默认的集成学习中间文件的输出目录
        Args:
            name: 最后一层目录名。方便区别通常指定为模型的名字: xgb, lgb等
        """
        return os.path.join(self.get_default_ensemble_root(), name)

    def get_default_ensemble_root(self):
        """获得默认的集成学习根目录"""
        return os.path.join(self.consts.output_folder, 'ensemble')

    def stacking(self,
                 targets: List,
                 fit_func: Callable[[DataFrame, Series], None],
                 predict_func: Callable[[DataFrame], List],
                 result_filepath: str,
                 ensemble_root_folder=None,
                 ) -> None:
        """Stacking融合
        Args:
            targets: 参与融合的第一层模型列表
            fit_func: 训练回调函数，需在里面完成fit()
                      参数: 训练集数据, 训练集标签
                      返回值: 无
            predict_func: 预测回调函数，需在里面完成predict()或者predict_proba()
                      参数: 测试集数据
                      返回值: 预测结果
            result_filepath: 预测结果保存的路径
            ensemble_root_folder: 用于模型融合的中间数据的根目录(不设置则将自动设置为get_default_ensemble_root()返回的目录)
        """
        if ensemble_root_folder is None:
            ensemble_root_folder = self.get_default_ensemble_root()

        # 融合对象的存在性检查
        dirs_not_exists = list(filter(lambda x: not os.path.isdir(os.path.join(ensemble_root_folder, x)), targets))
        if len(dirs_not_exists) > 0:
            self.log('以下对象集成学习的目录不存在，请先使用K折交叉验证方式创建')
            self.log('\t{}'.format(', '.join(dirs_not_exists)))
            return

        # 第二层训练集
        train_ds = []
        # 第二层测试集
        test_ds = []
        # 标签
        labels = None
        for target in targets:
            folder = os.path.join(ensemble_root_folder, target)
            self.log('读取`{}`中的数据'.format(folder))

            # 使用对验证集的预测结果拼接第二层的训练集
            df = pd.concat([pd.read_pickle(path) for path in glob.glob(os.path.join(folder, 'X_val_result_*.pkl'))])
            train_ds.append(pd.Series(df['y_pred'].values, index=df.index, name=target))
            if labels is None:
                labels = df['y_true']

            # 使用对测试集的预测结果就平均作为第二层的测试集
            results = [pd.read_pickle(path) for path in glob.glob(os.path.join(folder, 'X_test_result_*.pkl'))]
            ids = results[0].index
            test_ds.append(pd.Series(np.squeeze(np.mean(np.array(results), axis=0), axis=-1), index=ids, name=target))

        train_ds.append(labels)
        # 不同模型数据连接时，将根据索引自动对齐
        train_ds = pd.concat(train_ds, axis=1)
        test_ds = pd.concat(test_ds, axis=1)

        # 第二层训练
        fit_func(train_ds.drop(columns='y_true'), train_ds['y_true'])
        y_pred = predict_func(test_ds)
        self.save_results(y_pred, test_ds.index, result_filepath)

    @abc.abstractmethod
    def save_results(self, y_pred, ids: List, filepath: str) -> None:
        """保存预测结果
        Args:
            y_pred: 模型预测的输出结果
            ids: 测试集的标识
            filepath: 预测结果保存的路径
        """
        pass
