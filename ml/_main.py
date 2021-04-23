import abc
import os.path

import pandas as pd
import pickle

from .._main import MainBase
from .._loaders import DataFrameLoaderBase
from typing import Callable, List
from pandas import DataFrame, Series

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
            else:
                self.log('预测结果无法生成，上层目录不`{}`存在'.format(folder))

        if complete_func is not None:
            complete_func()

    def train_predict_with_cv(
            self,
            dl: DataFrameLoaderBase,
            fit_func: Callable[[DataFrame, Series, DataFrame, Series], None],
            predict_func: Callable[[DataFrame], List],
            complete_func: Callable[[], None] = None,
            n_splits: int = 5,
            stratified: bool = True,
            result_filepath: str = None,
            result_folder_for_ensemble: str = None,
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
                      参数: 无
                      返回值: 无
            n_splits: 折数
            stratified: 是否分层(保持标签的分布相同)
            result_filepath: 预测结果保存的路径(中间需要有且仅有一个占位符，折数将填入其中)
            result_folder_for_ensemble: 用于模型融合的预测结果保存目录
                                        训练集的预测结果将携带索引和真实结果保存成 X_result_{第N折}.pkl
                                        验证集的预测结果将携带索引和真实结果保存成 X_val_result_{第N折}.pkl
                                        测试集的预测结果将携带id保存成 X_test_result_{第N折}.pkl
        """
        self.log('开始训练({}折交叉验证)'.format(n_splits))
        X_indices_list, X_val_indices_list = dl.get_train_sample_indices_for_kfold(n_splits, stratified)
        labels = dl.get_train_labels()
        ids = dl.get_test_ids()

        for i, (X_indices, X_val_indices) in enumerate(zip(X_indices_list, X_val_indices_list)):
            self.log('执行第{}折'.format(i + 1))
            X = dl.train_samples.loc[X_indices].drop(columns=[dl.get_label_column()])
            y = labels.loc[X_indices]
            X_val = dl.train_samples.loc[X_val_indices].drop(columns=[dl.get_label_column()])
            y_val = labels.loc[X_val_indices]
            X_test = dl.test_samples

            # 训练
            fit_func(X, y, X_val, y_val)

            # 预测
            y_pred_on_X_test = None

            if result_filepath is not None:
                filepath = result_filepath.format(i + 1)
                folder = os.path.dirname(filepath)
                if os.path.exists(folder):
                    y_pred_on_X_test = predict_func(X_test)
                    self.save_results(y_pred_on_X_test, ids, filepath)
                else:
                    self.log('预测结果无法生成，上层目录不`{}`存在'.format(folder))

            if result_folder_for_ensemble is not None:
                os.makedirs(result_folder_for_ensemble, exist_ok=True)
                # 训练集预测结果
                y_pred_on_X = predict_func(X)
                df = pd.DataFrame({dl.get_label_column(): y_pred_on_X, 'y_true': y}, index=X.index)
                df.to_pickle(os.path.join(result_folder_for_ensemble, 'X_result_{}.pkl').format(i + 1),
                             protocol=pickle.DEFAULT_PROTOCOL)
                # 验证集预测结果
                y_pred_on_X_val = predict_func(X_val)
                df = pd.DataFrame({dl.get_label_column(): y_pred_on_X_val, 'y_true': y_val}, index=X_val.index)
                df.to_pickle(os.path.join(result_folder_for_ensemble, 'X_val_result_{}.pkl').format(i + 1),
                             protocol=pickle.DEFAULT_PROTOCOL)
                # 测试集预测结果
                if y_pred_on_X_test is None:
                    y_pred_on_X_test = predict_func(X_test)

                df = pd.DataFrame(y_pred_on_X_test, columns=[dl.get_label_column()], index=ids)
                df.to_pickle(os.path.join(result_folder_for_ensemble, 'X_test_result_{}.pkl').format(i + 1),
                             protocol=pickle.DEFAULT_PROTOCOL)

        if complete_func is not None:
            complete_func()

    @abc.abstractmethod
    def save_results(self, y_pred, ids: List, filepath: str) -> None:
        """保存预测结果
        Args:
            y_pred: 模型预测的输出结果
            ids: 测试集的标识
            filepath: 预测结果保存的路径
        """
        pass
