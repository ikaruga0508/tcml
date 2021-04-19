class KFoldCrossValidationConfig:
    def __init__(self, X_indices_list, X_val_indices_list):
        """初始化
        Args:
            X_indices_list: 训练集索引矩阵
            X_val_indices_list: 验证集索引矩阵
        """
        assert (len(X_indices_list) == len(X_val_indices_list))
        self.X_indices_list = X_indices_list
        self.X_val_indices_list = X_val_indices_list
        self.n_splits = len(X_indices_list)

    def get_indices(self, k):
        """获得当前折的训练集索引和验证集索引
        Args:
            k: 当前折

        Returns:
            (训练集索引, 验证集索引)
        """
        assert (k < self.n_splits)
        return self.X_indices_list[k], self.X_val_indices_list[k]
