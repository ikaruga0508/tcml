import time
import pandas as pd
import numpy as np

__all__ = [
    'def_log',
    'reduce_memory',
]


def def_log(*objs, **kwargs) -> None:
    """打印带时间戳的日志"""
    ts = time.asctime(time.localtime(time.time()))
    print('[{}]'.format(ts), *objs, **kwargs)


def reduce_memory(df: pd.DataFrame, log=def_log, df_name : str = None):
    """减少DataFrame对象的使用内存量
    Args:
        df: DataFrame对象
        log: 日志对象
        df_name: DataFrame对象的名字

    Returns:
        处理过后的DataFrame对象
    """
    def col_astype(col_min, col_max, _df, col, new_type, is_float):
        if is_float:
            if np.finfo(new_type).min < col_min and col_max < np.finfo(new_type).max:
                _df[col] = _df[col].astype(new_type)
                return True
        else:
            if np.iinfo(new_type).min < col_min and col_max < np.iinfo(new_type).max:
                _df[col] = _df[col].astype(new_type)
                return True

        return False

    int_types = ['int16', 'int32', 'int64']
    float_types = ['float16', 'float32', 'float64']
    raw_mem_used = df.memory_usage().sum() / 1024**2
    for column in df.columns:
        column_type = df[column].dtype
        if column_type in int_types or column_type in float_types:
            _col_min, _col_max = df[column].min(), df[column].max()
            if pd.isnull(_col_min) or pd.isnull(_col_max):
                continue

            if column_type in int_types:
                col_astype(_col_min, _col_max, df, column, np.int8, False) or\
                col_astype(_col_min, _col_max, df, column, np.int16, False) or\
                col_astype(_col_min, _col_max, df, column, np.int32, False) or\
                col_astype(_col_min, _col_max, df, column, np.int64, False)
            elif column_type in float_types:
                col_astype(_col_min, _col_max, df, column, np.float16, True) or\
                col_astype(_col_min, _col_max, df, column, np.float32, True)

    new_mem_used = df.memory_usage().sum() / 1024**2
    log('{}的DataFrame对象内存从 {:5.2f} Mb 减少至 {:5.2f} Mb'.format(
        df_name if df_name is not None else '未知', raw_mem_used, new_mem_used))
    return df
