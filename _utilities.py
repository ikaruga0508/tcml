import time

__all__ = [
    'def_log',
]


def def_log(*objs, **kwargs) -> None:
    """打印带时间戳的日志"""
    ts = time.asctime(time.localtime(time.time()))
    print('[{}]'.format(ts), *objs, **kwargs)
