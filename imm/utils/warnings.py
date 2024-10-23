import warnings
from functools import wraps


def suppress_warnings(warning_type=FutureWarning):
    def decorator(func):
        @wraps(func)
        def wrapped(*args, **kwargs):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", warning_type)
                return func(*args, **kwargs)

        return wrapped

    return decorator
