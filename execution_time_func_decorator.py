# from functools import wraps
from time import time

# https://dev.to/kcdchennai/python-decorator-to-measure-execution-time-54hk & ci

def timeit(func):
    # @wraps(func)
    def _timeit_wrapper(*args, **kwargs):
        start_time = time()
        result = func(*args, **kwargs)
        end_time = time()
        total_time = end_time - start_time
        print(
            # f'Function {func.__name__}{args} {kwargs} executed in {total_time:.4f} seconds')
            f'Function {func.__name__} executed in {total_time:.4f} seconds')
        return result
    return _timeit_wrapper


# test
@timeit
def _calculate_something(num):
    """
    Simple function that returns sum of all numbers up to the square of num.
    """
    total = sum((x for x in range(0, num**2)))
    return total


if __name__ == '__main__':
    _calculate_something(10)
    _calculate_something(100)
