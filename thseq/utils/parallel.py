from multiprocessing.pool import ThreadPool
import multiprocessing

def parallel_map(fn, input, num_thread):
    if num_thread < 1:
        raise RuntimeError(f'Expected num_thread >= 1. Got {num_thread}.')
    if num_thread == 1:
        return list(map(fn, input))
    pool = ThreadPool(num_thread)
    results = pool.map(fn, input)
    pool.close()
    pool.join()
    return results

def parallel_map2(fn, input, num_thread):
    if num_thread < 1:
        raise RuntimeError(f'Expected num_thread >= 1. Got {num_thread}.')
    if num_thread == 1:
        return list(map(fn, input))
    pool = multiprocessing.Pool(num_thread)
    results = pool.map(fn, input)
    pool.close()
    pool.join()
    return results
