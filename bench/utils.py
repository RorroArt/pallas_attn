""""
Benchmarking utils for pallas kernels and jax functions
"""

import jax 
import jax.numpy as jnp
import timeit
import numpy as np

def benchmark_fn(f, n_trials=100, n_warmup=1, print_fn=None):
    def run(*args, **kwargs):
        for _ in range(n_warmup):
            jax.block_until_ready(f(*args, **kwargs))
        
        result = timeit.timeit(
            lambda: jax.block_until_ready(f(*args, **kwargs)),
            number=n_trials
        ) 
        time = result / n_trials
        if print_fn is not None:
            print_fn(time)
        
        return time
    return run

def compare_fn(f1, f2, n_trials=100, n_warmup=1, print_fn=None):
    run_f1 = benchmark_fn(f1, n_trials, n_warmup)
    run_f2 = benchmark_fn(f2, n_trials, n_warmup)

    def run(*args, **kwargs):
        time_f1 = run_f1(*args, **kwargs); time_f2 = run_f2(*args, **kwargs)
        speed_up = time_f1 / time_f2

        if print_fn is not None:
            print_fn(time_f1, time_f2, speed_up)

        return speed_up, time_f1, time_f2
    
    return run

def compare_list(fns, n_trials=100, n_warmup=1, print_fn=None):
    run_fns = [benchmark_fn(fn, n_trials, n_warmup) for fn in fns]
    def run(*args, **kwargs):
        times = np.array([run_fn(*args, **kwargs) for run_fn in run_fns])
        sorted_indices = np.argsort(times)
        sorted_times = times[sorted_indices]

        if print_fn is not None:
            print_fn(times, sorted_indices, sorted_times)

        return times, sorted_indices, sorted_times
    return run


def compare_equiv(f1, f2, batch, atol=1e-6, print_fn=None):
    def run(x):
        f1_res = f1(x)
        f2_res = f2(x)

        return jnp.allclose(f1_res, f2_res, atol=atol)
    
    batch_results = jax.vmap(run)(batch)
    total = batch_results.shape[0]
    correct = batch_results.sum()
    accuracy = correct / total

    if print_fn is not None:
        print_fn(accuracy, batch_results)

    return accuracy, batch_results


        
        
