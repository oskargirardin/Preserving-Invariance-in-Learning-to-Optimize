from argparse import ArgumentParser
import time
import cocoex  # experimentation module
import imp
import numpy as np
scipy = cocoex.utilities.forgiving_import('scipy')  # solvers to benchmark
# cma = cocoex.utilities.forgiving_import('cma')  # solvers to benchmark
import cma
import pandas as pd
from fun_transform import FunScaler
from lto_wrapper2 import fmin_lto

class FunLogger:
    def __init__(self, f):
        self._f = f
        self.f_data = []
        self.n_evals = 0

    def __call__(self, *args, **kwargs):
        f_val = self._f(*args, **kwargs)
        self.n_evals += 1
        self.f_data.append((self.n_evals, f_val))
        return f_val


FUNC_MAP = {
    'sphere': cma.ff.sphere,
    'elli': cma.ff.elli,
    'rosenbrock': cma.ff.rosen,
    # 'rosenbrock0': cma.ff.rosen0,
}



if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument("--fun", nargs="+")
    parser.add_argument('--nruns', type=int, default=1)
    parser.add_argument('--dim', type=int, default=10)
    parser.add_argument('--savepath', type=str)
    parser.add_argument('--silent', action='store_true')
    args = parser.parse_args()
    
    if not args.fun:
        raise Exception(f"Pass at least one argument to --fun. Available functions: {list(FUNC_MAP.keys())}")

    funcs = args.fun
    nruns = args.nruns
    dim = args.dim
    verbose = not args.silent
    save_path: str = args.savepath


    x0s = np.random.uniform(-5, 5, (nruns, dim))
    sigma0 = 3

    scaler_vals = np.logspace(-3, 3, num=7)
    # scaler_vals = [1.0]

    if verbose:
        print(f"Functions: {funcs}\nDimension: {dim}\nRuns: {nruns}")

    data = []
    for func in funcs:
        if verbose:
            print(f"Function: {func}")
        for scaler in scaler_vals:
            if verbose:
                print(f"scaler={scaler}: ", end='')
            for i in range(args.nruns):
                if verbose:
                    print('.', end='')
                f = FUNC_MAP.get(func)
                f = FunLogger(f)
                f = FunScaler(f, scaler=scaler)
                maxiter = 10_000

                es_opts = {
                    'maxfevals': maxiter * dim,
                    'ftarget': 1e-8,
                    'verbose': -9
                }
                hyperparams = imp.load_source('hyperparams', 'examples/10BBOB/hyperparams_nodir.py')
                base_agent_config = hyperparams.agent

                xopt, stop = fmin_lto(
                    objective=f,
                    x0=x0s[i] / scaler,
                    sigma0=sigma0 / scaler,
                    es_opts=es_opts,
                    policy_path="examples/10BBOB/data_files/policy_itr_14.pkl",
                    agent_hyperparams=base_agent_config,
                    max_generations=maxiter,
                    network_config={"dim_hidden": [50, 50]},
                )
                data.append({
                    'func': func,
                    'scaler': scaler,
                    'dim': dim,
                    'run_idx': i,
                    'f_data': f.f_data,
                })
            if verbose:
                print('')
        
    if save_path:
        if not save_path.endswith('.pkl'):
            save_path += '.pkl'
        df = pd.DataFrame(data)
        df.to_pickle(save_path)
