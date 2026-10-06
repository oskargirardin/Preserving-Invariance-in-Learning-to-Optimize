from argparse import ArgumentParser
import time
import cocoex  # experimentation module
import numpy as np
scipy = cocoex.utilities.forgiving_import('scipy')  # solvers to benchmark
# cma = cocoex.utilities.forgiving_import('cma')  # solvers to benchmark
import cma
import pandas as pd
from evosax_wrapper import fmin_evosax
from evosax.algorithms import LearnedES
from fun_transform import FunScaler

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
    'rosenbrock0': cma.ff.rosen0,
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
                xopt = fmin_evosax(
                    f=f,
                    x0=x0s[i] / scaler,
                    std_init=sigma0/scaler,
                    popsize=16,
                    max_evals=1e4 * dim,
                    ftarget=1e-8
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
