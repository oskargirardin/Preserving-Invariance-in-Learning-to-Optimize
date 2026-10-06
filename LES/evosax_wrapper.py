__all__ = ['fmin_evosax']
from evosax.algorithms.distribution_based import LearnedES
import numpy as np
import jax
import jax.numpy as jnp
import evosax



def fmin_evosax(
    f,
    x0,
    Optimizer=LearnedES,
    std_init=1.0,
    max_evals=1000,
    ftarget=None,
    es_dict=None,
    popsize=16,
    callback=None,
    termination_callback=None,
    seed=None,
) -> tuple[np.ndarray, str]:
    if seed is None:
        seed = np.random.randint(0, int(1e6))
    if es_dict is None:
        es_dict = {}
    if termination_callback is None:
        def termination_callback(es, state):
            return False

    key = jax.random.key(seed)

    es = Optimizer(
        population_size=popsize,
        solution=jnp.asarray(x0),
        **es_dict,
    )
    # Load params
    if Optimizer == LearnedES:
        ckpt_fname = "2023_10_les_v2.pkl"
        path = f".venv/lib/python3.11/site-packages/evosax/algorithms/ckpt/les/{ckpt_fname}"
        les_params = evosax.learned_evolution.les_tools.load_pkl_object(path)
        params = evosax.algorithms.distribution_based.learned_es.Params(
            std_init=std_init,
            params=les_params
        )
    else:
        params = es.default_params

    stop = ''
    key, subkey = jax.random.split(key)
    state = es.init(subkey, x0, params)
    f_bsf = np.inf
    evals = 0
    while True:
        mean_old = state.mean.copy()
        key, subkey = jax.random.split(key)
        key_ask, key_eval, key_tell = jax.random.split(subkey, 3)

        population, state = es.ask(key_ask, state, params)
        if not jnp.all(jnp.isfinite(population)):
            stop = 'nan-population'
            break

        fitness = jnp.array([f(np.asarray(x)) for x in population])
        evals += len(population)
        f_min = jnp.min(fitness)
        if jnp.isfinite(f_min):
            f_bsf = min(f_bsf, f_min)
        else:
            stop = 'nan-fitness'
            break

        state, _ = es.tell(key_tell, population, fitness, state, params)

        if callback:
            callback(es, state)

        if not jnp.all(jnp.isfinite(state.mean)):
            stop = 'nan-mean'
            break

        if ftarget is not None and f_min < ftarget:
            stop = 'ftarget'
            break

        if evals > max_evals:
            stop = 'maxevals'
            break

        if termination_callback(es, state):
            stop = 'termination_callback'
            break

    xopt = mean_old if stop == 'nan-mean' else state.mean

    return np.asarray(xopt), stop

if __name__ == '__main__':
    def sphere(x):
        return np.sum(np.square(x)) - 1000
    res = fmin_evosax(sphere, [3] * 10, LearnedES, 2.0, max_evals=1000)
    print(res)

