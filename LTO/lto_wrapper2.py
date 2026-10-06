from copy import deepcopy
from pathlib import Path

import numpy as np

from gps.algorithm.policy.tf_policy import TfPolicy
from gps.algorithm.policy_opt.lto_model import fully_connected_tf_network
from gps.agent.lto.agent_cmaes import AgentCMAES
from gps.proto.gps_pb2 import PAST_OBJ_VAL_DELTAS, CUR_PS, PAST_SIGMA, CUR_SIGMA
# from examples.10BBOB.hyperparams import agent


def fmin_lto(
    objective,
    x0,
    sigma0,
    policy_path,
    agent_hyperparams,
    es_opts=None,
    C0=None,
    max_generations=10000,
    network_config=None,
):
    """
    Run a saved LTO-CMA policy on a callable objective.

    agent_hyperparams must be the agent configuration associated
    with the policy checkpoint, including its sensor/observation layout.
    """
    x0 = np.asarray(x0, dtype=float)

    if x0.ndim != 1 or x0.size == 0:
        raise ValueError("x0 must be a nonempty one-dimensional vector")
    if not np.all(np.isfinite(x0)):
        raise ValueError("x0 must contain only finite values")
    if not np.isfinite(sigma0) or sigma0 <= 0:
        raise ValueError("sigma0 must be finite and positive")
    if not isinstance(max_generations, int) or max_generations < 1:
        raise ValueError("max_generations must be a positive integer")

    policy_path = Path(policy_path).expanduser().resolve()
    if not policy_path.is_file():
        raise FileNotFoundError(str(policy_path))

    config = deepcopy(agent_hyperparams)
    config["conditions"] = 1
    config["T"] = max_generations + 1
    config["init_sigma"] = float(sigma0)
    config["fcns"] = [
        {
            "fcn_obj": objective,
            "dim": int(x0.size),
            "init_loc": x0.copy(),
            "init_sigma": float(sigma0),
        }
    ]

    if network_config is None:
        network_config = {"dim_hidden": [50, 50]}

    policy = TfPolicy.load_policy(
        str(policy_path),
        fully_connected_tf_network,
        network_config=network_config,
    )

    agent = AgentCMAES(config)
    condition = 0
    world = agent._worlds[condition]

    world.reset_world()
    world.run(ltorun=False, es_opts=es_opts, C0=C0)

    # sample = agent._init_sample(world.get_state())
    noise = np.zeros(agent.dU, dtype=float)

    policy.reset()
    numerics_error = False


    es = world.es
    for t in range(max_generations):

        state = world.get_state()
        obs_t = np.concatenate(
            [
                np.asarray(state[key]).reshape(-1)
                for key in [
                    PAST_OBJ_VAL_DELTAS,
                    CUR_PS,
                    PAST_SIGMA,
                    CUR_SIGMA
                ]
            ]
        )
        
        # for keys, vals in state.items():
        #     if not np.all(np.isfinite(np.asarray(vals))):
        #         numerics_error = True
        #         print("Numerics")
        #         break
        # try:
        #     X_t = agent.get_vectorized_state(state, condition)
        # except AssertionError:
        #     pass
        # obs_t = sample.get_obs(t=t)

        if es.stop():
            break

        if not np.all(np.isfinite(obs_t)):
            numerics_error = True
            break

        # if not np.all(np.isfinite(X_t)):
        #     raise FloatingPointError("Nonfinite policy state")
        # if not np.all(np.isfinite(obs_t)):
        #     raise FloatingPointError("Nonfinite policy observation")

        ### CUSTOM ###
        es.adapt_sigma.delta = 1.0  # Delta is not used by LTO but causes overflow errors
        ### END CUSTOM ###

        action = np.asarray(
            policy.act(
                None,  # not used by policy
                obs_t,
                None,  # not used by policy
                noise,
                es,
                world.func_values,
            ),
            dtype=float,
        ).reshape(-1)[0]

        if action.size != agent.dU:
            raise ValueError(
                "Expected {} action values, received {}".format(
                    agent.dU, action.size
                )
            )
        if not np.all(np.isfinite(action)):
            numerics_error = True
            break
            # raise FloatingPointError("Nonfinite policy action")

        es = world.es

        world.run_next(action)


        # agent._set_sample(sample, world.get_state(), t)

    stop_conditions = dict(world.es.stop()) if not numerics_error else {'numerics_error': True}
    # print(stop_conditions)

    return np.asarray(world.es.best.x, dtype=float).copy(), stop_conditions


if __name__ == "__main__":
    import imp

    scaler = 1e2
    def objective(x):
        return np.sum(np.square(x * scaler))

    # Obtain this dictionary from the configuration used with your checkpoint.
    # Preserve sensor_dims, state_include, obs_include, history_len, popsize,
    # and any other fields required by Agent.
    hyperparams = imp.load_source("hyperparams", "examples/10BBOB/hyperparams_nodir.py")
    base_agent_config = hyperparams.agent
    # print(base_agent_config)
    # raise
    dim = 10

    x0 = np.ones(dim) / scaler
    sigma0 = 1e2 / scaler
    C0 = np.diag(1e6**( - np.arange(dim) / (dim - 1)))

    result = fmin_lto(
        objective=objective,
        x0=x0,
        sigma0=sigma0,
        C0=C0,
        policy_path="examples/10BBOB/data_files/policy_itr_14.pkl",
        agent_hyperparams=base_agent_config,
        max_generations=1000,
        network_config={"dim_hidden": [50, 50]},
    )
    print(result)
