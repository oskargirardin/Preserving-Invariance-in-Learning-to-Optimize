#!/usr/bin/env python
"""A short yet complete example experiment script with restarts and batching.

Arguments
---------
This script must be called with 1-3 arguments:

    budget_multiplier [number_of_batches batch_to_execute]

``budget_multiplier`` times dimension is the budget within which problem
instances are run repeatedly as long as too few successes are observed.

``batch_to_execute`` can only be omitted when ``number_of_batches == 1``.
When ``number_of_batches > 1``, the script must be executed repeatedly
(e.g., in parallel) with the different values for ``batch_to_execute =
0..number_of_batches-1`` to get data for the full experiment.

Usage
-----
To apply the code to a different solver/algorithm, `fmin` must be
re-assigned or re-defined accordingly, and the below code must be edited at
the two places marked with "### input" around lines 40 and 80.

See also: https://coco-platform.org/getting-started#experiment
"""

__author__ = "Nikolaus Hansen"
__copyright__ = "public domain"

# import mkl_bugfix  # set *_NUM_THREADS=1, requires mkl_bugfix.py file from build/python/example
import collections
import time
import cocoex  # experimentation module
scipy = cocoex.utilities.forgiving_import('scipy')  # solvers to benchmark
cma = cocoex.utilities.forgiving_import('cma')  # solvers to benchmark
import numpy as np
from fun_transform import FunTransform
import lto_wrapper

### input: define suite and solver (see also "input" below where fmin is called)
suite_name = "bbob"  # filter for preliminary quick tests:
suite_filter = "dimensions: 10"    # "dimensions: 2,3,5,10,20 instance_indices:1-5"
# fmin = scipy.optimize.fmin  # optimizer to be benchmarked
# fmin = scipy.optimize.fmin_slsqp
# fmin = cocoex.solvers.random_search
# fmin = cma.fmin2
# fmin = cma.fmin_lq_surr2
# fmin = evosax_wrapper.fmin_evosax
fmin = lto_wrapper.fmin_lto
TRANSFORM = True
exp_name = ''

# Transform parameters
axis_ratio = 1e2
scaler = 1e-2
expo = 0.5
rotation = False
lbound_translation = -5
ubound_translation = 5

### reading in parameters
if __name__ == '__main__':
    import sys
    try:
        budget_multiplier = float(sys.argv[1])
        number_of_batches = int(sys.argv[2]) if len(sys.argv) > 2 else 1
        batch_to_execute = int(sys.argv[3]) if len(sys.argv) > 3 else None
    except Exception as e:
        print("Exception {} with calling arguments {}\n\n".format(e, sys.argv)
              + __doc__)
        raise

### prepare
suite = cocoex.Suite(suite_name, "", suite_filter)  # see https://numbbo.github.io/coco-doc/C/#suite-parameters
output_folder = '{}_of_{}_{}D_on_{}'.format(fmin.__name__, fmin.__module__ or '', int(budget_multiplier+0.499), suite_name)
if exp_name:
    output_folder = exp_name + '/' + output_folder
if TRANSFORM: 
    output_folder += f"_lgaxisratio{np.log10(axis_ratio):.0f}_lgscaler{np.log10(scaler):.0f}_expo{expo}_rotation{int(rotation)}_lb{lbound_translation}_ub{ubound_translation}"
else:
    output_folder += "_NoTransform"
if number_of_batches > 1:
    output_folder += ('_batch{:0' + str(len(str(number_of_batches-1))) + '}of{}').format(batch_to_execute, number_of_batches)
observer = cocoex.Observer(suite_name,
            # see https://numbbo.github.io/coco-doc/C/#observer-parameters
            'result_folder: {0}  algorithm_name: {1}'.format(
                output_folder, fmin.__module__ + '.' + fmin.__name__))
repeater = cocoex.ExperimentRepeater(budget_multiplier,  # x dimension
                                     min_successes=0.75 * int(suite_filter.split('-')[1])
                                         if "indices:1-" in suite_filter else 11,
                                     )  # possible sweeps over the suite
batcher = cocoex.BatchScheduler(number_of_batches, batch_to_execute)
minimal_print = cocoex.utilities.MiniPrint()
timings = collections.defaultdict(list)  # key is the dimension
final_conditions = collections.defaultdict(list)  # key is (id_fun, dimension, id_inst)
cocoex.utilities.write_setting(locals(), [observer.result_folder, 'parameters.pydat'])

### go
time0 = time.time()
while not repeater.done():  # while budget is left and successes are few
    for problem in suite:  # loop takes 2-3 minutes x budget_multiplier
        if not batcher.is_in_batch(problem) or repeater.done(problem):
            continue  # skip problem and bypass repeater.track
        problem.observe_with(observer)  # generate data for cocopp

        if TRANSFORM:
            seed = int(np.array([1, 1e2, 1e4]).dot(problem.id_triple))
            np.random.seed(seed)
            scaling_vec = axis_ratio**(np.arange(problem.dimension) / (problem.dimension - 1))
            x_offset = np.random.uniform(lbound_translation, ubound_translation, problem.dimension)
            problem = FunTransform(
                problem,
                scaler=scaler,
                expo=expo,
                scaling_vec=scaling_vec,
                x_offset=x_offset,
                rotation=rotation,
                rotation_seed=seed
            )
        else:
            problem = FunTransform(problem)


        time1 = time.time()


        problem(problem.transform_inverse(problem.dimension * [0]))  # for better comparability


        ### input: implement/amend the next few lines for another fmin
        # if fmin == env_wrapper.fmin:
        #     sigma0_base = 2
        #     x0 = problem.transform_inverse(problem.initial_solution_proposal())
        #     sigma0 = sigma0_base / problem.scaler
        #     xopt, env = fmin(
        #         problem,
        #         x0,
        #         sigma0,
        #         model_path=model_path,
        #         C0=problem.get_isotropic_cov_matrix(problem.dimension),
        #         es_opts={
        #             "tolstagnation": 0,
        #             "tolxstagnation": 0,
        #             "tolx": 0,
        #             "conditioncov_alleviate": 0,
        #             "mean_shift_line_samples": 1,
        #             "CSA_invariant_path": True,
        #             "tolfacupx": 1e6,
        #             "tolfun": 0,
        #             "tolfunhist": 0,
        #             "CMA_on": 1,
        #             "verbose": -9,
        #         },
        #         es_cls_kwargs={
        #             'initial_lr_mean_generator': None,
        #             'initial_damping_generator': None
        #         }
        #     )
        #     final_condition = env.es.stop()

        if fmin == cma.fmin2:
            if fmin == cma.fmin_lq_surr2 and problem.dimension == 40:
                continue  # takes prohibitively long!?
            options = {'maxfevals': 8 * problem.dimension * budget_multiplier,
                       'termination_callback': lambda es: problem.final_target_hit,
                       'conditioncov_alleviate': 2 * [False] if fmin == cma.fmin_lq_surr2 else None,
                       'verbose': -9 }
            xopt, es = fmin(problem, problem.initial_solution_proposal, 2,
                            options, restarts={'maxfevals': problem.dimension * budget_multiplier})
            final_condition = es.stop()
        elif fmin == lto_wrapper.fmin_lto:
            sigma0_base = 2
            x0 = problem.transform_inverse(problem.initial_solution_proposal())
            sigma0 = sigma0_base / problem.scaler
            options = {'maxfevals': problem.dimension * budget_multiplier,
                       'termination_callback': lambda es: problem.final_target_hit,
                       'conditioncov_alleviate': None,
                       'verbose': -9 }
            xopt, final_condition = fmin(
                problem,
                x0=x0,
                sigma0=sigma0,
                es_opts=options
            )
        else:
            raise ValueError('case for fmin={} not found'.format(fmin))

        problem(problem.transform_inverse(xopt))  # make sure the returned solution is evaluated

        if repeater._sweeps == 1:  # time only the first (full) sweep through suite
            timings[problem.dimension].append((time.time() - time1) / problem.evaluations)
        repeater.track(problem)  # track evaluations and final_target_hit
        minimal_print(problem)  # show progress
        final_conditions[problem.id_triple].append(repr([problem.evaluations, final_condition]))
        with open(observer.result_folder + '/final_conditions.pydict', 'wt') as file_:
            file_.write(str(dict(final_conditions)).replace('],', '],\n'))

### final messaging
print("\nTiming summary over all functions without repetitions:\n"
      "  dimension  median time [seconds/evaluation]\n"
      "  -------------------------------------")
for dimension in sorted(timings):
    ts = sorted(timings[dimension])
    print("    {:3}       {:.1e}".format(dimension, (ts[len(ts)//2] + ts[-1-len(ts)//2]) / 2))
print("  -------------------------------------")

if number_of_batches > 1:
    print("\n*** Batch {} of {} batches finished in {}."
          " Make sure to run *all* batches (0..{}) ***".format(
          batch_to_execute, number_of_batches,
          cocoex.utilities.ascetime(time.time() - time0), number_of_batches - 1))
else:
    print("\n*** Full experiment done in %s ***"
          % cocoex.utilities.ascetime(time.time() - time0))
print("    Data written into {}".format(observer.result_folder))

### post-process data
if number_of_batches == 1:
    print("    Postprocess with 'python cocopp {} [...]'".format(observer.result_folder))
    import cocopp  # post-processing module
    dsl = cocopp.main(observer.result_folder)  # re-run folders look like "...-001" etc
