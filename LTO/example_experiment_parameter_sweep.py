#!/usr/bin/env python
"""An example experiment script with restarts and batching sweeping over a parameter.

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

import mkl_bugfix  # set *_NUM_THREADS=1, requires mkl_bugfix.py file from build/python/example
import collections
import time
import cocoex  # experimentation module
import numpy as np
scipy = cocoex.utilities.forgiving_import('scipy')  # solvers to benchmark
cma = cocoex.utilities.forgiving_import('cma')  # solvers to benchmark
# from evosax_wrapper import fmin_evosax
# from evosax.algorithms import LearnedES, Sep_CMA_ES, DiscoveredES, CMA_ES, Open_ES
from fun_scaler import FunScaler
from lto_wrapper import fmin_lto


### input: define suite and solver (see also "input" below where fmin is called)
suite_name = "bbob"  # filter for preliminary quick tests:
suite_filter = "dimensions: 2,3,5,10,20"
# fmin = scipy.optimize.fmin  # optimizer to be benchmarked
# fmin = scipy.optimize.fmin_slsqp
# fmin = cocoex.solvers.random_search
# fmin = cma.fmin2
# fmin = cma.fmin_lq_surr2
# fmin = LearnedES
fmin = fmin_lto

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

# for expo in [0.25, 0.5, 1.0, 2.0, 3.0]:
for scaler in np.logspace(-3, 3, 7):
    print('scaler = {0}'.format(scaler))
    # output_folder = '{}e{}_of_{}_{}D_on_{}{}'.format(
    output_folder = '{}s{}_of_{}_{}D_on_{}{}'.format(
            # float(expo),  # folder name must start with a number for cocopp
            float(scaler),  # folder name must start with a number for cocopp
            fmin.__name__, fmin.__module__ or '', int(budget_multiplier+0.499), suite_name,
            ('_batch{:0' + str(len(str(number_of_batches-1))) + '}of{}').format(
                batch_to_execute, number_of_batches) if number_of_batches > 1 else '')
    observer = cocoex.Observer(suite_name,
            # see https://numbbo.github.io/coco-doc/C/#observer-parameters
            'result_folder: {0}  algorithm_name: {1}  settings: "{2}"'
                .format(output_folder,
                        fmin.__module__ + '.' + fmin.__name__,
                        # "expo" + str(expo)))
                        "scaler" + str(scaler)))
    repeater = cocoex.ExperimentRepeater(budget_multiplier,  # x dimension
                                         min_successes=0.75 * int(suite_filter.split('-')[1])
                                            if "indices:1-" in suite_filter else 11,
                                         max_sweeps=100)  # possible sweeps over the suite
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

            expo = 1.0
            problem = FunScaler(problem, expo=expo, scaler=scaler)
            x0 = repeater.initial_solution_proposal(problem) / scaler
            sigma0 = 2.0 / scaler

            time1 = time.time()
            problem(problem.dimension * [0])  # for better comparability

            ### input: implement/amend the next few lines for another fmin
            if fmin == cma.fmin2:
                options = {'maxfevals': problem.dimension * budget_multiplier,
                           'termination_callback': lambda es: problem.final_target_hit,
                           'conditioncov_alleviate': None,
                           'verbose': -9 }
                xopt, es = fmin(problem, x0, sigma0, options,
                                restarts=False
                            )
                final_condition = es.stop()
            elif fmin == fmin_lto:
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

            problem(xopt)  # make sure the returned solution is evaluated

            if repeater._sweeps == 1:  # time only the first (full) sweep through suite
                timings[problem.dimension].append((time.time() - time1) / problem.evaluations)
            repeater.track(problem)  # track evaluations and final_target_hit
            minimal_print(problem)  # show progress
            final_conditions[problem.id_triple].append(repr([problem.evaluations, final_condition]))
            with open(observer.result_folder + '/final_conditions.pydict', 'wt') as file_:
                file_.write(str(dict(final_conditions)).replace('],', '],\n'))

### final messaging
print("\nTiming summary of last experiment without repetitions:\n"
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
    import glob
    try:
        import cocopp  # post-processing module
    except ImportError:
        print("Install postprocessing like: pip install --pre cocopp\n"
              "and call\n\n  python -m cocopp --parameter-sweep exdata/0*")
    else:
        print("    Postprocessing with 'python cocopp exdata/0*'")
        dsl = cocopp.main('--parameter-sweep ' + ' '.join(glob.glob('exdata/0*')))  # re-run folders look like "...-001" etc
        if not dsl or isinstance(dsl, int):
            print("Uninstall cocopp and install postprocessing like: \n\n"
                  "  pip uninstall cocopp\n"
                  "  pip install --pre cocopp")
