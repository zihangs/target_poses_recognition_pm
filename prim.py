from ema_workbench import (RealParameter, IntegerParameter, CategoricalParameter, 
                           ScalarOutcome, Constant, save_results,
                           Model, ema_logging, optimize, perform_experiments)

from ema_workbench import MultiprocessingEvaluator
from ema_workbench import ema_logging

from pm_recognizer import run_system
import os
import sys


subject_id = int(sys.argv[1])
n_features = int(sys.argv[2])
n_clusters = int(sys.argv[3])


ema_logging.LOG_FORMAT = '[%(name)s/%(levelname)s/%(processName)s] %(message)s'
ema_logging.log_to_stderr(ema_logging.INFO)


model = Model('pmgr', function=run_system)

model.uncertainties = [IntegerParameter("phi", 0,100),
                        RealParameter("delta", 0, 5),
                        RealParameter("lamb", 1, 5),
                        RealParameter("theta", 0.6, 1.0)]


model.constants = [Constant("subject_id", subject_id),
                    Constant("n_features", n_features),
                    Constant("n_clusters", n_clusters)]


model.outcomes = [ScalarOutcome('p'),
                  ScalarOutcome('r'),
                  ScalarOutcome('f1'),
                  ScalarOutcome('bacc')]

# model.outcomes = [ScalarOutcome('f1', ScalarOutcome.MAXIMIZE)]
# ema_logging.log_to_stderr(ema_logging.INFO)
# results = optimize(model, nfe=3, searchover="levers", epsilons=[0.1] * len(model.outcomes))

results = perform_experiments(model, 1000)

# results.to_csv('f1_100_scenarios_%s_.tar.gz'%subject_id, index=False)
save_results(results, 'f1_1000_scenarios_sub%s_.tar.gz'%subject_id)

