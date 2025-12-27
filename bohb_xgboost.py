import os
import numpy as np
import xgboost as xgb
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from hpbandster.core.worker import Worker
import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
from hpbandster.optimizers import BOHB
import ConfigSpace as CS
import logging
logging.basicConfig(level=logging.WARNING)

# Generate synthetic data (replace with your real data)
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, random_state=42)
dtrain = xgb.DMatrix(X, label=y)

class XGBoostWorker(Worker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute(self, config, budget, **kwargs):
        params = {
            'max_depth': int(config['max_depth']),
            'learning_rate': config['learning_rate'],
            'min_child_weight': config['min_child_weight'],
            'subsample': config['subsample'],
            'colsample_bytree': config['colsample_bytree'],
            'gamma': config['gamma'],
            'reg_alpha': config['reg_alpha'],
            'reg_lambda': config['reg_lambda'],
            'objective': 'binary:logistic',
            'eval_metric': 'error',
            'tree_method': 'hist',
            'seed': 42
        }
        
        # Train with early stopping
        cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=1000,
            nfold=3,
            stratified=True,
            early_stopping_rounds=30,
            maximize=False,
            verbose_eval=False,
            as_pandas=False
        )
        
        best_score = cv_results['test-error-mean'][-1]
        return {
            'loss': best_score,
            'info': {'num_rounds': len(cv_results['test-error-mean'])}
        }

    @staticmethod
    def get_configspace():
        cs = CS.ConfigurationSpace()
        cs.add_hyperparameter(CS.UniformIntegerHyperparameter('max_depth', lower=3, upper=15))
        cs.add_hyperparameter(CS.UniformFloatHyperparameter('learning_rate', lower=0.01, upper=0.3, log=True))
        cs.add_hyperparameter(CS.UniformFloatHyperparameter('min_child_weight', lower=1, upper=10))
        cs.add_hyperparameter(CS.UniformFloatHyperparameter('subsample', lower=0.5, upper=1.0))
        cs.add_hyperparameter(CS.UniformFloatHyperparameter('colsample_bytree', lower=0.5, upper=1.0))
        cs.add_hyperparameter(CS.UniformFloatHyperparameter('gamma', lower=0, upper=5))
        cs.add_hyperparameter(CS.UniformFloatHyperparameter('reg_alpha', lower=0, upper=1))
        cs.add_hyperparameter(CS.UniformFloatHyperparameter('reg_lambda', lower=0, upper=1))
        return cs

# Run BOHB
print("Starting BOHB optimization...")
ns = hpns.NameServer(run_id='xgboost_bohb', host='127.0.0.1', port=0)
ns.start()

worker = XGBoostWorker(run_id='xgboost_bohb', host='127.0.0.1', nameserver=ns.host, nameserver_port=ns.port)
worker.start()

bohb = BOHB(configspace=worker.get_configspace(), run_id='xgboost_bohb', nameserver=ns.host, nameserver_port=ns.port, min_budget=10, max_budget=100)
res = bohb.run(n_iterations=20)

bohb.shutdown(shutdown_workers=True)
ns.shutdown()

id2config = res.get_id2config_mapping()
incumbent = res.get_incumbent_id()
best_config = id2config[incumbent]['config']

print("\n=== BEST HYPERPARAMETERS FOUND ===")
for k, v in best_config.items():
    print(f"{k}: {v}")

print(f"\nBest validation error: {res.get_incumbent_trajectory()['losses'][-1]:.4f}")
print("BOHB optimization complete!")
