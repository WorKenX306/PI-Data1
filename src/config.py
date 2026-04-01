MODEL_PARAMS = {
    "rf": {
        "n_estimators": 200,
        "max_depth": 15,
        "min_samples_leaf": 5,
        "random_state": 42,
        "n_jobs": -1,
        "class_weight": "balanced"
    },
    "xgb": {
        "n_estimators": 300,
        "random_state": 42,
        "n_jobs": -1,
        "scale_pos_weight": 1,
        "eval_metric": "logloss",
        "verbosity": 0
    },
    "lr": {
        "max_iter": 1000,
        "random_state": 42,
        "class_weight": "balanced"
    },
    "et": {
        "n_estimators": 200,
        "max_depth": 15,
        "min_samples_leaf": 5,
        "random_state": 42,
        "n_jobs": -1,
        "class_weight": "balanced"
    },
    "mlp": {
        "hidden_layer_sizes": (128, 64),
        "activation": "relu",
        "solver": "adam",
        "max_iter": 300,
        "random_state": 42,
        "early_stopping": True,
        "validation_fraction": 0.1,
        "n_iter_no_change": 15,
        "verbose": False
    },
    "lgbm": {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": -1,
        "min_child_samples": 20,
        "scale_pos_weight": 1,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1
    }
}