"""krisha_ml: shared code for the Final (reproduction of the submitted Endterm + CV harness).

The API every track uses: core (data, folds, base model, CV harness), dedup/cv_schemes (leakage), geo, photos, text, trust, fusion.
"""
from .core import (BOOL, CAT, ENDTERM_REFERENCE, FEATURES, HGB_PARAMS, NUM, SEED, CVResult, Dataset,
                   base_cv, base_residual_inputs, cv_evaluate, cv_evaluate_residual, cv_folds, encode_features,
                   join_features, load_clean, load_train, load_train_test, make_base_model, make_preprocessor,
                   metrics, oof_log_pred)

__all__ = ["BOOL", "CAT", "ENDTERM_REFERENCE", "FEATURES", "HGB_PARAMS", "NUM", "SEED", "CVResult", "Dataset",
           "base_cv", "base_residual_inputs", "cv_evaluate", "cv_evaluate_residual", "cv_folds", "encode_features",
           "join_features", "load_clean", "load_train", "load_train_test", "make_base_model", "make_preprocessor",
           "metrics", "oof_log_pred"]
