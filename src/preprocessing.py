"""Leakage-safe preprocessing pipelines.

All transformers are returned *unfitted*; put them inside an sklearn
``Pipeline`` together with the model so that, in cross-validation, they are
refitted on the training folds only.
"""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.data import BANK_BINARY_DERIVED, BANK_CATEGORICAL, BANK_NUMERIC, LEAKY_FEATURE


def _numeric_cols(include_duration: bool):
    return BANK_NUMERIC + ([LEAKY_FEATURE] if include_duration else [])


def make_bank_preprocessor(kind: str = "linear", include_duration: bool = False):
    """Column transformer for the Bank Marketing features.

    kind = "linear"  -> median imputation + missing indicator + standard scaling
                        for numerics, one-hot for categoricals. For distance /
                        gradient based models (LogReg, kNN, linear SVM).
    kind = "tree"    -> median imputation + missing indicator, *no* scaling;
                        ordinal codes for categoricals (trees are invariant to
                        monotone transformations).
    kind = "onehot_only" -> no scaling, one-hot categoricals (e.g. for
                        Naive Bayes variants or ablations).

    Missing categoricals ("unknown" in the raw file) are imputed with an
    explicit "missing" level, i.e. missingness stays visible to the model.
    Categories never seen during fitting are mapped to all-zeros (one-hot) or
    to -1 (ordinal) instead of raising an error.
    """
    num_cols = _numeric_cols(include_duration)
    num_steps = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if kind == "linear":
        num_steps.append(("scale", StandardScaler()))

    if kind == "tree":
        cat_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    else:
        cat_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encode", cat_encoder),
    ])
    return ColumnTransformer(
        [
            ("num", Pipeline(num_steps), num_cols),
            ("bin", "passthrough", BANK_BINARY_DERIVED),
            ("cat", cat_pipe, BANK_CATEGORICAL),
        ],
        verbose_feature_names_out=False,
    )
