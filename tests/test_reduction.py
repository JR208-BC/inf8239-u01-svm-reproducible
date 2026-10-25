import numpy as np
import pandas as pd

from inf8239_u01.features import build_preprocessor, build_preprocessor_pca
from test_features import toy_frame
from inf8239_u01.features import split_features_target


def test_pca_block_has_fewer_or_equal_columns_than_dense():
    X, _ = split_features_target(toy_frame(80))
    dense = build_preprocessor(X, dense=True).fit_transform(X)
    reduced = build_preprocessor_pca(X).fit_transform(X)
    assert reduced.shape[1] <= dense.shape[1]


def test_pca_block_is_dense_array():
    X, _ = split_features_target(toy_frame(80))
    out = build_preprocessor_pca(X).fit_transform(X)
    assert isinstance(out, np.ndarray)


def test_pca_components_explain_requested_variance():
    X, _ = split_features_target(toy_frame(80))
    prep = build_preprocessor_pca(X, n_components=0.95).fit(X)
    pca = prep.named_transformers_["num"].named_steps["pca"]
    assert pca.explained_variance_ratio_.sum() >= 0.95 - 1e-6
