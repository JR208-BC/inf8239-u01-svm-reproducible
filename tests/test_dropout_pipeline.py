import numpy as np

from inf8239_u01.evaluation import summarize
from inf8239_u01.features import split_features_target
from inf8239_u01.models import build_dropout_baseline, build_dropout_svm
from test_features import toy_frame


def test_svm_pipeline_predicts_one_label_per_row():
    X, y = split_features_target(toy_frame(60))
    model = build_dropout_svm(X).fit(X.iloc[:45], y.iloc[:45])
    assert len(model.predict(X.iloc[45:])) == 15


def test_baseline_predicts_majority_class():
    X, y = split_features_target(toy_frame(60))
    model = build_dropout_baseline(X).fit(X, y)
    assert set(model.predict(X)) == {y.value_counts().idxmax()}


def test_summarize_returns_expected_metrics():
    y = np.array(["Dropout", "Graduate", "Enrolled", "Dropout"])
    result = summarize(y, y)
    assert result["f1_macro"] == 1.0 and result["recall_dropout"] == 1.0
