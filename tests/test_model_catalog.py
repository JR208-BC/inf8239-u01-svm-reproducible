from inf8239_u01.features import split_features_target
from inf8239_u01.models import build_dropout_svm_pca, build_model_catalog
from test_features import toy_frame


def test_catalog_has_six_configurations():
    X, _ = split_features_target(toy_frame(60))
    catalog = build_model_catalog(X)
    assert set(catalog) == {"logistic", "svm_c1", "svm_c10", "rf_100", "rf_300", "boost"}


def test_catalog_models_fit_and_predict_on_same_split():
    X, y = split_features_target(toy_frame(60))
    Xtr, ytr, Xte = X.iloc[:45], y.iloc[:45], X.iloc[45:]
    for name, model in build_model_catalog(X).items():
        model.fit(Xtr, ytr)
        assert len(model.predict(Xte)) == len(Xte), f"{name} no predice una fila por caso"


def test_reduction_variant_fits_and_predicts():
    X, y = split_features_target(toy_frame(60))
    model = build_dropout_svm_pca(X).fit(X.iloc[:45], y.iloc[:45])
    assert len(model.predict(X.iloc[45:])) == 15


def test_reduction_variant_rejects_non_positive_c():
    X, _ = split_features_target(toy_frame(60))
    try:
        build_dropout_svm_pca(X, C=0)
        assert False, "debía lanzar ValueError"
    except ValueError:
        pass
