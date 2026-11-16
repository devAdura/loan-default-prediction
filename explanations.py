"""Positive Bad-class SHAP with explicit output scale and reconstruction."""
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from nigeria_features import DISPLAY_NAMES


def explain(model, frame):
    preprocessor = model.named_steps['preprocessor']
    classifier = model.named_steps['classifier']
    x = np.asarray(preprocessor.transform(frame))
    if isinstance(classifier, LogisticRegression):
        mean = model.shap_background_mean_
        explainer = shap.LinearExplainer(classifier, (mean, np.zeros((len(mean), len(mean)))))
        scale, dependence = 'log_odds', 'interventional training mean'
        expected = classifier.decision_function(x)
    elif isinstance(classifier, (RandomForestClassifier, GradientBoostingClassifier)):
        explainer = shap.TreeExplainer(classifier, feature_perturbation='tree_path_dependent', model_output='raw')
        scale = 'probability' if isinstance(classifier, RandomForestClassifier) else 'log_odds'
        dependence = 'tree_path_dependent, no separate background rows'
        expected = classifier.predict_proba(x)[:, 1] if scale == 'probability' else classifier.decision_function(x)
    else:
        raise TypeError('Unsupported classifier for SHAP')
    values = explainer.shap_values(x)
    if isinstance(values, list):
        values = values[1]
    values = np.asarray(values)
    baseline = np.asarray(explainer.expected_value).reshape(-1)
    if values.ndim == 3:
        values = values[:, :, 1]
    base = float(baseline[1] if baseline.size == 2 else baseline[0])
    reconstruction = base + values.sum(axis=1)
    if not np.allclose(reconstruction, expected, rtol=1e-5, atol=1e-6):
        raise ValueError('SHAP additive reconstruction failed')
    names = preprocessor.get_feature_names_out()
    grouped = pd.DataFrame(0.0, index=range(len(frame)), columns=list(DISPLAY_NAMES))
    for i, name in enumerate(names):
        name = name.split('__', 1)[-1]
        source = next((f for f in DISPLAY_NAMES if name == f or name.startswith(f+'_')), None)
        if source is None:
            raise ValueError(f'Unmapped encoded feature {name}')
        grouped[source] += values[:, i]
    return grouped, {'baseline': base, 'scale': scale, 'dependence': dependence, 'shap_version': shap.__version__, 'max_reconstruction_error': float(np.max(np.abs(reconstruction-expected))), 'explained_output': np.asarray(expected).tolist()}
