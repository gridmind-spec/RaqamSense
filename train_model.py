"""Reproducible training: scikit-learn Digits → cleaning → validation → test."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import joblib
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn import __version__ as sklearn_version

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / 'model' / 'digit_tree.joblib'
METRICS_PATH = ROOT / 'reports' / 'metrics.json'


def build_data():
    d = load_digits()
    cols = [f'pixel_{i:02d}' for i in range(64)]
    df = pd.DataFrame(d.data, columns=cols)
    df['digit'] = d.target.astype(int)
    original_count = len(df)
    missing_before = int(df.isna().sum().sum())
    duplicates_before = int(df.duplicated().sum())
    # De-duplicate complete labelled rows only; record exact removals.
    df = df.drop_duplicates().reset_index(drop=True)
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    missing_after_conversion = int(df.isna().sum().sum())
    # Median fill is fit only on training in larger production pipelines;
    # here the digits source has no missing values. Fail if that changes.
    if missing_after_conversion:
        raise ValueError('Unexpected missing pixel values; revise training-only imputation.')
    if not ((df[cols] >= 0) & (df[cols] <= 16)).all().all():
        raise ValueError('Pixel values must lie between 0 and 16.')
    clean_path = ROOT / 'cleaned_data.csv'
    df.to_csv(clean_path, index=False)
    cleaning = {
        'source': 'scikit-learn load_digits (8x8 handwritten digits)',
        'source_url': 'https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html',
        'original_rows': original_count,
        'duplicate_rows_removed': duplicates_before,
        'missing_values_before': missing_before,
        'missing_values_after': int(df.isna().sum().sum()),
        'clean_rows': len(df),
        'features': len(cols),
        'classes': sorted(df['digit'].unique().tolist()),
    }
    (ROOT / 'reports' / 'cleaning_report.json').write_text(json.dumps(cleaning, ensure_ascii=False, indent=2), encoding='utf-8')
    return df, cols, cleaning


def save_examples(X_test, y_test):
    for digit in range(10):
        idx = np.flatnonzero(y_test.to_numpy() == digit)[0]
        px = X_test.iloc[idx].to_numpy().reshape(8, 8)
        img = Image.fromarray(np.uint8(np.clip(px / 16 * 255, 0, 255)), 'L')
        img.resize((240, 240), Image.Resampling.NEAREST).save(ROOT / 'app_assets' / 'samples' / f'digit_{digit}.png')


def train_and_export():
    (ROOT / 'model').mkdir(exist_ok=True, parents=True)
    (ROOT / 'reports').mkdir(exist_ok=True, parents=True)
    (ROOT / 'app_assets' / 'samples').mkdir(exist_ok=True, parents=True)
    df, cols, cleaning = build_data()
    X = df[cols]
    y = df['digit']

    # Keep final test set untouched during depth selection.
    X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_dev, y_dev, test_size=0.25, stratify=y_dev, random_state=42)

    tuning = []
    for depth in (3, 5, 7, 9, 12, None):
        model = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=2, random_state=42)
        model.fit(X_train, y_train)
        val_accuracy = float(accuracy_score(y_val, model.predict(X_val)))
        tuning.append({'max_depth': depth, 'val_accuracy': round(val_accuracy, 4)})
    best = max(tuning, key=lambda p: p['val_accuracy'])  # prefers smaller on exact ties
    # Refit using all development data after validation-only parameter selection.
    final_model = DecisionTreeClassifier(max_depth=best['max_depth'], min_samples_leaf=2, random_state=42)
    final_model.fit(X_dev, y_dev)
    predictions = final_model.predict(X_test)
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    conf = confusion_matrix(y_test, predictions, labels=list(range(10)))
    baseline = float(y_test.value_counts(normalize=True).max())
    metrics = {
      'problem': 'Classification of handwritten digits from 8x8 grayscale images',
      'train_samples_for_tuning': len(X_train),
      'validation_samples': len(X_val),
      'development_samples_for_final_fit': len(X_dev),
      'test_samples': len(X_test),
      'test_accuracy': round(float(accuracy_score(y_test, predictions)), 4),
      'majority_baseline_accuracy': round(baseline, 4),
      'selected_max_depth': best['max_depth'],
      'min_samples_leaf': 2,
      'validation_accuracy_at_selection': best['val_accuracy'],
      'tuning': tuning,
      'per_class': {str(i): {'precision':round(report[str(i)]['precision'],4), 'recall':round(report[str(i)]['recall'],4), 'support':int(report[str(i)]['support'])} for i in range(10)},
      'confusion_matrix': conf.tolist(),
      'random_state': 42,
      'library': 'scikit-learn',
      'sklearn_version': sklearn_version,
      'caveats': ['Digits dataset consists of small 8×8 images.', 'Hand-drawn or smartphone uploaded images may differ from training distribution.', 'Decision tree leaf frequencies are not calibrated confidence probabilities.'],
    }
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({'model': final_model, 'columns':cols, 'sklearn_version':sklearn_version}, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    pd.DataFrame(tuning).to_csv(ROOT / 'reports' / 'hyperparameter_search.csv', index=False)
    pd.DataFrame(conf, index=[f'actual_{i}' for i in range(10)], columns=[f'predicted_{i}' for i in range(10)]).to_csv(ROOT / 'reports' / 'confusion_matrix.csv')
    pd.DataFrame({'actual':y_test.to_numpy(), 'predicted':predictions}).to_csv(ROOT / 'reports' / 'test_predictions.csv', index=False)
    save_examples(X_test, y_test)
    print('Test accuracy:', metrics['test_accuracy'], '| Best depth:',best['max_depth'], '| Rows:',cleaning['clean_rows'])
    return final_model, metrics

if __name__=='__main__':
    train_and_export()
