# Breast Cancer Classifier

A beginner-friendly machine learning project that predicts whether a breast tumour is **malignant** or **benign** from 30 measurements of cell nuclei, using scikit-learn.

It covers the core supervised-learning workflow: exploring the data, preprocessing, training two models, cross-validation, hyperparameter tuning and evaluation.

## Dataset

The [Wisconsin Breast Cancer dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html) that ships with scikit-learn (569 samples, 30 numeric features, no missing values). No download is needed. Classes are slightly imbalanced: 357 benign and 212 malignant.

## Approach

1. **Explore** the data: class balance and a correlation heatmap.
2. **Split** 80/20 into train and test sets, stratified by class.
3. **Preprocess** with `StandardScaler` inside a `Pipeline`, so the scaler is fitted only on training data and nothing leaks into the test set.
4. **Train** Logistic Regression and Random Forest.
5. **Tune** hyperparameters with `GridSearchCV` (5-fold cross-validation, scored on F1).
6. **Evaluate** on the held-out test set with accuracy, precision, recall, F1, confusion matrices and ROC curves.

## Results

Test set of 114 samples. Precision, recall and F1 treat **benign as the positive class**.

| Model | Best params | CV F1 | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| Logistic Regression | `C=0.1` | 0.985 | 0.974 | 0.973 | 0.986 | 0.979 |
| Random Forest | `n_estimators=300`, `max_depth=None` | 0.967 | 0.947 | 0.958 | 0.958 | 0.958 |

Logistic Regression came out slightly ahead here. With only 114 test samples, a gap of one or two predictions is within normal variation, so I would not claim either model is clearly better.

Plots are saved in [`outputs/`](outputs): class distribution, correlation heatmap, confusion matrices, ROC curves and Random Forest feature importances. Full numbers are in `outputs/metrics.json`.

## Limitations

- This is a learning project, not a medical tool. It must not be used for real diagnosis.
- In a screening setting, missing a malignant tumour is much worse than a false alarm, so recall on the malignant class would matter most. This project reports benign-class metrics for simplicity.
- A single train/test split on a small dataset gives noisy estimates. Repeated or nested cross-validation would be more reliable.

## Run it

```bash
git clone <your-repo-url>
cd breast-cancer-classifier
pip install -r requirements.txt
python src/train.py
```

## Project structure

```
breast-cancer-classifier/
├── src/train.py        # full pipeline: explore, train, tune, evaluate
├── outputs/            # generated plots and metrics.json
├── requirements.txt
└── README.md
```

## Possible next steps

- Add more models (SVM, gradient boosting) and compare them.
- Optimise for malignant-class recall and choose the decision threshold from the ROC curve.
- Wrap the model in a small Streamlit app.
