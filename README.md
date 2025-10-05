# DiceyLoans

This project trains a loan-approval classifier and surfaces minimal counterfactual edits that flip each decision. It packages a lightweight `scikit-learn` pipeline with `dice-ml` to explain model behavior on structured CSV data.



## Architecture at a Glance

- **Dataset ingestion**: `src/utils/data_loader.py` reads a CSV, infers feature types (large integer columns remain numeric), imputes missing values, one-hot encodes categoricals, and returns a `DatasetBundle` with train/test splits plus metadata for reversing encodings.
- **Predictor**: `src/models/tabular_predictor.py` fits a `StandardScaler` + `LogisticRegression` pipeline. Probabilities, labels, and accuracy are exposed through helper methods.
- **Counterfactual engine**: `src/counterfactuals/dice_explainer.py` wraps `dice-ml`. It scores generated candidates, falls back to the genetic method if the primary search (`random` by default) misses the target class, and returns only counterfactuals that truly flip the prediction along with readable feature deltas.
- **CLI orchestrator**: `src/generate_counterfactuals.py` wires everything together. It loads the default dataset, trains the model, selects a record, resolves the opposite outcome when no target is provided, and prints a JSON summary describing the minimal change.

## Dataset

- Default source: `loan_approval_dataset.csv` (packaged in the repo).
- Target column: `loan_status`.
- Any alternate CSV with a binary outcome can be supplied at runtime; categorical columns should use string values, numerics remain scalar.

## Usage

From the project root:

```powershell
python -m src.generate_counterfactuals
```

Optional switches:

- `--csv PATH` custom dataset location.
- `--target COL` outcome column name.
- `--positive-label LABEL` label interpreted as the positive class.
- `--index N` explain a specific row; otherwise one is auto-selected.
- `--desired-label LABEL` or `--desired INT` pin the desired outcome (defaults to the opposite of the model’s prediction).
- `--method {random,kdtree,genetic}` choose the DiCE search strategy.
- `--total K` request additional candidate counterfactuals before the closest valid flip is chosen.
- `--threshold P` adjust the classification cutoff when evaluating flips.
- `--tune` enable logistic-regression hyperparameter tuning (searches over regularization strength and class weights).
- `--c-grid v1,v2,...` override the default grid of C values evaluated when tuning.
- `--cv F` change the cross-validation fold count used during tuning (default: 5).
- `--scoring METRIC` pick the scikit-learn scoring metric for tuning (default: `roc_auc`).
- `--n-jobs N` control the parallelism for the tuning search.

Output: a JSON document containing the original and counterfactual probabilities, textual labels, the method that produced the final counterfactual, raw feature values before/after, and a `changes` array with per-feature adjustments. When tuning is enabled the summary also includes the best-performing hyperparameters.

### Examples

Run the default pipeline and auto-selected record:

```powershell
python -m src.generate_counterfactuals --index 11
```

Repeat for the same record with hyperparameter tuning enabled:

```powershell
python -m src.generate_counterfactuals --index 11 --tune
```

Expand the tuning search with a custom regularization grid:

```powershell
python -m src.generate_counterfactuals --index 11 --tune --c-grid 0.01,0.1,1,10 --cv 5
```