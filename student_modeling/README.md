# Student Knowledge Modeling

This project implements and compares four student knowledge modeling approaches for predicting learner performance on programming exercises. Each model is evaluated on two educational platform datasets using a rigorous, student-level cross-validation framework with no data leakage.

## Models

| Model | Type | Description |
|---|---|---|
| **BKT** (Bayesian Knowledge Tracing) | Probabilistic | Hidden Markov Model tracking latent skill mastery via learn, guess, slip, and forget parameters. Uses the [pyBKT](https://github.com/CAHLR/pyBKT) library. |
| **DKT** (Deep Knowledge Tracing) | Neural | LSTM-based sequence model that learns a continuous knowledge state from interaction histories. Implemented in PyTorch. |
| **IRT** (Item Response Theory) | Probabilistic | 1PL Rasch model estimating student ability and item difficulty. P(correct) = sigmoid(theta - b), fit via gradient descent. |
| **PFA** (Performance Factor Analysis) | Logistic Regression | Predicts performance from one-hot skill features and cumulative success/failure counts per student-skill pair. Uses scikit-learn. |

## Project Structure

```
student_modeling/
├── config/                                  # Configuration files
│   ├── exercises.json                       # Exercise metadata (164 exercises, series A-G)
│   ├── error-type-of-task-association.json   # Error tag to task type mapping
│   ├── exercise-association-platform-a.json
│   ├── exercise-type-of-task-association-platform-a.json
│   └── exercise_tag_to_exercise_type_association_platform_a.json
│
├── data/
│   ├── platform_a_dataset.csv               # Primary learning platform dataset
│   └── platform_b_dataset.csv               # Secondary learning platform dataset
│
├── src/
│   ├── common/
│   │   └── dataset_config.py                # Multi-dataset configuration management
│   │
│   ├── bkt/                                 # Bayesian Knowledge Tracing
│   │   ├── config.py                        # BKT hyperparameters and paths
│   │   ├── data_loader.py                   # CSV loading and JSON field parsing
│   │   ├── data_transformer.py              # Transforms to long format (student, skill, correct)
│   │   ├── student_split.py                 # Student-level train/test split
│   │   ├── trainer.py                       # pyBKT model training
│   │   ├── evaluator.py                     # Metrics, CV, incremental evaluation
│   │   ├── plotting.py                      # ROC curves, AUC plots, CSV import/export
│   │   ├── run_bkt_platform_a.py            # Full pipeline for Platform A
│   │   └── run_bkt_platform_b.py            # Full pipeline for Platform B
│   │
│   ├── dkt/                                 # Deep Knowledge Tracing
│   │   ├── config.py                        # DKT hyperparameters (hidden_dim, lr, etc.)
│   │   ├── model.py                         # DKTModel (Embedding -> LSTM -> Linear -> Sigmoid)
│   │   ├── data_transformer.py              # Builds per-student interaction sequences
│   │   ├── trainer.py                       # Training loop with early stopping
│   │   ├── evaluator.py                     # Metrics computation
│   │   ├── cross_validation.py              # K-fold CV with student stratification
│   │   ├── incremental.py                   # Incremental eval by samples and students
│   │   ├── run_dkt_experiment.py            # Ablation study on hidden dimensions
│   │   └── plot_dkt_results.py              # DKT-specific result visualizations
│   │
│   ├── irt/                                 # Item Response Theory (1PL Rasch)
│   │   ├── config.py                        # IRT hyperparameters (lr, regularization)
│   │   ├── model.py                         # IRTModel with theta/b parameters
│   │   ├── data_transformer.py              # Transforms with student/item indexing
│   │   ├── trainer.py                       # Gradient descent fitting
│   │   ├── evaluator.py                     # Evaluation on held-out data
│   │   ├── plotting.py                      # IRT-specific plots
│   │   ├── run_irt_platform_a.py            # Full pipeline for Platform A
│   │   └── run_irt_platform_b.py            # Full pipeline for Platform B
│   │
│   ├── pfa_model/                           # Performance Factor Analysis
│   │   ├── config.py                        # PFA hyperparameters
│   │   ├── data_transformer.py              # Cumulative success/failure feature building
│   │   ├── trainer.py                       # LogisticRegression + StandardScaler
│   │   ├── evaluator.py                     # Evaluation pipeline
│   │   ├── plotting.py                      # PFA-specific plots
│   │   ├── run_pfa_platform_a.py            # Full pipeline for Platform A
│   │   └── run_pfa_platform_b.py            # Full pipeline for Platform B
│   │
│   ├── compare_dkt_classical.py             # DKT vs classical model comparison
│   ├── plot_incremental_training.py         # Incremental training plots (by samples)
│   ├── plot_incremental_training_by_students.py  # Incremental training plots (by students)
│   ├── statistical_significance_test_global.py   # Statistical significance tests across models
│   └── summarize_learned_parameters.py      # Summarizes learned BKT/IRT parameters
│
└── README.md
```

## Key Design Choices

### Two Error Modalities

Each model is trained and evaluated under two complementary error modalities:

- **Error-Independent**: All expected skills for a submission receive the global outcome (ok/ko). Every skill listed in `expected_type_tasks` is marked with the submission's overall status.
- **Error-Dependent**: Only the skills identified in `tasks_from_errors` are marked as incorrect; all other expected skills are marked as correct. This provides a more granular learning signal at the skill level.

### Multi-Level Analysis

Models are trained at three granularity levels (two for Platform B):

| Level | Description |
|---|---|
| **Global** | A single model trained on all data |
| **Exercise Type** | Separate models per exercise type (e.g., consoleDisplay, robot, design) |
| **Exercise Tag** | Individual models per exercise |

### No Data Leakage

Student-level splitting ensures that no student appears in both training and test sets. Skills present in the test set but absent from training are filtered out. Cross-validation folds are stratified by student.

### Incremental Training

All models are evaluated on learning curves with fixed sample sizes (10, 50, 100, 500, 1K, 5K, 10K, 50K, 100K interactions) and fixed student counts (5, 10, 20, 50, 100, 200, 500, 1K, 2K, 5K students) to assess data efficiency.

## Data

### Datasets

- **`data/platform_a_dataset.csv`** -- Student interaction traces from Platform A. Key columns: `compte_hash` (student ID), `statut` (ok/ko), `date_created`, `exercise_id`, `exercise_type`, `exercise_tag`, `expected_type_tasks` (JSON list of expected skills), `tasks_from_errors` (JSON list of error-specific skills).
- **`data/platform_b_dataset.csv`** -- Student interaction traces from Platform B with equivalent structure (different column names: `id_compte`, `status`, `expected_task_types`, `task_from_errors`, `exercise_tag`).

### Configuration Files

- **`config/exercises.json`** -- Metadata for 164 programming exercises across 6 series (A-G), including exercise type, title, tasks, and correct codes.
- **`config/error-type-of-task-association.json`** -- Maps error tags to task type codes from the didactic typology.
- **`config/exercise-association-platform-a.json`** -- Exercise-level associations for Platform A.
- **`config/exercise-type-of-task-association-platform-a.json`** -- Maps exercises to their task types on Platform A.
- **`config/exercise_tag_to_exercise_type_association_platform_a.json`** -- Maps exercise tags to exercise types on Platform A.

## Usage

### Running a Model Pipeline

Each model has a self-contained run script per platform. From the `student_modeling/` directory:

```bash
# BKT
python src/bkt/run_bkt_platform_a.py
python src/bkt/run_bkt_platform_b.py

# DKT (ablation study over hidden dimensions: 8, 16, 32, 64, 128)
python src/dkt/run_dkt_experiment.py

# IRT
python src/irt/run_irt_platform_a.py
python src/irt/run_irt_platform_b.py

# PFA
python src/pfa_model/run_pfa_platform_a.py
python src/pfa_model/run_pfa_platform_b.py
```

Each script executes the full pipeline: load data, transform, split students, cross-validate, evaluate incrementally, generate plots, and save results.

### Cross-Model Comparison

```bash
# Compare DKT against classical models (BKT, IRT, PFA)
python src/compare_dkt_classical.py

# Plot incremental training curves across all models
python src/plot_incremental_training.py
python src/plot_incremental_training_by_students.py

# Run statistical significance tests (e.g., paired t-tests across CV folds)
python src/statistical_significance_test_global.py

# Summarize learned parameters (BKT: learn/guess/slip/forget, IRT: theta/b)
python src/summarize_learned_parameters.py
```

## Output Structure

Each model run produces a hierarchical output directory:

```
results/{platform}/{model}/
├── global/
│   ├── error_independent/
│   │   ├── fold_metrics.csv              # Per-fold CV metrics
│   │   ├── predictions.csv               # Per-interaction predictions
│   │   ├── learned_parameters.csv        # Model parameters
│   │   ├── roc_data.csv                  # ROC curve data
│   │   └── incremental_training_metrics.csv
│   └── error_dependent/
│       └── ...
├── exercise_type/
│   ├── error_independent/{type}/...
│   └── error_dependent/{type}/...
├── exercise_tag/
│   ├── error_independent/{exercise}/...
│   └── error_dependent/{exercise}/...
├── global_accuracy.csv
├── per_exercise_type_accuracy.csv
├── per_exercise_accuracy.csv
├── roc_comparison.png
└── incremental_training_auc.png
```

## Evaluation Metrics

All models are evaluated using:

- **AUC** (Area Under ROC Curve) -- Primary metric for discriminative performance
- **Accuracy** -- Overall classification accuracy at a fixed threshold
- **F1 Score** -- Harmonic mean of precision and recall

## Dependencies

- `pandas` -- Data manipulation
- `numpy` -- Numerical computing
- `scikit-learn` (1.3.2) -- ML metrics, logistic regression, cross-validation utilities
- `torch` -- Deep learning (DKT model)
- `pyBKT` (1.4.1) -- Bayesian Knowledge Tracing library
- `matplotlib` -- Plotting
- `seaborn` -- Statistical visualizations
- `tqdm` -- Progress bars

Install all dependencies:

```bash
pip install -r requirements.txt
```

## License

This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0). See the license headers in source files for details.