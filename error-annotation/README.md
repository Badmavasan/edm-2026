# AST-Based Error Annotation for Student Code

This project provides an automated pipeline for detecting and annotating errors in student Python code submissions. It compares student code against reference solutions using Abstract Syntax Tree (AST) analysis and the Zhang-Shasha tree edit distance algorithm, producing fine-grained error annotations grounded in a didactic task typology.

## Project Structure

```
error-annotation/
├── ast_error_detection/          # Core Python package
│   ├── __init__.py               # Public API exports
│   ├── convert_ast_to_custom_node.py  # AST to custom Node conversion
│   ├── node.py                   # Custom tree node representation
│   ├── node_functions.py         # Node utility functions
│   ├── zang_shasha_distance.py   # Zhang-Shasha tree edit distance
│   ├── error_annotation.py       # Error annotation from edit operations
│   ├── annotated_tree.py         # Annotated tree structures
│   ├── error_checks.py           # Typology-based error tag filtering
│   ├── error_diagnosis.py        # High-level error detection functions
│   ├── ast_visualizer.py         # AST visualization with Graphviz
│   └── constants.py              # Shared constants
│
├── scripts/
│   └── annotate_data.py          # Batch annotation script
│
├── tests/
│   ├── test_new_errors.py        # Ablation tests for error detection (v1)
│   └── test_new_errors_v2.py     # Ablation tests for error detection (v2)
│
├── config/
│   ├── TaskType.json             # Task typology definitions (FR/EN)
│   ├── level-task-types-platform_b.json       # Per-level expected task types
│   ├── exercise-correct-codes-platform_b.json # Reference solutions per exercise
│   └── error-type-of-task-association_platform_b.json  # Mapping from error tags to task codes
│
├── data/
│   ├── raw-data-interaction-traces-platform_b.csv   # Raw interaction traces from Platform B
│   └── error-annotated-data-platform_b.csv          # Error-annotated output
│
├── main.py                       # Interactive demo and visualization
└── README.md
```

## The `ast_error_detection` Package

The core package exposes four main functions:

| Function | Description |
|---|---|
| `get_primary_code_errors(code1, code2)` | Computes the tree edit distance between two code snippets and returns detailed error annotations based on the edit operations. |
| `get_typology_based_code_error(incorrect_code, correct_code_list)` | Compares incorrect code against multiple reference solutions, applies a typology-based filter, and returns the closest match with its error tags. |
| `visualize_custom_ast_from_code(code)` | Renders the custom AST representation of a code snippet as a PNG image. |
| `visualize_plain_ast_from_code(code)` | Renders the standard Python AST of a code snippet as a PNG image. |

### How It Works

1. Both the student code and the reference code are parsed into Python ASTs.
2. Each AST is converted into a custom `Node` tree tailored for comparison.
3. The Zhang-Shasha tree edit distance algorithm computes the minimal set of edit operations (insert, delete, rename) to transform one tree into the other.
4. The edit operations are processed into human-readable error annotations with structural paths (e.g., `While > Body > MISSING_CALL_STATEMENT > avancer`).
5. A typology overlay filters these annotations into a controlled set of error tags aligned with a didactic task typology.

## Quick Start

### Try It Out with `main.py`

Use `main.py` to interactively test error detection and visualize AST differences:

```python
from ast_error_detection import *

code1 = """
for i in range(7):
    print(i+1)
"""

code2 = """
for i in range(6):
    print(i*2)
"""

# Get error annotations between two code snippets
dist, errors = get_primary_code_errors(code1, code2)

# Get typology-based errors (compares against a list of correct solutions)
dist, error_tags = get_typology_based_code_error(code1, [code2])

# Visualize the AST of a code snippet
visualize_custom_ast_from_code(code1)
visualize_plain_ast_from_code(code1)
```

### Batch Annotation with `scripts/annotate_data.py`

To annotate raw interaction trace data from Platform B:

```bash
python scripts/annotate_data.py
```

This script:
1. Loads the raw interaction traces from `data/raw-data-interaction-traces-platform_b.csv`
2. Filters for valid Python code submissions
3. Compares each incorrect submission against reference solutions using `get_typology_based_code_error`
4. Maps detected errors to task codes via the config files
5. Produces one-hot encoded task columns
6. Saves the annotated dataset to `data/error-annotated-data-platform_b.csv`

## Data

- **`data/raw-data-interaction-traces-platform_b.csv`** -- Raw student interaction traces collected from Platform B, including code submissions, timestamps, and level metadata.
- **`data/error-annotated-data-platform_b.csv`** -- The annotated dataset produced by the annotation pipeline. Each row includes the original submission data plus detected error tags, associated task codes, and one-hot encoded task columns.

## Configuration

- **`config/TaskType.json`** -- Defines the full task typology with bilingual labels (`task_name_fr`, `task_name_en`). Tasks are organized hierarchically by domain: algorithms (AL), programs (PR), functions (FO), loops (BO), variables (VA), expressions (EXP), and conditional structures (CS).
- **`config/level-task-types-platform_b.json`** -- Maps each exercise level to its expected task types and related errors.
- **`config/exercise-correct-codes-platform_b.json`** -- Contains the reference (correct) code solutions for each exercise level, used to compare against student submissions.
- **`config/error-type-of-task-association_platform_b.json`** -- Maps each error tag to one or more task codes from the typology.

## Tests

The `tests/` directory contains ablation studies that verify the error detection module against known code pairs from Platform B:

```bash
python tests/test_new_errors.py
python tests/test_new_errors_v2.py
```

These tests cover:
- Platform-specific function call errors (`ouvrir`, `sauter`, `coup`, `sauter_hauteur`, `lire_chaine`, `tirer`, `detecter_obstacle`, `lire_nombre`, `mesurer_hauteur`, `sauter_haut`)
- While loop errors (stop condition, misplaced loop, missing update)
- Conditional structure errors (unnecessary/missing `if`/`elif`/`else` branches, body errors)
- Expression errors (assignment, operation, unassigned function results)

## License

This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0). See the license headers in source files for details.