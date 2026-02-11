"""
Annotation script for student code submission data.
Processes raw interaction traces and adds error annotations, task types, and one-hot encodings.
"""

import sys
import os
import ast
import json
import pandas as pd
from collections import defaultdict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ast_error_detection import get_typology_based_code_error


def load_config_files(project_root):
    """Load configuration files."""
    with open(os.path.join(project_root, 'config', 'level-task-types-platform_b.json'), 'r') as f:
        level_task_types = json.load(f)

    with open(os.path.join(project_root, 'config', 'exercise-correct-codes-platform_b.json'), 'r') as f:
        correct_codes = json.load(f)

    with open(os.path.join(project_root, 'config', 'error-type-of-task-association_platform_b.json'), 'r') as f:
        error_task_assoc = json.load(f)

    return level_task_types, correct_codes, error_task_assoc


def is_valid_python(code):
    """Check if code is valid Python syntax."""
    if pd.isna(code) or code is None or str(code).strip() == '' or str(code).strip().upper() == 'NULL':
        return False
    try:
        ast.parse(str(code))
        return True
    except SyntaxError:
        return False


def get_expected_task_types(level_id, level_task_types):
    """Get expected task types for a given level."""
    level_key = f"level{level_id}"
    if level_key not in level_task_types or level_key == "_metadata":
        return []

    task_types = level_task_types[level_key].get('task_types', [])
    return [tt['task_code'] for tt in task_types]


def get_errors_for_code(code, level_id, correct_codes):
    """Get typology errors for a given code compared to correct code."""
    level_key = f"level{level_id}"
    if level_key not in correct_codes:
        return set()

    correct_code_list = correct_codes[level_key]

    try:
        _, errors = get_typology_based_code_error(str(code), correct_code_list)
        return errors
    except Exception as e:
        print(f"Error processing code: {e}")
        return set()


def get_tasks_from_errors(errors, error_task_assoc):
    """Map errors to task codes using error-task association."""
    task_codes = set()
    for assoc in error_task_assoc:
        if assoc['error_tag'] in errors:
            task_codes.add(assoc['task_code'])
    return list(task_codes)


def build_error_to_task_mapping(error_task_assoc):
    """Build a mapping from error tags to task codes."""
    mapping = defaultdict(set)
    for assoc in error_task_assoc:
        mapping[assoc['error_tag']].add(assoc['task_code'])
    return mapping


def main():
    # Project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Load configuration files
    print("Loading configuration files...")
    level_task_types, correct_codes, error_task_assoc = load_config_files(project_root)
    error_to_task = build_error_to_task_mapping(error_task_assoc)

    # Collect all unique task codes for one-hot encoding
    all_task_codes = set()
    for level_key, level_data in level_task_types.items():
        if level_key == "_metadata":
            continue
        for tt in level_data.get('task_types', []):
            all_task_codes.add(tt['task_code'])

    # Also add task codes from error associations
    for assoc in error_task_assoc:
        all_task_codes.add(assoc['task_code'])

    all_task_codes = sorted(list(all_task_codes))
    print(f"Total unique task codes: {len(all_task_codes)}")

    # Load raw data
    print("Loading raw data...")
    raw_data_path = os.path.join(project_root, 'data', 'raw-data-interaction-traces-platform_b.csv')
    df = pd.read_csv(raw_data_path, sep=';', low_memory=False, encoding='latin-1')
    print(f"Total rows in raw data: {len(df)}")

    # Filter: only LAUNCHED actions
    print("Filtering LAUNCHED actions...")
    df_launched = df[df['action_name'] == 'LAUNCHED'].copy()
    print(f"Rows with LAUNCHED action: {len(df_launched)}")

    # Filter: valid Python code
    print("Filtering valid Python code...")
    df_launched['is_valid_python'] = df_launched['code'].apply(is_valid_python)
    df_valid = df_launched[df_launched['is_valid_python']].copy()
    df_valid = df_valid.drop(columns=['is_valid_python'])
    print(f"Rows with valid Python code: {len(df_valid)}")

    # Select required columns
    df_result = df_valid[['id', 'game_id', 'level_id', 'date', 'code', 'action_name', 'object_name']].copy()

    # Add status column
    print("Adding status column...")
    df_result['status'] = df_result['object_name'].apply(
        lambda x: 'ok' if x == 'LEVEL_COMPLETED_PROGRAM' else 'ko'
    )

    # Add expected_task_types column
    print("Adding expected_task_types column...")
    df_result['expected_task_types'] = df_result['level_id'].apply(
        lambda x: get_expected_task_types(x, level_task_types)
    )

    # Add errors column
    print("Adding errors column (this may take a while)...")
    errors_list = []
    total = len(df_result)
    for idx, (i, row) in enumerate(df_result.iterrows()):
        if (idx + 1) % 1000 == 0:
            print(f"  Processing {idx + 1}/{total}...")

        if row['status'] == 'ok':
            # No errors for successful submissions
            errors_list.append([])
        else:
            errors = get_errors_for_code(row['code'], row['level_id'], correct_codes)
            errors_list.append(list(errors))

    df_result['errors'] = errors_list

    # Add task_from_errors column
    print("Adding task_from_errors column...")
    df_result['task_from_errors'] = df_result['errors'].apply(
        lambda errors: get_tasks_from_errors(set(errors), error_task_assoc)
    )

    # One-hot encoding: task__(task_code) = 1 if task_code is in task_from_errors AND in expected_task_types
    print("Adding one-hot encoded columns...")
    for task_code in all_task_codes:
        col_name = f"task__{task_code}"
        df_result[col_name] = df_result.apply(
            lambda row: 1 if task_code in row['task_from_errors'] and task_code in row['expected_task_types'] else 0,
            axis=1
        )

    # Rename columns
    print("Renaming columns...")
    df_result = df_result.rename(columns={
        'date': 'date_created',
        'game_id': 'id_compte',
        'level_id': 'exercise_tag'
    })

    # Drop intermediate columns not needed
    df_result = df_result.drop(columns=['action_name', 'object_name'])

    # Remove rows where status is "ko" AND errors are empty (keep all "ok" rows)
    print("Removing rows with status 'ko' and empty errors...")
    before_count = len(df_result)
    df_result = df_result[~((df_result['status'] == 'ko') & (df_result['errors'].apply(len) == 0))]
    print(f"Removed {before_count - len(df_result)} rows with ko status and empty errors")
    print(f"Rows after filtering: {len(df_result)}")

    # Save output
    output_path = os.path.join(project_root, 'data', 'error-annotated-data-platform_b.csv')
    print(f"Saving annotated data to {output_path}...")
    df_result.to_csv(output_path, index=False)

    # Print summary statistics
    print("\n" + "="*60)
    print("ANNOTATION SUMMARY")
    print("="*60)
    print(f"Total annotated rows: {len(df_result)}")
    print(f"Status distribution:")
    print(df_result['status'].value_counts())
    print(f"\nExercise tag distribution:")
    print(df_result['exercise_tag'].value_counts().sort_index())

    # Count how many rows have at least one task flagged
    task_cols = [col for col in df_result.columns if col.startswith('task__')]
    df_result['has_task_error'] = df_result[task_cols].sum(axis=1) > 0
    print(f"\nRows with at least one task error flagged: {df_result['has_task_error'].sum()}")

    # Show sample of errors detected
    print("\nSample of detected errors (first 10 non-empty):")
    error_samples = df_result[df_result['errors'].apply(len) > 0].head(10)
    for _, row in error_samples.iterrows():
        print(f"  Level {row['exercise_tag']}: {row['errors'][:5]}...")

    print("\nAnnotation complete!")
    return df_result


if __name__ == "__main__":
    df = main()
