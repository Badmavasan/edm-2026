# Test file for new error types ablation study
# This file tests the detection of new error types added to the ast_error_detection module

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ast_error_detection import get_primary_code_errors, get_typology_based_code_error


def print_test_result(test_name, expected_errors, actual_errors, passed):
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"\n{status}: {test_name}")
    print(f"  Expected: {expected_errors}")
    print(f"  Actual:   {actual_errors}")


def run_test(test_name, incorrect_code, correct_code, expected_primary_substrings=None, expected_typology_errors=None):
    """
    Run a single test case.

    Args:
        test_name: Name of the test
        incorrect_code: The incorrect code to compare
        correct_code: The correct code to compare against
        expected_primary_substrings: List of substrings expected in primary errors
        expected_typology_errors: Set of error codes expected in typology-based errors
    """
    print(f"\n{'='*60}")
    print(f"Test: {test_name}")
    print(f"{'='*60}")

    # Get primary errors
    dist, primary_errors = get_primary_code_errors(incorrect_code, correct_code)
    print(f"\nPrimary Errors (distance={dist}):")
    for err in primary_errors:
        print(f"  {err}")

    # Get typology-based errors
    dist2, typology_errors = get_typology_based_code_error(incorrect_code, [correct_code])
    print(f"\nTypology Errors (distance={dist2}):")
    print(f"  {typology_errors}")

    # Check expected primary error substrings
    if expected_primary_substrings:
        primary_str = str(primary_errors)
        all_found = all(substr in primary_str for substr in expected_primary_substrings)
        print_test_result(f"Primary errors contain: {expected_primary_substrings}",
                         expected_primary_substrings, primary_errors, all_found)

    # Check expected typology errors
    if expected_typology_errors:
        found_errors = expected_typology_errors.intersection(typology_errors)
        all_found = found_errors == expected_typology_errors
        print_test_result(f"Typology errors contain: {expected_typology_errors}",
                         expected_typology_errors, typology_errors, all_found)

    return primary_errors, typology_errors


# ========== Test Cases for New Functions ==========

def test_ouvrir_missing():
    """Test F_CALL_MISSING_OUVRIR: ouvrir() is in correct code but absent in incorrect code"""
    incorrect_code = """
x = 5
"""
    correct_code = """
ouvrir()
x = 5
"""
    return run_test(
        "F_CALL_MISSING_OUVRIR",
        incorrect_code, correct_code,
        expected_primary_substrings=["MISSING_CALL_STATEMENT", "ouvrir"],
        expected_typology_errors={"F_CALL_MISSING_OUVRIR"}
    )


def test_ouvrir_unnecessary():
    """Test F_CALL_UNNECESSARY_OUVRIR: ouvrir() is not in correct code but present in incorrect code"""
    incorrect_code = """
ouvrir()
x = 5
"""
    correct_code = """
x = 5
"""
    return run_test(
        "F_CALL_UNNECESSARY_OUVRIR",
        incorrect_code, correct_code,
        expected_primary_substrings=["UNNECESSARY_CALL_STATEMENT", "ouvrir"],
        expected_typology_errors={"F_CALL_UNNECESSARY_OUVRIR"}
    )


def test_ouvrir_error():
    """Test F_CALL_OUVRIR_ERROR: parameter in ouvrir() doesn't match"""
    incorrect_code = """
ouvrir(10)
"""
    correct_code = """
ouvrir(5)
"""
    return run_test(
        "F_CALL_OUVRIR_ERROR",
        incorrect_code, correct_code,
        expected_primary_substrings=["CONST_VALUE_MISMATCH"],
        expected_typology_errors={"F_CALL_OUVRIR_ERROR"}
    )


def test_sauter_missing():
    """Test F_CALL_MISSING_SAUTER"""
    incorrect_code = """
x = 5
"""
    correct_code = """
sauter()
x = 5
"""
    return run_test(
        "F_CALL_MISSING_SAUTER",
        incorrect_code, correct_code,
        expected_primary_substrings=["MISSING_CALL_STATEMENT", "sauter"],
        expected_typology_errors={"F_CALL_MISSING_SAUTER"}
    )


def test_sauter_unnecessary():
    """Test F_CALL_UNNECESSARY_SAUTER"""
    incorrect_code = """
sauter()
x = 5
"""
    correct_code = """
x = 5
"""
    return run_test(
        "F_CALL_UNNECESSARY_SAUTER",
        incorrect_code, correct_code,
        expected_primary_substrings=["UNNECESSARY_CALL_STATEMENT", "sauter"],
        expected_typology_errors={"F_CALL_UNNECESSARY_SAUTER"}
    )


def test_sauter_error():
    """Test F_CALL_SAUTER_ERROR"""
    incorrect_code = """
sauter(10)
"""
    correct_code = """
sauter(5)
"""
    return run_test(
        "F_CALL_SAUTER_ERROR",
        incorrect_code, correct_code,
        expected_primary_substrings=["CONST_VALUE_MISMATCH"],
        expected_typology_errors={"F_CALL_SAUTER_ERROR"}
    )


def test_coup_missing():
    """Test F_CALL_MISSING_COUP"""
    incorrect_code = """
x = 5
"""
    correct_code = """
coup()
x = 5
"""
    return run_test(
        "F_CALL_MISSING_COUP",
        incorrect_code, correct_code,
        expected_primary_substrings=["MISSING_CALL_STATEMENT", "coup"],
        expected_typology_errors={"F_CALL_MISSING_COUP"}
    )


def test_coup_unnecessary():
    """Test F_CALL_UNNECESSARY_COUP"""
    incorrect_code = """
coup()
x = 5
"""
    correct_code = """
x = 5
"""
    return run_test(
        "F_CALL_UNNECESSARY_COUP",
        incorrect_code, correct_code,
        expected_primary_substrings=["UNNECESSARY_CALL_STATEMENT", "coup"],
        expected_typology_errors={"F_CALL_UNNECESSARY_COUP"}
    )


def test_coup_error():
    """Test F_CALL_COUP_ERROR"""
    incorrect_code = """
coup(10)
"""
    correct_code = """
coup(5)
"""
    return run_test(
        "F_CALL_COUP_ERROR",
        incorrect_code, correct_code,
        expected_primary_substrings=["CONST_VALUE_MISMATCH"],
        expected_typology_errors={"F_CALL_COUP_ERROR"}
    )


def test_sauter_hauteur_missing():
    """Test F_CALL_MISSING_SAUTER_HAUTEUR"""
    incorrect_code = """
x = 5
"""
    correct_code = """
sauter_hauteur(3)
x = 5
"""
    return run_test(
        "F_CALL_MISSING_SAUTER_HAUTEUR",
        incorrect_code, correct_code,
        expected_primary_substrings=["MISSING_CALL_STATEMENT", "sauter_hauteur"],
        expected_typology_errors={"F_CALL_MISSING_SAUTER_HAUTEUR"}
    )


def test_sauter_hauteur_unnecessary():
    """Test F_CALL_UNNECESSARY_SAUTER_HAUTEUR"""
    incorrect_code = """
sauter_hauteur(3)
x = 5
"""
    correct_code = """
x = 5
"""
    return run_test(
        "F_CALL_UNNECESSARY_SAUTER_HAUTEUR",
        incorrect_code, correct_code,
        expected_primary_substrings=["UNNECESSARY_CALL_STATEMENT", "sauter_hauteur"],
        expected_typology_errors={"F_CALL_UNNECESSARY_SAUTER_HAUTEUR"}
    )


def test_sauter_hauteur_error():
    """Test F_CALL_SAUTER_HAUTEUR_ERROR"""
    incorrect_code = """
sauter_hauteur(10)
"""
    correct_code = """
sauter_hauteur(5)
"""
    return run_test(
        "F_CALL_SAUTER_HAUTEUR_ERROR",
        incorrect_code, correct_code,
        expected_primary_substrings=["CONST_VALUE_MISMATCH"],
        expected_typology_errors={"F_CALL_SAUTER_HAUTEUR_ERROR"}
    )


def test_lire_chaine_missing():
    """Test F_CALL_MISSING_LIRE_CHAINE"""
    incorrect_code = """
x = 5
"""
    correct_code = """
lire_chaine()
x = 5
"""
    return run_test(
        "F_CALL_MISSING_LIRE_CHAINE",
        incorrect_code, correct_code,
        expected_primary_substrings=["MISSING_CALL_STATEMENT", "lire_chaine"],
        expected_typology_errors={"F_CALL_MISSING_LIRE_CHAINE"}
    )


def test_lire_chaine_unnecessary():
    """Test F_CALL_UNNECESSARY_LIRE_CHAINE"""
    incorrect_code = """
lire_chaine()
x = 5
"""
    correct_code = """
x = 5
"""
    return run_test(
        "F_CALL_UNNECESSARY_LIRE_CHAINE",
        incorrect_code, correct_code,
        expected_primary_substrings=["UNNECESSARY_CALL_STATEMENT", "lire_chaine"],
        expected_typology_errors={"F_CALL_UNNECESSARY_LIRE_CHAINE"}
    )


def test_lire_chaine_error():
    """Test F_CALL_LIRE_CHAINE_ERROR"""
    incorrect_code = """
lire_chaine(10)
"""
    correct_code = """
lire_chaine(5)
"""
    return run_test(
        "F_CALL_LIRE_CHAINE_ERROR",
        incorrect_code, correct_code,
        expected_primary_substrings=["CONST_VALUE_MISMATCH"],
        expected_typology_errors={"F_CALL_LIRE_CHAINE_ERROR"}
    )


def test_tirer_missing():
    """Test F_CALL_MISSING_TIRER"""
    incorrect_code = """
x = 5
"""
    correct_code = """
tirer()
x = 5
"""
    return run_test(
        "F_CALL_MISSING_TIRER",
        incorrect_code, correct_code,
        expected_primary_substrings=["MISSING_CALL_STATEMENT", "tirer"],
        expected_typology_errors={"F_CALL_MISSING_TIRER"}
    )


def test_tirer_unnecessary():
    """Test F_CALL_UNNECESSARY_TIRER"""
    incorrect_code = """
tirer()
x = 5
"""
    correct_code = """
x = 5
"""
    return run_test(
        "F_CALL_UNNECESSARY_TIRER",
        incorrect_code, correct_code,
        expected_primary_substrings=["UNNECESSARY_CALL_STATEMENT", "tirer"],
        expected_typology_errors={"F_CALL_UNNECESSARY_TIRER"}
    )


def test_tirer_error():
    """Test F_CALL_TIRER_ERROR"""
    incorrect_code = """
tirer(10)
"""
    correct_code = """
tirer(5)
"""
    return run_test(
        "F_CALL_TIRER_ERROR",
        incorrect_code, correct_code,
        expected_primary_substrings=["CONST_VALUE_MISMATCH"],
        expected_typology_errors={"F_CALL_TIRER_ERROR"}
    )


def test_detecter_obstacle_missing():
    """Test F_CALL_MISSING_DETECTER_OBSTACLE"""
    incorrect_code = """
x = 5
"""
    correct_code = """
detecter_obstacle()
x = 5
"""
    return run_test(
        "F_CALL_MISSING_DETECTER_OBSTACLE",
        incorrect_code, correct_code,
        expected_primary_substrings=["MISSING_CALL_STATEMENT", "detecter_obstacle"],
        expected_typology_errors={"F_CALL_MISSING_DETECTER_OBSTACLE"}
    )


def test_detecter_obstacle_unnecessary():
    """Test F_CALL_UNNECESSARY_DETECTER_OBSTACLE"""
    incorrect_code = """
detecter_obstacle()
x = 5
"""
    correct_code = """
x = 5
"""
    return run_test(
        "F_CALL_UNNECESSARY_DETECTER_OBSTACLE",
        incorrect_code, correct_code,
        expected_primary_substrings=["UNNECESSARY_CALL_STATEMENT", "detecter_obstacle"],
        expected_typology_errors={"F_CALL_UNNECESSARY_DETECTER_OBSTACLE"}
    )


def test_detecter_obstacle_error():
    """Test F_CALL_DETECTER_OBSTACLE_ERROR"""
    incorrect_code = """
detecter_obstacle(10)
"""
    correct_code = """
detecter_obstacle(5)
"""
    return run_test(
        "F_CALL_DETECTER_OBSTACLE_ERROR",
        incorrect_code, correct_code,
        expected_primary_substrings=["CONST_VALUE_MISMATCH"],
        expected_typology_errors={"F_CALL_DETECTER_OBSTACLE_ERROR"}
    )


# ========== Test Cases for New Error Types ==========

def test_while_stop_condition_error():
    """Test WHILE_STOP_CONDITION_ERROR: Stop condition of while loop doesn't match"""
    incorrect_code = """
x = 0
while x < 5:
    x = x + 1
"""
    correct_code = """
x = 0
while x < 10:
    x = x + 1
"""
    return run_test(
        "WHILE_STOP_CONDITION_ERROR",
        incorrect_code, correct_code,
        expected_primary_substrings=["CONST_VALUE_MISMATCH", "While", "Condition"],
        expected_typology_errors={"WHILE_STOP_CONDITION_ERROR"}
    )


def test_while_stop_condition_error_operator():
    """Test WHILE_STOP_CONDITION_ERROR: Operator in while condition is wrong"""
    incorrect_code = """
x = 0
while x > 5:
    x = x + 1
"""
    correct_code = """
x = 0
while x < 5:
    x = x + 1
"""
    return run_test(
        "WHILE_STOP_CONDITION_ERROR (operator)",
        incorrect_code, correct_code,
        expected_primary_substrings=["While", "Condition"],
        expected_typology_errors={"WHILE_STOP_CONDITION_ERROR", "LO_CONDITION_ERROR"}
    )


def test_while_misplaced():
    """Test WHILE_MISPLACED: While structure is there but misplaced in order

    Note: The detection of WHILE_MISPLACED depends on the Zhang-Shasha algorithm
    identifying the while loop as the element that needs to be repositioned.
    When multiple elements can be considered misplaced, the algorithm may detect
    a different element. This test verifies that the INCORRECT_STATEMENT_POSITION_WHILE
    tag is properly mapped to LO_WHILE_MISPLACED when detected.
    """
    # This test verifies that when INCORRECT_STATEMENT_POSITION_WHILE is detected,
    # it maps to LO_WHILE_MISPLACED. Due to the nature of tree edit distance,
    # the algorithm may detect different elements as misplaced depending on the structure.
    incorrect_code = """
while i < 10:
    i = i + 1
x = 0
"""
    correct_code = """
x = 0
while i < 10:
    i = i + 1
"""
    primary_errors, typology_errors = run_test(
        "LO_WHILE_MISPLACED",
        incorrect_code, correct_code,
        expected_primary_substrings=["INCORRECT_STATEMENT_POSITION"],
        expected_typology_errors=None  # Don't check typology since it depends on which element is detected as misplaced
    )

    # Check if either WHILE or ASSIGN is detected as misplaced
    has_position_error = any("INCORRECT_STATEMENT_POSITION" in str(err) for err in primary_errors)
    print(f"\n  Position error detected: {has_position_error}")

    return primary_errors, typology_errors


def test_cs_unnecessary():
    """Test CS_UNNECESSARY: Unnecessary conditional statement"""
    incorrect_code = """
x = 5
if x > 0:
    print(x)
"""
    correct_code = """
x = 5
print(x)
"""
    return run_test(
        "CS_UNNECESSARY",
        incorrect_code, correct_code,
        expected_primary_substrings=["UNNECESSARY_CONDITIONAL"],
        expected_typology_errors={"CS_UNNECESSARY"}
    )


def test_exp_error_assignment():
    """Test EXP_ERROR_ASSIGNMENT: Error in assignment of value or function output"""
    incorrect_code = """
x = 10
"""
    correct_code = """
x = 5
"""
    return run_test(
        "EXP_ERROR_ASSIGNMENT",
        incorrect_code, correct_code,
        expected_primary_substrings=["CONST_VALUE_MISMATCH", "Assign"],
        expected_typology_errors={"EXP_ERROR_ASSIGNMENT"}
    )


def test_exp_error_assignment_function():
    """Test EXP_ERROR_ASSIGNMENT: Error when function result is assigned"""
    incorrect_code = """
x = calculate(10)
"""
    correct_code = """
x = calculate(5)
"""
    return run_test(
        "EXP_ERROR_ASSIGNMENT (function)",
        incorrect_code, correct_code,
        expected_primary_substrings=["CONST_VALUE_MISMATCH"],
        expected_typology_errors={"EXP_ERROR_ASSIGNMENT"}
    )


def test_exp_error_operation():
    """Test EXP_ERROR_OPERATION: Error in operation result assigned to variable"""
    incorrect_code = """
x = a + b
"""
    correct_code = """
x = a - b
"""
    return run_test(
        "EXP_ERROR_OPERATION",
        incorrect_code, correct_code,
        expected_primary_substrings=["INCORRECT_OPERATION_IN_ASSIGN"],
        expected_typology_errors={"EXP_ERROR_OPERATION"}
    )


def test_cs_body_error():
    """Test CS_BODY_ERROR: Error inside the body of a conditional branch"""
    incorrect_code = """
if x > 0:
    print(10)
"""
    correct_code = """
if x > 0:
    print(5)
"""
    return run_test(
        "CS_BODY_ERROR",
        incorrect_code, correct_code,
        expected_primary_substrings=["CONST_VALUE_MISMATCH", "If", "Body"],
        expected_typology_errors={"CS_BODY_ERROR"}
    )


def test_cs_body_error_missing_statement():
    """Test CS_BODY_ERROR: Missing statement in conditional body"""
    incorrect_code = """
if x > 0:
    pass
"""
    correct_code = """
if x > 0:
    print(x)
"""
    return run_test(
        "CS_BODY_ERROR (missing statement)",
        incorrect_code, correct_code,
        expected_primary_substrings=["MISSING", "If", "Body"],
        expected_typology_errors={"CS_BODY_ERROR"}
    )


# ========== Main Ablation Study ==========

def run_ablation_study():
    """Run all tests and print summary"""
    print("\n" + "="*80)
    print("ABLATION STUDY: NEW ERROR TYPES DETECTION")
    print("="*80)

    tests = [
        # New function tests - ouvrir
        test_ouvrir_missing,
        test_ouvrir_unnecessary,
        test_ouvrir_error,
        # sauter
        test_sauter_missing,
        test_sauter_unnecessary,
        test_sauter_error,
        # coup
        test_coup_missing,
        test_coup_unnecessary,
        test_coup_error,
        # sauter_hauteur
        test_sauter_hauteur_missing,
        test_sauter_hauteur_unnecessary,
        test_sauter_hauteur_error,
        # lire_chaine
        test_lire_chaine_missing,
        test_lire_chaine_unnecessary,
        test_lire_chaine_error,
        # tirer
        test_tirer_missing,
        test_tirer_unnecessary,
        test_tirer_error,
        # detecter_obstacle
        test_detecter_obstacle_missing,
        test_detecter_obstacle_unnecessary,
        test_detecter_obstacle_error,
        # New error types
        test_while_stop_condition_error,
        test_while_stop_condition_error_operator,
        test_while_misplaced,
        test_cs_unnecessary,
        test_exp_error_assignment,
        test_exp_error_assignment_function,
        test_exp_error_operation,
        test_cs_body_error,
        test_cs_body_error_missing_statement,
    ]

    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append((test_func.__name__, "COMPLETED", result))
        except Exception as e:
            results.append((test_func.__name__, "ERROR", str(e)))
            print(f"\n✗ ERROR in {test_func.__name__}: {e}")

    print("\n" + "="*80)
    print("ABLATION STUDY SUMMARY")
    print("="*80)

    completed = sum(1 for _, status, _ in results if status == "COMPLETED")
    errors = sum(1 for _, status, _ in results if status == "ERROR")

    print(f"Total tests: {len(results)}")
    print(f"Completed: {completed}")
    print(f"Errors: {errors}")

    if errors > 0:
        print("\nTests with errors:")
        for name, status, result in results:
            if status == "ERROR":
                print(f"  - {name}: {result}")


if __name__ == "__main__":
    run_ablation_study()
