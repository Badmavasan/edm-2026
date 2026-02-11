# Test file for new error types (v2) ablation study
# Tests: Missing functions, CS_MISSING_ELIF/ELSE, LO_WHILE_MISSING_UPDATE, EXP_FUNCTION_RESULT_NOT_ASSIGNED

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
    """Run a single test case."""
    print(f"\n{'='*60}")
    print(f"Test: {test_name}")
    print(f"{'='*60}")

    dist, primary_errors = get_primary_code_errors(incorrect_code, correct_code)
    print(f"\nPrimary Errors (distance={dist}):")
    for err in primary_errors:
        print(f"  {err}")

    dist2, typology_errors = get_typology_based_code_error(incorrect_code, [correct_code])
    print(f"\nTypology Errors (distance={dist2}):")
    print(f"  {typology_errors}")

    if expected_primary_substrings:
        primary_str = str(primary_errors)
        all_found = all(substr in primary_str for substr in expected_primary_substrings)
        print_test_result(f"Primary errors contain: {expected_primary_substrings}",
                         expected_primary_substrings, primary_errors, all_found)

    if expected_typology_errors:
        found_errors = expected_typology_errors.intersection(typology_errors)
        all_found = found_errors == expected_typology_errors
        print_test_result(f"Typology errors contain: {expected_typology_errors}",
                         expected_typology_errors, typology_errors, all_found)

    return primary_errors, typology_errors


# ========== Test Cases for New Functions (lire_nombre, mesurer_hauteur, sauter_haut) ==========

def test_lire_nombre_missing():
    """Test F_CALL_MISSING_LIRE_NOMBRE"""
    incorrect_code = """
x = 5
"""
    correct_code = """
x = lire_nombre()
"""
    return run_test(
        "F_CALL_MISSING_LIRE_NOMBRE",
        incorrect_code, correct_code,
        expected_primary_substrings=["MISSING_CALL_STATEMENT", "lire_nombre"],
        expected_typology_errors={"F_CALL_MISSING_LIRE_NOMBRE"}
    )


def test_lire_nombre_unnecessary():
    """Test F_CALL_UNNECESSARY_LIRE_NOMBRE"""
    incorrect_code = """
x = lire_nombre()
"""
    correct_code = """
x = 5
"""
    return run_test(
        "F_CALL_UNNECESSARY_LIRE_NOMBRE",
        incorrect_code, correct_code,
        expected_primary_substrings=["UNNECESSARY_CALL_STATEMENT", "lire_nombre"],
        expected_typology_errors={"F_CALL_UNNECESSARY_LIRE_NOMBRE"}
    )


def test_mesurer_hauteur_missing():
    """Test F_CALL_MISSING_MESURER_HAUTEUR"""
    incorrect_code = """
h = 5
"""
    correct_code = """
h = mesurer_hauteur()
"""
    return run_test(
        "F_CALL_MISSING_MESURER_HAUTEUR",
        incorrect_code, correct_code,
        expected_primary_substrings=["MISSING_CALL_STATEMENT", "mesurer_hauteur"],
        expected_typology_errors={"F_CALL_MISSING_MESURER_HAUTEUR"}
    )


def test_mesurer_hauteur_unnecessary():
    """Test F_CALL_UNNECESSARY_MESURER_HAUTEUR"""
    incorrect_code = """
h = mesurer_hauteur()
"""
    correct_code = """
h = 5
"""
    return run_test(
        "F_CALL_UNNECESSARY_MESURER_HAUTEUR",
        incorrect_code, correct_code,
        expected_primary_substrings=["UNNECESSARY_CALL_STATEMENT", "mesurer_hauteur"],
        expected_typology_errors={"F_CALL_UNNECESSARY_MESURER_HAUTEUR"}
    )


def test_sauter_haut_missing():
    """Test F_CALL_MISSING_SAUTER_HAUT"""
    incorrect_code = """
avancer()
"""
    correct_code = """
sauter_haut()
avancer()
"""
    return run_test(
        "F_CALL_MISSING_SAUTER_HAUT",
        incorrect_code, correct_code,
        expected_primary_substrings=["MISSING_CALL_STATEMENT", "sauter_haut"],
        expected_typology_errors={"F_CALL_MISSING_SAUTER_HAUT"}
    )


def test_sauter_haut_unnecessary():
    """Test F_CALL_UNNECESSARY_SAUTER_HAUT"""
    incorrect_code = """
sauter_haut()
avancer()
"""
    correct_code = """
avancer()
"""
    return run_test(
        "F_CALL_UNNECESSARY_SAUTER_HAUT",
        incorrect_code, correct_code,
        expected_primary_substrings=["UNNECESSARY_CALL_STATEMENT", "sauter_haut"],
        expected_typology_errors={"F_CALL_UNNECESSARY_SAUTER_HAUT"}
    )


# ========== Test Cases for CS_MISSING_ELIF/ELSE ==========

def test_cs_missing_else():
    """Test CS_MISSING_ELSE: Missing else branch"""
    incorrect_code = """
if x > 0:
    print("positive")
"""
    correct_code = """
if x > 0:
    print("positive")
else:
    print("non-positive")
"""
    return run_test(
        "CS_MISSING_ELSE",
        incorrect_code, correct_code,
        expected_primary_substrings=["MISSING"],
        expected_typology_errors={"CS_MISSING_ELSE"}
    )


def test_cs_missing_elif():
    """Test CS_MISSING_ELIF: Missing elif branch"""
    incorrect_code = """
if x == 0:
    avancer()
else:
    sauter()
"""
    correct_code = """
if x == 0:
    avancer()
elif x == 1:
    sauter()
else:
    sauter_haut()
"""
    return run_test(
        "CS_MISSING_ELIF",
        incorrect_code, correct_code,
        expected_primary_substrings=["MISSING"],
        expected_typology_errors={"CS_MISSING_ELIF"}
    )


def test_cs_unnecessary_else():
    """Test CS_UNNECESSARY_ELSE: Unnecessary else branch"""
    incorrect_code = """
if x > 0:
    print("positive")
else:
    print("non-positive")
"""
    correct_code = """
if x > 0:
    print("positive")
"""
    return run_test(
        "CS_UNNECESSARY_ELSE",
        incorrect_code, correct_code,
        expected_primary_substrings=["UNNECESSARY"],
        expected_typology_errors={"CS_UNNECESSARY_ELSE"}
    )


def test_cs_unnecessary_elif():
    """Test CS_UNNECESSARY_ELIF: Unnecessary elif branch"""
    incorrect_code = """
if x == 0:
    avancer()
elif x == 1:
    sauter()
else:
    sauter_haut()
"""
    correct_code = """
if x == 0:
    avancer()
else:
    sauter_haut()
"""
    return run_test(
        "CS_UNNECESSARY_ELIF",
        incorrect_code, correct_code,
        expected_primary_substrings=["UNNECESSARY"],
        expected_typology_errors={"CS_UNNECESSARY_ELIF"}
    )


# ========== Test Cases for LO_WHILE_MISSING_UPDATE ==========

def test_while_missing_update():
    """Test LO_WHILE_MISSING_UPDATE: While loop body doesn't update condition variable"""
    incorrect_code = """
obstacle = detecter_obstacle()
while obstacle == True:
    coup()
"""
    correct_code = """
obstacle = detecter_obstacle()
while obstacle == True:
    coup()
    obstacle = detecter_obstacle()
"""
    return run_test(
        "LO_WHILE_MISSING_UPDATE",
        incorrect_code, correct_code,
        expected_primary_substrings=["MISSING", "While", "Body"],
        expected_typology_errors={"LO_WHILE_MISSING_UPDATE"}
    )


def test_while_missing_update_assignment():
    """Test LO_WHILE_MISSING_UPDATE: Missing assignment in while body"""
    incorrect_code = """
x = 0
while x < 5:
    print(x)
"""
    correct_code = """
x = 0
while x < 5:
    print(x)
    x = x + 1
"""
    return run_test(
        "LO_WHILE_MISSING_UPDATE (assignment)",
        incorrect_code, correct_code,
        expected_primary_substrings=["MISSING", "While", "Body"],
        expected_typology_errors={"LO_WHILE_MISSING_UPDATE"}
    )


# ========== Test Cases for EXP_FUNCTION_RESULT_NOT_ASSIGNED ==========

def test_function_result_not_assigned():
    """Test EXP_FUNCTION_RESULT_NOT_ASSIGNED: Function that returns value should be assigned

    This test checks when a function like lire_nombre() or detecter_obstacle()
    should be called and its result assigned, but in the incorrect code
    the assignment is missing entirely.
    """
    # Scenario: Student forgot to call and assign the function
    incorrect_code = """
avancer()
"""
    correct_code = """
x = lire_nombre()
avancer()
"""
    return run_test(
        "EXP_FUNCTION_RESULT_NOT_ASSIGNED",
        incorrect_code, correct_code,
        expected_primary_substrings=["MISSING", "lire_nombre"],
        expected_typology_errors={"EXP_FUNCTION_RESULT_NOT_ASSIGNED", "F_CALL_MISSING_LIRE_NOMBRE"}
    )


# ========== Main Ablation Study ==========

def run_ablation_study():
    """Run all tests and print summary"""
    print("\n" + "="*80)
    print("ABLATION STUDY V2: NEW ERROR TYPES DETECTION")
    print("="*80)

    tests = [
        # New function tests - lire_nombre
        test_lire_nombre_missing,
        test_lire_nombre_unnecessary,
        # mesurer_hauteur
        test_mesurer_hauteur_missing,
        test_mesurer_hauteur_unnecessary,
        # sauter_haut
        test_sauter_haut_missing,
        test_sauter_haut_unnecessary,
        # Conditional structure tests
        test_cs_missing_else,
        test_cs_missing_elif,
        test_cs_unnecessary_else,
        test_cs_unnecessary_elif,
        # While loop update tests
        test_while_missing_update,
        test_while_missing_update_assignment,
        # Function result not assigned
        test_function_result_not_assigned,
    ]

    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append((test_func.__name__, "COMPLETED", result))
        except Exception as e:
            results.append((test_func.__name__, "ERROR", str(e)))
            print(f"\n✗ ERROR in {test_func.__name__}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*80)
    print("ABLATION STUDY V2 SUMMARY")
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