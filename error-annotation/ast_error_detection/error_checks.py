# This file is part of ast_error_detection.
# Copyright (C) 2025 Badmavasan Kirouchenassamy & Eva Chouaki.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or any later version.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

from ast_error_detection.constants import *
import re


def process_tag_triplets(input_list, required_tags, match_start, match_end, error):
    """
    Checks if all required_tags are present in the input_list (as first element in sublists).
    If so, extracts those sublists and compares the part of their third element from match_start to match_end.
    If all match, return the dedicated error tag

    Args:
        input_list (list of lists): Each sublist must contain [tag, context1, context2].
        required_tags (set): A set of tags to look for.
        match_start (str): Start delimiter for substring match in context2.
        match_end (str): End delimiter for substring match in context2.

    Returns:
        str: Error Tag
    """
    # Filter entries with tags in required_tags
    filtered = [entry for entry in input_list if entry[0] in required_tags]

    # Check if the same tags are present in both sets
    tags_present = [entry[0] for entry in filtered]

    if not all(elem in tags_present for elem in required_tags):
        return None  # Do nothing if tags don't exactly match

    # Extract the entries matching the tags
    tag_entries = [entry for entry in filtered if entry[0] in required_tags]

    def extract_context_segment(context, start, end):
        try:
            start_index = context.index(start)
            end_index = context.index(end, start_index) + len(end)
            return context[start_index:end_index]
        except ValueError:
            return None

    # Extract segments from context2
    segments = [
        extract_context_segment(entry[2], match_start, match_end)
        for entry in tag_entries
    ]

    if all(seg == segments[0] and seg is not None for seg in segments):
        # If all segments are the same and not None, append EXP_ERR
        if match_start == ANNOTATION_CONTEXT_MODULE and match_end == ANNOTATION_CONTEXT_ASSIGN:
            return error

    return None


def get_customized_error_tags(input_list):  # new version
    """
    Analyzes a list of error details for specific tag and context patterns,
    returning a list of error code strings based on the following rules.

    Each element in the input list should be a list of either 3 or 4 elements.
    The first element is treated as the error tag and the last element as the error context.

    Rules:
        1. If the tag is "CONST_VALUE_MISMATCH" and the context contains
           "For > Condition: > Call: rang > Const", or "While > Condition: > Compare" then add:
               "LO_NUMBER_ITERATION_ERROR" OR "LO_NUMBER_ITERATION_ERROR_UNDER2"
           (Indicates a constant value mismatch in a for loop's condition. The difference being either 1 or greater.)

        2. If the tag exactly matches "MISSING_FOR_LOOP" or "MISSING_WHILE_LOOP", then add:
               "LO_FOR_MISSING" "LO_WHILE_MISSING"
           (Indicates that a for loop is missing where expected.)



        4. If the tag contains the substring "MISSING", then add:
               "MISSING_STATEMENT"
           (Indicates that a required statement is missing.)

        5. If the tag is "CONST_VALUE_MISMATCH" and the context ends with a pattern matching
           "Call: <any_text> > Const: <any_text>", then add:
               "ERROR_VALUE_PARAMETER"
           (Indicates that there is an error in the value parameter of a call.)

    Note: The context matching does not require an exact match; it is sufficient for the
    context string to contain the specified substrings or patterns.

    Args:
        input_list (list): A list of error detail lists. Each error detail list must contain
                           3 or 4 elements. The first element is the error tag and the last
                           element is the context.

    Returns:
        list: A list of error code strings that match the conditions. If no conditions match,
              an empty list is returned.
    """
    error_list = []

    # Check for Missing Assignments
    # An assignment is considered missing if the assignment operator, variable, and constant are all absent
    # simultaneously within the same context.
    # When this condition is met, the error code EXP_ERROR_ASSIGNMENT_MISSING is added to the error list.
    missing_assignment_expression_error = process_tag_triplets(input_list,
                                                               ANNOTATION_TAG_LIST_MISSING_EXPRESSION_ASSIGNMENT ,
                                                               ANNOTATION_CONTEXT_MODULE, ANNOTATION_CONTEXT_ASSIGN,
                                                               EXP_ERROR_ASSIGNMENT_MISSING)
    if missing_assignment_expression_error is not None:
        error_list.append(missing_assignment_expression_error)

    # Check for Unnecessary Assignments
    # An assignment is considered unnecessary if the assignment operator, variable, and constant
    # are present simultaneously within the same context but not required within the current context.
    # When this condition is met, the error code EXP_ERROR_ASSIGNMENT_UNNECESSARY is added to the error list.
    unnecessary_assignment_expression_error = process_tag_triplets(input_list,
                                                                   ANNOTATION_TAG_LIST_UNNECESSARY_EXPRESSION_ASSIGNMENT,
                                                                   ANNOTATION_CONTEXT_MODULE, ANNOTATION_CONTEXT_ASSIGN,
                                                                   EXP_ERROR_ASSIGNMENT_UNNECESSARY)
    if unnecessary_assignment_expression_error is not None:
        error_list.append(unnecessary_assignment_expression_error)

    for error_details in input_list:
        # Ensure the error detail has the expected number of elements; if not, skip it.
        if len(error_details) not in (3, 4):
            continue

        if len(error_details) == 3:
            tag = error_details[0]
            context = error_details[-1]
            context2 = error_details[-2]
        else:
            tag = error_details[0]
            context = error_details[-1]
            context2 = error_details[-3]

        # Check for Missing Function Calls
        # The error code F_CALL_MISSING is added to the error list whenever a MISSING_CALL_STATEMENT
        # tag is detected in the primary errors.
        if tag == ANNOTATION_TAG_MISSING_CALL_STATEMENT:
            # Check for Precise Missing Function Calls
            # A precise error code is used when the missing call is identified as a native print function.
            # If the missing call is specifically to the 'print' function, the error code
            # F_CALL_MISSING_PRINT is added to the error list.
            if context2.split(" ")[-1] == ANNOTATION_CONTEXT_CALL_NATIVE_FUNCTION_PRINT_NODE_NAME:
                error_list.append(F_CALL_MISSING_PRINT)
            elif context2.split(" ")[-1] == ANNOTATION_CONTEXT_CALL_NATIVE_FUNCTION_AVANCER_NODE_NAME:
                error_list.append(F_CALL_MISSING_AVANCER)
            elif context2.split(" ")[-1] == ANNOTATION_CONTEXT_CALL_NATIVE_FUNCTION_TOURNER_NODE_NAME:
                error_list.append(F_CALL_MISSING_TOURNER)
            elif context2.split(" ")[-1] == ANNOTATION_CONTEXT_CALL_NATIVE_FUNCTION_COULEUR_NODE_NAME:
                error_list.append(F_CALL_MISSING_COULEUR)
            elif context2.split(" ")[-1] == ANNOTATION_CONTEXT_CALL_NATIVE_FUNCTION_ARC_NODE_NAME:
                error_list.append(F_CALL_MISSING_ARC)
            elif context2.split(" ")[-1] == ANNOTATION_CONTEXT_CALL_NATIVE_FUNCTION_GAUCHE_NODE_NAME:
                error_list.append(F_CALL_MISSING_GAUCHE)
            elif context2.split(" ")[-1] == ANNOTATION_CONTEXT_CALL_NATIVE_FUNCTION_HAUT_NODE_NAME:
                error_list.append(F_CALL_MISSING_HAUT)
            elif context2.split(" ")[-1] == ANNOTATION_CONTEXT_CALL_NATIVE_FUNCTION_BAS_NODE_NAME:
                error_list.append(F_CALL_MISSING_BAS)
            elif context2.split(" ")[-1] == ANNOTATION_CONTEXT_CALL_NATIVE_FUNCTION_DROITE_NODE_NAME:
                error_list.append(F_CALL_MISSING_DROITE)
            elif context2.split(" ")[-1] == ANNOTATION_CONTEXT_CALL_NATIVE_FUNCTION_POSER_NODE_NAME:
                error_list.append(F_CALL_MISSING_POSER)
            elif context2.split(" ")[-1] == ANNOTATION_CONTEXT_CALL_NATIVE_FUNCTION_LEVER_NODE_NAME:
                error_list.append(F_CALL_MISSING_LEVER)
            elif context2.split(" ")[-1] == ANNOTATION_CONTEXT_CALL_NATIVE_FUNCTION_OUVRIR_NODE_NAME:
                error_list.append(F_CALL_MISSING_OUVRIR)
            elif context2.split(" ")[-1] == ANNOTATION_CONTEXT_CALL_NATIVE_FUNCTION_SAUTER_NODE_NAME:
                error_list.append(F_CALL_MISSING_SAUTER)
            elif context2.split(" ")[-1] == ANNOTATION_CONTEXT_CALL_NATIVE_FUNCTION_COUP_NODE_NAME:
                error_list.append(F_CALL_MISSING_COUP)
            elif context2.split(" ")[-1] == ANNOTATION_CONTEXT_CALL_NATIVE_FUNCTION_SAUTER_HAUTEUR_NODE_NAME:
                error_list.append(F_CALL_MISSING_SAUTER_HAUTEUR)
            elif context2.split(" ")[-1] == ANNOTATION_CONTEXT_CALL_NATIVE_FUNCTION_LIRE_CHAINE_NODE_NAME:
                error_list.append(F_CALL_MISSING_LIRE_CHAINE)
            elif context2.split(" ")[-1] == ANNOTATION_CONTEXT_CALL_NATIVE_FUNCTION_TIRER_NODE_NAME:
                error_list.append(F_CALL_MISSING_TIRER)
            elif context2.split(" ")[-1] == ANNOTATION_CONTEXT_CALL_NATIVE_FUNCTION_DETECTER_OBSTACLE_NODE_NAME:
                error_list.append(F_CALL_MISSING_DETECTER_OBSTACLE)
            elif context2.split(" ")[-1] == ANNOTATION_CONTEXT_CALL_NATIVE_FUNCTION_LIRE_NOMBRE_NODE_NAME:
                error_list.append(F_CALL_MISSING_LIRE_NOMBRE)
            elif context2.split(" ")[-1] == ANNOTATION_CONTEXT_CALL_NATIVE_FUNCTION_MESURER_HAUTEUR_NODE_NAME:
                error_list.append(F_CALL_MISSING_MESURER_HAUTEUR)
            elif context2.split(" ")[-1] == ANNOTATION_CONTEXT_CALL_NATIVE_FUNCTION_SAUTER_HAUT_NODE_NAME:
                error_list.append(F_CALL_MISSING_SAUTER_HAUT)
            else:
                error_list.append(F_CALL_MISSING)

        # Check for Unnecessary Function Calls
        # The error code F_CALL_UNNECESSARY is added to the error list whenever an UNNECESSARY_CALL_STATEMENT
        # tag is detected in the primary errors.
        if tag == ANNOTATION_TAG_UNNECESSARY_CALL_STATEMENT:
            # Check for Precise Unnecessary Function Calls
            # A precise error code is used when the unnecessary call is identified as a native function.
            # If the unnecessary call is specifically to the 'print' function, the error code
            # F_CALL_UNNECESSARY_PRINT is added to the error list.
            if (context2.split(" ")[-1] == ANNOTATION_CONTEXT_CALL_NATIVE_FUNCTION_PRINT_NODE_NAME
                    or re.search(ANNOTATION_CONTEXT_CALL_NATIVE_FUNCTION_PRINT, context)):
                error_list.append(F_CALL_UNNECESSARY_PRINT)
            elif (context2.split(" ")[-1] == ANNOTATION_CONTEXT_CALL_NATIVE_FUNCTION_AVANCER_NODE_NAME
                  or re.search(ANNOTATION_CONTEXT_CALL_NATIVE_FUNCTION_AVANCER, context)):
                error_list.append(F_CALL_UNNECESSARY_AVANCER)
            elif (context2.split(" ")[-1] == ANNOTATION_CONTEXT_CALL_NATIVE_FUNCTION_TOURNER_NODE_NAME
                  or re.search(ANNOTATION_CONTEXT_CALL_NATIVE_FUNCTION_TOURNER, context)):
                error_list.append(F_CALL_UNNECESSARY_TOURNER)
            elif (context2.split(" ")[-1] == ANNOTATION_CONTEXT_CALL_NATIVE_FUNCTION_COULEUR_NODE_NAME
                  or re.search(ANNOTATION_CONTEXT_CALL_NATIVE_FUNCTION_COULEUR, context)):
                error_list.append(F_CALL_UNNECESSARY_COULEUR)
            elif (context2.split(" ")[-1] == ANNOTATION_CONTEXT_CALL_NATIVE_FUNCTION_ARC_NODE_NAME
                  or re.search(ANNOTATION_CONTEXT_CALL_NATIVE_FUNCTION_ARC, context)):
                error_list.append(F_CALL_UNNECESSARY_ARC)
            elif (context2.split(" ")[-1] == ANNOTATION_CONTEXT_CALL_NATIVE_FUNCTION_GAUCHE_NODE_NAME
                  or re.search(ANNOTATION_CONTEXT_CALL_NATIVE_FUNCTION_GAUCHE, context)):
                error_list.append(F_CALL_UNNECESSARY_GAUCHE)
            elif (context2.split(" ")[-1] == ANNOTATION_CONTEXT_CALL_NATIVE_FUNCTION_HAUT_NODE_NAME
                  or re.search(ANNOTATION_CONTEXT_CALL_NATIVE_FUNCTION_HAUT, context)):
                error_list.append(F_CALL_UNNECESSARY_HAUT)
            elif (context2.split(" ")[-1] == ANNOTATION_CONTEXT_CALL_NATIVE_FUNCTION_DROITE_NODE_NAME
                  or re.search(ANNOTATION_CONTEXT_CALL_NATIVE_FUNCTION_DROITE, context)):
                error_list.append(F_CALL_UNNECESSARY_DROITE)
            elif (context2.split(" ")[-1] == ANNOTATION_CONTEXT_CALL_NATIVE_FUNCTION_BAS_NODE_NAME
                  or re.search(ANNOTATION_CONTEXT_CALL_NATIVE_FUNCTION_BAS, context)):
                error_list.append(F_CALL_UNNECESSARY_BAS)
            elif (context2.split(" ")[-1] == ANNOTATION_CONTEXT_CALL_NATIVE_FUNCTION_LEVER_NODE_NAME
                  or re.search(ANNOTATION_CONTEXT_CALL_NATIVE_FUNCTION_LEVER, context)):
                error_list.append(F_CALL_UNNECESSARY_LEVER)
            elif (context2.split(" ")[-1] == ANNOTATION_CONTEXT_CALL_NATIVE_FUNCTION_POSER_NODE_NAME
                  or re.search(ANNOTATION_CONTEXT_CALL_NATIVE_FUNCTION_POSER, context)):
                error_list.append(F_CALL_UNNECESSARY_POSER)
            elif (context2.split(" ")[-1] == ANNOTATION_CONTEXT_CALL_NATIVE_FUNCTION_OUVRIR_NODE_NAME
                  or re.search(ANNOTATION_CONTEXT_CALL_NATIVE_FUNCTION_OUVRIR, context)):
                error_list.append(F_CALL_UNNECESSARY_OUVRIR)
            elif (context2.split(" ")[-1] == ANNOTATION_CONTEXT_CALL_NATIVE_FUNCTION_SAUTER_NODE_NAME
                  or re.search(ANNOTATION_CONTEXT_CALL_NATIVE_FUNCTION_SAUTER, context)):
                error_list.append(F_CALL_UNNECESSARY_SAUTER)
            elif (context2.split(" ")[-1] == ANNOTATION_CONTEXT_CALL_NATIVE_FUNCTION_COUP_NODE_NAME
                  or re.search(ANNOTATION_CONTEXT_CALL_NATIVE_FUNCTION_COUP, context)):
                error_list.append(F_CALL_UNNECESSARY_COUP)
            elif (context2.split(" ")[-1] == ANNOTATION_CONTEXT_CALL_NATIVE_FUNCTION_SAUTER_HAUTEUR_NODE_NAME
                  or re.search(ANNOTATION_CONTEXT_CALL_NATIVE_FUNCTION_SAUTER_HAUTEUR, context)):
                error_list.append(F_CALL_UNNECESSARY_SAUTER_HAUTEUR)
            elif (context2.split(" ")[-1] == ANNOTATION_CONTEXT_CALL_NATIVE_FUNCTION_LIRE_CHAINE_NODE_NAME
                  or re.search(ANNOTATION_CONTEXT_CALL_NATIVE_FUNCTION_LIRE_CHAINE, context)):
                error_list.append(F_CALL_UNNECESSARY_LIRE_CHAINE)
            elif (context2.split(" ")[-1] == ANNOTATION_CONTEXT_CALL_NATIVE_FUNCTION_TIRER_NODE_NAME
                  or re.search(ANNOTATION_CONTEXT_CALL_NATIVE_FUNCTION_TIRER, context)):
                error_list.append(F_CALL_UNNECESSARY_TIRER)
            elif (context2.split(" ")[-1] == ANNOTATION_CONTEXT_CALL_NATIVE_FUNCTION_DETECTER_OBSTACLE_NODE_NAME
                  or re.search(ANNOTATION_CONTEXT_CALL_NATIVE_FUNCTION_DETECTER_OBSTACLE, context)):
                error_list.append(F_CALL_UNNECESSARY_DETECTER_OBSTACLE)
            elif (context2.split(" ")[-1] == ANNOTATION_CONTEXT_CALL_NATIVE_FUNCTION_LIRE_NOMBRE_NODE_NAME
                  or re.search(ANNOTATION_CONTEXT_CALL_NATIVE_FUNCTION_LIRE_NOMBRE, context)):
                error_list.append(F_CALL_UNNECESSARY_LIRE_NOMBRE)
            elif (context2.split(" ")[-1] == ANNOTATION_CONTEXT_CALL_NATIVE_FUNCTION_MESURER_HAUTEUR_NODE_NAME
                  or re.search(ANNOTATION_CONTEXT_CALL_NATIVE_FUNCTION_MESURER_HAUTEUR, context)):
                error_list.append(F_CALL_UNNECESSARY_MESURER_HAUTEUR)
            elif (context2.split(" ")[-1] == ANNOTATION_CONTEXT_CALL_NATIVE_FUNCTION_SAUTER_HAUT_NODE_NAME
                  or re.search(ANNOTATION_CONTEXT_CALL_NATIVE_FUNCTION_SAUTER_HAUT, context)):
                error_list.append(F_CALL_UNNECESSARY_SAUTER_HAUT)
            else:
                error_list.append(F_CALL_UNNECESSARY)

        # Check for Errors Inside Print Function Calls
        # The error code F_CALL_PRINT_ERROR_ARG is added to the error list whenever an error is detected
        # inside the arguments of a 'print' function call.
        # The type of error originates from the primary errors and can be of any kind.
        if tag not in F_CALL_PRINT_ERROR_ARG_EXCEPTION_ANNOTATION_TAGS and re.search(ANNOTATION_CONTEXT_CALL_NATIVE_FUNCTION_PRINT_ARG, context):
            error_list.append(F_CALL_PRINT_ERROR_ARG)

        if tag not in F_CALL_DESIGN_ERROR_ARG_EXCEPTION_ANNOTATION_TAGS and re.search(ANNOTATION_CONTEXT_CALL_NATIVE_FUNCTION_AVANCER_ARG, context):
            error_list.append(F_CALL_AVANCER_ERROR)

        if tag not in F_CALL_DESIGN_ERROR_ARG_EXCEPTION_ANNOTATION_TAGS and re.search(ANNOTATION_CONTEXT_CALL_NATIVE_FUNCTION_TOURNER_ARG, context):
            error_list.append(F_CALL_TOURNER_ERROR)

        if tag not in F_CALL_DESIGN_ERROR_ARG_EXCEPTION_ANNOTATION_TAGS and re.search(ANNOTATION_CONTEXT_CALL_NATIVE_FUNCTION_COULEUR_ARG, context):
            error_list.append(F_CALL_COULEUR_ERROR)

        if tag not in F_CALL_DESIGN_ERROR_ARG_EXCEPTION_ANNOTATION_TAGS and re.search(ANNOTATION_CONTEXT_CALL_NATIVE_FUNCTION_ARC_ARG, context):
            error_list.append(F_CALL_ARC_ERROR)

        if tag not in F_CALL_ROBOT_ERROR_ARG_EXCEPTION_ANNOTATION_TAGS and re.search(ANNOTATION_CONTEXT_CALL_NATIVE_FUNCTION_GAUCHE_ARG, context):
            error_list.append(F_CALL_GAUCHE_ERROR)

        if tag not in F_CALL_ROBOT_ERROR_ARG_EXCEPTION_ANNOTATION_TAGS and re.search(ANNOTATION_CONTEXT_CALL_NATIVE_FUNCTION_HAUT_ARG, context):
            error_list.append(F_CALL_HAUT_ERROR)

        if tag not in F_CALL_ROBOT_ERROR_ARG_EXCEPTION_ANNOTATION_TAGS and re.search(ANNOTATION_CONTEXT_CALL_NATIVE_FUNCTION_DROITE_ARG, context):
            error_list.append(F_CALL_DROITE_ERROR)

        if tag not in F_CALL_ROBOT_ERROR_ARG_EXCEPTION_ANNOTATION_TAGS and re.search(ANNOTATION_CONTEXT_CALL_NATIVE_FUNCTION_BAS_ARG, context):
            error_list.append(F_CALL_BAS_ERROR)

        if tag not in F_CALL_DESIGN_ERROR_ARG_EXCEPTION_ANNOTATION_TAGS and re.search(ANNOTATION_CONTEXT_CALL_NATIVE_FUNCTION_OUVRIR_ARG, context):
            error_list.append(F_CALL_OUVRIR_ERROR)

        if tag not in F_CALL_DESIGN_ERROR_ARG_EXCEPTION_ANNOTATION_TAGS and re.search(ANNOTATION_CONTEXT_CALL_NATIVE_FUNCTION_SAUTER_ARG, context):
            error_list.append(F_CALL_SAUTER_ERROR)

        if tag not in F_CALL_DESIGN_ERROR_ARG_EXCEPTION_ANNOTATION_TAGS and re.search(ANNOTATION_CONTEXT_CALL_NATIVE_FUNCTION_COUP_ARG, context):
            error_list.append(F_CALL_COUP_ERROR)

        if tag not in F_CALL_DESIGN_ERROR_ARG_EXCEPTION_ANNOTATION_TAGS and re.search(ANNOTATION_CONTEXT_CALL_NATIVE_FUNCTION_SAUTER_HAUTEUR_ARG, context):
            error_list.append(F_CALL_SAUTER_HAUTEUR_ERROR)

        if tag not in F_CALL_DESIGN_ERROR_ARG_EXCEPTION_ANNOTATION_TAGS and re.search(ANNOTATION_CONTEXT_CALL_NATIVE_FUNCTION_LIRE_CHAINE_ARG, context):
            error_list.append(F_CALL_LIRE_CHAINE_ERROR)

        if tag not in F_CALL_DESIGN_ERROR_ARG_EXCEPTION_ANNOTATION_TAGS and re.search(ANNOTATION_CONTEXT_CALL_NATIVE_FUNCTION_TIRER_ARG, context):
            error_list.append(F_CALL_TIRER_ERROR)

        if tag not in F_CALL_DESIGN_ERROR_ARG_EXCEPTION_ANNOTATION_TAGS and re.search(ANNOTATION_CONTEXT_CALL_NATIVE_FUNCTION_DETECTER_OBSTACLE_ARG, context):
            error_list.append(F_CALL_DETECTER_OBSTACLE_ERROR)

        if tag not in F_CALL_DESIGN_ERROR_ARG_EXCEPTION_ANNOTATION_TAGS and re.search(ANNOTATION_CONTEXT_CALL_NATIVE_FUNCTION_LIRE_NOMBRE_ARG, context):
            error_list.append(F_CALL_LIRE_NOMBRE_ERROR)

        if tag not in F_CALL_DESIGN_ERROR_ARG_EXCEPTION_ANNOTATION_TAGS and re.search(ANNOTATION_CONTEXT_CALL_NATIVE_FUNCTION_MESURER_HAUTEUR_ARG, context):
            error_list.append(F_CALL_MESURER_HAUTEUR_ERROR)

        if tag not in F_CALL_DESIGN_ERROR_ARG_EXCEPTION_ANNOTATION_TAGS and re.search(ANNOTATION_CONTEXT_CALL_NATIVE_FUNCTION_SAUTER_HAUT_ARG, context):
            error_list.append(F_CALL_SAUTER_HAUT_ERROR)

        # Check for Errors Related to Invalid Operations
        # The error code EXP_ERROR_OPERATION is added to the error list whenever an error is detected
        # and the error context contains "Operation:".
        # This indicates there is a problem with an operation, such as:
        #   - Variable-to-variable operations (e.g., a + b)
        #   - Literal-to-literal operations (e.g., 1 + 2)
        #   - Mixed variable-literal operations (e.g., a + 1)
        if tag == ANNOTATION_TAG_CONST_VALUE_MISMATCH and re.search(ANNOTATION_CONTEXT_OPERATION, context):
            error_list.append(EXP_ERROR_OPERATION)

        # Check for Incorrect Position of 'print' Function Calls
        # The error code F_CALL_INCORRECT_POSITION_PRINT is added to the error list whenever
        # the annotation tag indicates print is not called in the right position in the code
        # and the error context matches a native 'print' function call.
        # This ensures that misplaced 'print' calls are flagged separately from other call errors.
        incorrect_position_tags = [
            (ANNOTATION_CONTEXT_CALL_NATIVE_FUNCTION_PRINT, F_CALL_INCORRECT_POSITION_PRINT),
            (ANNOTATION_CONTEXT_CALL_NATIVE_FUNCTION_HAUT, F_CALL_INCORRECT_POSITION_HAUT),
            (ANNOTATION_CONTEXT_CALL_NATIVE_FUNCTION_BAS, F_CALL_INCORRECT_POSITION_BAS),
            (ANNOTATION_CONTEXT_CALL_NATIVE_FUNCTION_GAUCHE, F_CALL_INCORRECT_POSITION_GAUCHE),
            (ANNOTATION_CONTEXT_CALL_NATIVE_FUNCTION_DROITE, F_CALL_INCORRECT_POSITION_DROITE),
            (ANNOTATION_CONTEXT_CALL_NATIVE_FUNCTION_TOURNER, F_CALL_INCORRECT_POSITION_TOURNER),
            (ANNOTATION_CONTEXT_CALL_NATIVE_FUNCTION_AVANCER, F_CALL_INCORRECT_POSITION_AVANCER),
            (ANNOTATION_CONTEXT_CALL_NATIVE_FUNCTION_LEVER, F_CALL_INCORRECT_POSITION_POSER),
            (ANNOTATION_CONTEXT_CALL_NATIVE_FUNCTION_POSER, F_CALL_INCORRECT_POSITION_LEVER),
            (ANNOTATION_CONTEXT_CALL_NATIVE_FUNCTION_ARC, F_CALL_INCORRECT_POSITION_ARC),
            (ANNOTATION_CONTEXT_CALL_NATIVE_FUNCTION_OUVRIR, F_CALL_INCORRECT_POSITION_OUVRIR),
            (ANNOTATION_CONTEXT_CALL_NATIVE_FUNCTION_SAUTER, F_CALL_INCORRECT_POSITION_SAUTER),
            (ANNOTATION_CONTEXT_CALL_NATIVE_FUNCTION_COUP, F_CALL_INCORRECT_POSITION_COUP),
            (ANNOTATION_CONTEXT_CALL_NATIVE_FUNCTION_SAUTER_HAUTEUR, F_CALL_INCORRECT_POSITION_SAUTER_HAUTEUR),
            (ANNOTATION_CONTEXT_CALL_NATIVE_FUNCTION_LIRE_CHAINE, F_CALL_INCORRECT_POSITION_LIRE_CHAINE),
            (ANNOTATION_CONTEXT_CALL_NATIVE_FUNCTION_TIRER, F_CALL_INCORRECT_POSITION_TIRER),
            (ANNOTATION_CONTEXT_CALL_NATIVE_FUNCTION_DETECTER_OBSTACLE, F_CALL_INCORRECT_POSITION_DETECTER_OBSTACLE),
            (ANNOTATION_CONTEXT_CALL_NATIVE_FUNCTION_LIRE_NOMBRE, F_CALL_INCORRECT_POSITION_LIRE_NOMBRE),
            (ANNOTATION_CONTEXT_CALL_NATIVE_FUNCTION_MESURER_HAUTEUR, F_CALL_INCORRECT_POSITION_MESURER_HAUTEUR),
            (ANNOTATION_CONTEXT_CALL_NATIVE_FUNCTION_SAUTER_HAUT, F_CALL_INCORRECT_POSITION_SAUTER_HAUT)
        ]

        for pattern, error_tag in incorrect_position_tags:
            if tag == ANNOTATION_TAG_INCORRECT_POSITION_CALL and re.search(pattern, context):
                error_list.append(error_tag)

        if tag == ANNOTATION_TAG_INCORRECT_POSITION_ASSIGN:
            error_list.append(EXP_ERROR_ASSIGNMENT_MISPLACED)

        if tag == ANNOTATION_TAG_INCORRECT_POSITION_FOR:
            error_list.append(LO_FOR_MISPLACED)

        # WHILE_MISPLACED: When the while structure is there but misplaced in terms of order in the code
        if tag == ANNOTATION_TAG_INCORRECT_POSITION_WHILE:
            error_list.append(LO_WHILE_MISPLACED)

        if tag == ANNOTATION_TAG_UNNECESSARY_FOR_LOOP and (re.search(ANNOTATION_CONTEXT_FOR_LOOP, context) or (context2 and context2.split(" ")[-1] == ANNOTATION_CONTEXT_FOR_NODE_NAME)):
            error_list.append(LO_FOR_UNNECESSARY)

        if tag == ANNOTATION_TAG_UNNECESSARY_WHILE_LOOP and (re.search(ANNOTATION_CONTEXT_WHILE_LOOP, context) or context2.split(" ")[-1] == ANNOTATION_CONTEXT_WHILE_NODE_NAME):
            error_list.append(LO_WHILE_UNNECESSARY)

        if tag == ANNOTATION_TAG_UNNECESSARY_FUNCTION_DEFINITION:
            error_list.append(F_DEFINITION_UNNECESSARY)


        # ITERATION ERROR
        if tag == ANNOTATION_TAG_CONST_VALUE_MISMATCH and "For > Condition: > Call: range > Const" in context:
            number1 = int(context.split(" ")[-1])
            number2 = int(error_details[2].split(" ")[-1]) if len(error_details) == 4 else int(context2.split(" ")[-1])
            if abs(number1 - number2) > 1:
                error_list.append(LO_FOR_NUMBER_ITERATION_ERROR)
            else:
                error_list.append(LO_FOR_NUMBER_ITERATION_ERROR_UNDER2)
        if tag == ANNOTATION_TAG_CONST_VALUE_MISMATCH and "While > Condition: > Compare" in context:
            number1 = int(context.split(" ")[-1])
            number2 = int(context2.split(" ")[-1])
            if abs(number1 - number2) > 1:
                error_list.append(LO_WHILE_NUMBER_ITERATION_ERROR)
            else:
                error_list.append(LO_WHILE_NUMBER_ITERATION_ERROR_UNDER2)

        if ANNOTATION_TAG_INCORRECT_POSITION in tag and ANNOTATION_CONTEXT_FOR_LOOP_BODY in context:
            error_list.append(LO_BODY_MISPLACED)

        # BODY MISSING
        if "INCORRECT_STATEMENT_POSITION" in tag and ANNOTATION_CONTEXT_FOR_LOOP_BODY in context:
            error_list.append(LO_BODY_MISPLACED)
        if ANNOTATION_TAG_MISSING in tag and (
                ANNOTATION_CONTEXT_FOR_LOOP_BODY in context or ANNOTATION_CONTEXT_WHILE_LOOP_BODY in context):
            error_list.append(LO_BODY_MISSING_NOT_PRESENT_ANYWHERE)

        # WHILE (a retirer par la suite)
        if tag == ANNOTATION_TAG_INCORRECT_OPERATION_IN_COMP and ANNOTATION_CONTEXT_WHILE_LOOP_CONDITION in context:
            error_list.append(LO_CONDITION_ERROR)

        # WHILE_STOP_CONDITION_ERROR: When the stop condition of the expected correct code does not match the incorrect code
        # This captures cases where the comparison operator or the condition itself is wrong in a while loop
        if (tag == ANNOTATION_TAG_INCORRECT_OPERATION_IN_COMP or tag == ANNOTATION_TAG_CONST_VALUE_MISMATCH) and ANNOTATION_CONTEXT_WHILE_LOOP_CONDITION in context:
            error_list.append(WHILE_STOP_CONDITION_ERROR)

        # CS_UNNECESSARY: There is an unnecessary conditional statement in the incorrect code compared to the correct code
        if tag == ANNOTATION_TAG_UNNECESSARY_CS:
            error_list.append(CS_UNNECESSARY)

        # MISSING LOOP OR CS OR FUNCTION
        if tag == ANNOTATION_TAG_MISSING_FOR_LOOP:
            error_list.append(LO_FOR_MISSING)

        if tag == ANNOTATION_TAG_MISSING_WHILE_LOOP:
            error_list.append(LO_WHILE_MISSING)

        if tag == ANNOTATION_TAG_MISSING_CS:
            error_list.append(CS_MISSING)

        if tag == ANNOTATION_TAG_MISSING_FUNCTION_DEFINITION:
            error_list.append(F_DEFINITION_MISSING)

        # CS : error 2 : body error or body missing
        # CS_BODY_ERROR: There is an error inside the body of a conditional branch
        # This includes missing statements, incorrect values, or any other error within an if body
        # Note: Context can be "If > Body" or "If > Body:" (with colon for structure markers)
        if ANNOTATION_TAG_MISSING in tag and (ANNOTATION_CONTEXT_CS_BODY in context or "If > Body:" in context or ("If" in context and "> Body" in context)):
            error_list.append(CS_BODY_ERROR)
        if tag == ANNOTATION_TAG_CONST_VALUE_MISMATCH and (ANNOTATION_CONTEXT_CS_BODY in context or "If > Body:" in context or ("If" in context and "> Body" in context)):
            error_list.append(CS_BODY_ERROR)
        if tag == ANNOTATION_TAG_UNNECESSARY_CALL_STATEMENT and (ANNOTATION_CONTEXT_CS_BODY in context or "If > Body:" in context or ("If" in context and "> Body" in context)):
            error_list.append(CS_BODY_ERROR)
        if tag == ANNOTATION_TAG_INCORRECT_OPERATION_IN_ASSIGN and (ANNOTATION_CONTEXT_CS_BODY in context or "If > Body:" in context or ("If" in context and "> Body" in context)):
            error_list.append(CS_BODY_ERROR)

        # CS : error 3 : body_misplaced
        if tag == ANNOTATION_TAG_INCORRECT_POSITION_CS:
            error_list.append(CS_BODY_MISPLACED)

        # Error : VA_DECLARATION_INITIALIZATION_ERROR
        # Check for Variable Initialization Errors
        # The error code VA_DECLARATION_INITIALIZATION_ERROR is added to the error list whenever
        # the annotation tag indicates a constant mismatch (CONST_VALUE_MISMATCH)
        # and the error context matches a variable declaration (ANNOTATION_CONTEXT_VAR).
        # This occurs when a variable is initialized with a value,
        # but the value used for initialization is incorrect.
        if tag == VAR_CONST_MISMATCH and re.search(ANNOTATION_CONTEXT_VAR, context):
            error_list.append(VA_DECLARATION_INITIALIZATION_ERROR)

        # FUNCTION : error 2 : definition error return
        if tag == ANNOTATION_TAG_MISSING_RETURN or tag == ANNOTATION_TAG_UNNECESSARY_RETURN or (
                tag == ANNOTATION_TAG_MISSING_VARIABLE and ANNOTATION_CONTEXT_RETURN_1 in context and ANNOTATION_CONTEXT_RETURN_2 in context):
            error_list.append(F_DEFINITION_ERROR_RETURN)

        # EXP : error 1 : error conditional branch
        if tag == ANNOTATION_TAG_INCORRECT_OPERATION_IN_COMP and ANNOTATION_CONTEXT_CS_CONDITION in context:
            error_list.append(EXP_ERROR_CONDITIONAL_BRANCH)

        # EXP_ERROR_ASSIGNMENT: When a value or an output of a function is assigned to a variable
        # and if there is any error related to that, this error tag should be flagged.
        # This captures errors in assignment statements where:
        # - A function call result is assigned incorrectly
        # - A constant value is assigned incorrectly
        # - The variable being assigned to is wrong
        if ANNOTATION_CONTEXT_ASSIGN in context:
            if tag == ANNOTATION_TAG_CONST_VALUE_MISMATCH:
                error_list.append(EXP_ERROR_ASSIGNMENT)
            if tag == ANNOTATION_TAG_INCORRECT_FUNCTION_CALL:
                error_list.append(EXP_ERROR_ASSIGNMENT)
            if tag == ANNOTATION_TAG_MISSING_CALL_STATEMENT:
                error_list.append(EXP_ERROR_ASSIGNMENT)
            if tag == ANNOTATION_TAG_UNNECESSARY_CALL_STATEMENT:
                error_list.append(EXP_ERROR_ASSIGNMENT)

        # EXP_ERROR_OPERATION: If a result of an operation is assigned to a variable
        # and if there is a problem with that operation, this error must be tagged.
        # Improved detection: capture operation errors in assignment context
        if tag == ANNOTATION_TAG_INCORRECT_OPERATION_IN_ASSIGN:
            error_list.append(EXP_ERROR_OPERATION)
        if tag == ANNOTATION_TAG_MISSING_OPERATION and ANNOTATION_CONTEXT_ASSIGN in context:
            error_list.append(EXP_ERROR_OPERATION)
        if tag == ANNOTATION_TAG_UNNECESSARY_OPERATION and ANNOTATION_CONTEXT_ASSIGN in context:
            error_list.append(EXP_ERROR_OPERATION)

        # CS_MISSING_ELIF: Missing elif branch (an If inside an Else context)
        # When a MISSING_IF_STATEMENT is detected inside an Else context, it means
        # an elif branch is missing
        if tag == ANNOTATION_TAG_MISSING_CS and "Else:" in context:
            error_list.append(CS_MISSING_ELIF)

        # CS_MISSING_ELSE: Missing else branch
        # Detected when MISSING_ELSE tag is present or when there's a missing statement
        # inside an Else context (indicating the else branch content is missing)
        if tag == ANNOTATION_TAG_MISSING_ELSE:
            error_list.append(CS_MISSING_ELSE)
        # Also detect when statements are missing inside an Else context
        if ANNOTATION_TAG_MISSING in tag and "Else:" in context and "If" in context:
            error_list.append(CS_MISSING_ELSE)

        # CS_UNNECESSARY_ELIF: Unnecessary elif branch
        # When an UNNECESSARY_CONDITIONAL is detected inside an Else context
        if tag == ANNOTATION_TAG_UNNECESSARY_CS and "Else:" in context:
            error_list.append(CS_UNNECESSARY_ELIF)

        # CS_UNNECESSARY_ELSE: Unnecessary else branch
        # Detected when UNNECESSARY_ELSE tag is present or when there's an unnecessary
        # statement inside an Else context
        if tag == ANNOTATION_TAG_UNNECESSARY_ELSE:
            error_list.append(CS_UNNECESSARY_ELSE)
        # Also detect when statements are unnecessary inside an Else context
        if "UNNECESSARY" in tag and "Else:" in context and "If" in context and tag != ANNOTATION_TAG_UNNECESSARY_CS:
            error_list.append(CS_UNNECESSARY_ELSE)

        # LO_WHILE_MISSING_UPDATE: While loop body doesn't update the condition variable
        # This is detected when there's a missing assignment inside a while body
        # that could update the loop condition variable
        if "MISSING_ASSIGN" in tag and (ANNOTATION_CONTEXT_WHILE_LOOP_BODY in context or "While > Body:" in context or ("While" in context and "> Body" in context)):
            error_list.append(LO_WHILE_MISSING_UPDATE)
        if tag == ANNOTATION_TAG_MISSING_CALL_STATEMENT and (ANNOTATION_CONTEXT_WHILE_LOOP_BODY in context or "While > Body:" in context or ("While" in context and "> Body" in context)):
            # A missing call inside while body could be an update function
            error_list.append(LO_WHILE_MISSING_UPDATE)

        # EXP_FUNCTION_RESULT_NOT_ASSIGNED: Function that returns a value is called
        # but the result is not assigned to a variable
        # This is detected when an assignment with a function call is missing
        # (i.e., the student called the function but didn't assign the result)
        # Detection approach: Look for MISSING_ASSIGN_STATEMENT that could involve
        # a function returning a value
        if "MISSING_ASSIGN" in tag:
            # Check if a function that returns a value is mentioned anywhere in the error
            for func in FUNCTIONS_RETURNING_VALUES:
                if (context2 and func in context2.lower()) or func in context.lower():
                    error_list.append(EXP_FUNCTION_RESULT_NOT_ASSIGNED)
                    break
            # Also flag when there's a missing assignment and we have a missing call
            # that returns a value in the input list (check other errors)
        if tag == ANNOTATION_TAG_MISSING_CALL_STATEMENT and context2:
            func_name = context2.split(" ")[-1].lower() if " " in context2 else context2.lower()
            func_name = func_name.replace("call:", "").strip()
            if func_name in FUNCTIONS_RETURNING_VALUES:
                error_list.append(EXP_FUNCTION_RESULT_NOT_ASSIGNED)

        """
            SPECIFIC CODE SECTION
        """

        rules = [
            (ANNOTATION_TAG_MISSING_CONST_VALUE, ANNOTATION_CONTEXT_FOR_LOOP_BODY),
            (ANNOTATION_TAG_MISSING_CALL_STATEMENT, ANNOTATION_CONTEXT_FOR_LOOP_BODY ),
            (ANNOTATION_TAG_UNNECESSARY_CALL_STATEMENT, ANNOTATION_CONTEXT_FOR_LOOP_BODY),
            (ANNOTATION_TAG_CONST_VALUE_MISMATCH, ANNOTATION_CONTEXT_WHILE_LOOP_BODY),
            (ANNOTATION_TAG_INCORRECT_POSITION_ASSIGN, ANNOTATION_CONTEXT_FOR_LOOP_BODY),
        ]

        for rule_tag, rule_context in rules:
            if tag == rule_tag and rule_context in context:
                error_list.append(LO_BODY_ERROR)

        """
            TRANSLATION OF ABOVE CODE

            if tag == ANNOTATION_TAG_UNNECESSARY_CALL_STATEMENT and ANNOTATION_CONTEXT_FOR_LOOP_BODY in context:
                error_list.append(LO_BODY_ERROR)

            if tag == ANNOTATION_TAG_CONST_VALUE_MISMATCH and ANNOTATION_CONTEXT_WHILE_LOOP_BODY in context:
                error_list.append(LO_BODY_ERROR)

            if tag == ANNOTATION_TAG_INCORRECT_POSITION_ASSIGN and ANNOTATION_CONTEXT_FOR_LOOP_BODY in context :
                error_list.append(LO_BODY_ERROR)
        """

        '''
        # Rule 4: Tag contains "MISSING".
        if ANNOTATION_TGA_MISSING in tag and tag != ANNOTATION_TAG_MISSING_FOR_LOOP:#not in [ANNOTATION_TAG_MISSING_FOR_LOOP, ANNOTATION_TAG_MISSING_WHILE_LOOP, ANNOTATION_TAG_MISSING_CS, ANNOTATION_CONTEXT_FOR_LOOP_BODY]:
            error_list.append(MISSING_STATEMENT)

        # Rule 5: CONST_VALUE_MISMATCH with context ending with the specified pattern.
        if tag == ANNOTATION_TAG_CONST_VALUE_MISMATCH and pattern_value_parameter.search(context):
            error_list.append(ERROR_VALUE_PARAMETER)
        '''

    return set(error_list)
