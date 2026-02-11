# This file is part of ast_error_detection.
# Copyright (C) 2025 Badmavasan Kirouchenassamy & Eva Chouaki.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or any later version.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
from ast_error_detection.constants import ANNOTATION_TAG_CONST_VALUE_MISMATCH, \
    ANNOTATION_TAG_INCORRECT_OPERATION_IN_COMP, ANNOTATION_TAG_INCORRECT_OPERATION_IN_ASSIGN, \
    ANNOTATION_CONTEXT_RANGE_FUNCTION_CALL, ANNOTATION_TAG_MISSING_CALL_STATEMENT, \
    ANNOTATION_CONTEXT_CALL_NATIVE_FUNCTION_PRINT, ANNOTATION_TAG_UNNECESSARY_CALL_STATEMENT, \
    ANNOTATION_CONTEXT_FUNCTION_CALL_NODE, F_CALL_UNNECESSARY_PRINT, F_CALL_UNNECESSARY_AVANCER, \
    F_CALL_UNNECESSARY_TOURNER, ANNOTATION_CONTEXT_CALL_NATIVE_FUNCTION_PRINT_NODE_NAME, \
    ANNOTATION_CONTEXT_CALL_NATIVE_FUNCTION_AVANCER_NODE_NAME, \
    ANNOTATION_CONTEXT_CALL_NATIVE_FUNCTION_TOURNER_NODE_NAME, ANNOTATION_TAG_UNNECESSARY_FOR_LOOP, \
    ANNOTATION_TAG_UNNECESSARY_WHILE_LOOP, ANNOTATION_TAG_MISSING_FOR_LOOP
import re

### HIGH LEVEL RULES ##

# Rule: If there is an UNNECESSARY CALL statement, suppress ALL other errors
# that are *inside* that call's context (i.e., deeper in the same path).
# Keep the UNNECESSARY_CALL_STATEMENT trigger itself.
def rule_remove_inside_unnecessary_call_context():
    def rule(errors):
        # Collect exact contexts where an unnecessary call is flagged
        # Example context: "Module > Expr > Call: print"
        trigger_contexts = {
            (e[-1] or "").strip()
            for e in errors
            if e[0] == ANNOTATION_TAG_UNNECESSARY_CALL_STATEMENT
        }

        if not trigger_contexts:
            return errors
        filtered = []

        for e in errors:
            code = e[0]
            ctx  = (e[-1] or "").strip()
            # Always keep the trigger(s) themselves
            if code == ANNOTATION_TAG_UNNECESSARY_CALL_STATEMENT:
                filtered.append(e)
                continue
            # Drop errors that are strictly *under* a trigger call context:
            # i.e., context starts with "<trigger> >"
            if any(ctx.startswith(tc + " > ") for tc in trigger_contexts):
                continue
            # Everything else stays
            filtered.append(e)
        return filtered

    return rule

# Rule: If there is an UNNECESSARY FOR loop, suppress ALL other errors
# that are *inside* that FOR's context (i.e., deeper in the same path).
# Keep the UNNECESSARY_FOR_LOOP trigger itself.
def rule_remove_inside_unnecessary_loop_context():
    def rule(errors):
        # Collect exact contexts where an unnecessary FOR is flagged
        # Example context: "Module > For[0]"
        trigger_contexts = {
            (e[-1] or "").strip()
            for e in errors
            if e[0] == ANNOTATION_TAG_UNNECESSARY_FOR_LOOP or e[0] == ANNOTATION_TAG_UNNECESSARY_WHILE_LOOP
        }

        if not trigger_contexts:
            return errors

        filtered = []
        for e in errors:
            code = e[0]
            ctx  = (e[-1] or "").strip()

            # Always keep the trigger(s) themselves
            if code == ANNOTATION_TAG_UNNECESSARY_FOR_LOOP or code == ANNOTATION_TAG_UNNECESSARY_WHILE_LOOP:
                filtered.append(e)
                continue

            # Drop errors that are strictly *under* a trigger FOR context:
            # i.e., context starts with "<trigger> >"
            if any(ctx.startswith(tc + " > ") for tc in trigger_contexts):
                continue

            # Everything else stays
            filtered.append(e)

        return filtered

    return rule


# Rule: If there is a MISSING CALL statement, suppress ALL other errors
# that are *inside* that call's context (i.e., deeper in the same path).
# Keep the MISSING_CALL_STATEMENT trigger itself.
def rule_remove_inside_missing_call_context():
    def rule(errors):
        # Collect exact contexts where a missing call is flagged
        # Example context: "Module > Expr > Call: print"
        trigger_contexts = {
            (e[-1] or "").strip()
            for e in errors
            if e[0] == ANNOTATION_TAG_MISSING_CALL_STATEMENT
        }

        if not trigger_contexts:
            return errors

        filtered = []
        for e in errors:
            code = e[0]
            ctx  = (e[-1] or "").strip()

            # Always keep the trigger(s) themselves
            if code == ANNOTATION_TAG_MISSING_CALL_STATEMENT:
                filtered.append(e)
                continue

            # Drop errors that are strictly *under* a trigger call context:
            # i.e., context starts with "<trigger> >"
            if any(ctx.startswith(tc + " > ") for tc in trigger_contexts):
                continue

            # Everything else stays
            filtered.append(e)

        return filtered

    return rule

def rule_remove_inside_missing_for_loop():
    def rule(errors):
        # Collect exact contexts where a missing for-loop is flagged
        # e.g., "Module > For" or "FunctionDef: foo > For"
        trigger_contexts = {
            (e[-1] or "").strip()
            for e in errors
            if e[0] == ANNOTATION_TAG_MISSING_FOR_LOOP
        }

        if not trigger_contexts:
            return errors

        filtered = []
        for e in errors:
            code = e[0]
            ctx  = (e[-1] or "").strip()

            # Always keep the trigger(s) themselves
            if code == ANNOTATION_TAG_MISSING_FOR_LOOP:
                filtered.append(e)
                continue

            # Drop errors that are strictly *under* a trigger "for" context:
            # i.e., context starts with "<trigger> >"
            if any(ctx.startswith(tc + " > ") for tc in trigger_contexts):
                continue

            # Everything else stays
            filtered.append(e)

        return filtered

    return rule


def high_level_filtering():
    """
    Returns a function that applies a pipeline of rules to an error list.
    Default rules include:
        - Removing all 'Call: range' errors if a MISSING_CALL_STATEMENT
          on 'Call: range' is present.
    """
    rules = [
        rule_remove_inside_unnecessary_call_context(),
        rule_remove_inside_missing_call_context(),
        rule_remove_inside_unnecessary_loop_context(),
        rule_remove_inside_missing_for_loop()
    ]

    def apply_rules(errors):
        for rule in rules:
            errors = rule(errors)
        return errors

    return apply_rules


class ErrorAnnotation:

    def concatenate_all_errors(self, patterns):
        """
        Collect errors from all detection functions and concatenate them into a single list.

        Args:
            patterns (list): A list of dictionaries containing the type of operation,
                             path, current value, and new value for transformations.

        Returns:
            list: A combined list of tuples from all error detection functions.
        """

        # Call individual detection functions
        missing_statements = self.detect_specific_missing_constructs(patterns)
        unnecessary_deletions = self.detect_unnecessary_deletions(patterns)
        incorrect_positions = self.detect_incorrect_statement_positions(patterns)
        updates = self.track_all_updates(patterns)
        variable_mismatches = self.detect_variable_mismatches(patterns)


        # Combine all errors into one list
        all_errors = []
        all_errors.extend(missing_statements)
        all_errors.extend(unnecessary_deletions)
        all_errors.extend(incorrect_positions)
        all_errors.extend(updates)
        all_errors.extend(variable_mismatches)
        # High Level filtering
        error_filter = high_level_filtering()
        filtered = error_filter(all_errors)

        return filtered

    def detect_specific_missing_constructs(self, patterns):
        """
        Detect specific missing constructs in the code based on the nodes that need to be inserted,
        ensuring the node is not marked for removal elsewhere in the patterns.

        The detection focuses on node types like "Assign", "For", "While", "Call", "If", "Function",
        and "Return". Each result includes:
        - The missing construct type.
        - The value (if present) extracted from the node type after a ":".
        - The context (path) where the missing construct occurs.

        Args:
            patterns (list): A list of dictionaries containing the type of operation,
                             path, current value, and new value for transformations.

        Returns:
            list: A list of tuples in the format:
                  (missing_construct, value (or None), context_path)
        """
        missing_errors = []


        # Helper function to remove indices from path elements
        def structural_path_element(element):
            return element.split("[")[0]

        # Extract all delete nodes for comparison
        deleted_nodes = {d['current'] for d in patterns if d['type'] == 'delete'}
        updated_nodes = {u['current'] for u in patterns if u['type'] == 'update'}

        # Analyze insert operations
        for insert in patterns:
            if insert['type'] == 'insert':
                node_type_with_value = structural_path_element(insert['new']).upper()
                context_path = " > ".join(map(str, insert['path'])) if insert.get('path') else ""

                # Handle cases where ":" is present
                if ":" in node_type_with_value:
                    node_type, value = node_type_with_value.split(":", 1)
                    node_type = node_type.strip().upper()
                    value = value.strip()
                else:
                    node_type = node_type_with_value
                    value = None

                # Check if the node is also being removed elsewhere
                if insert['new'] not in deleted_nodes and insert['new'] not in updated_nodes:
                    # Determine the missing construct
                    if node_type == "FOR":
                        missing_errors.append(("MISSING_FOR_LOOP", insert['new'], context_path))
                    elif node_type == "WHILE":
                        missing_errors.append(("MISSING_WHILE_LOOP", insert['new'], context_path))
                    elif node_type == "CALL":
                        missing_errors.append(("MISSING_CALL_STATEMENT", insert['new'], context_path))
                    elif node_type == "IF":
                        missing_errors.append(("MISSING_IF_STATEMENT", insert['new'], context_path))
                    elif node_type == "ASSIGN":
                        missing_errors.append(("MISSING_ASSIGN_STATEMENT", insert['new'], context_path))
                    elif node_type == "FUNCTION":
                        missing_errors.append(("MISSING_FUNCTION_DEFINITION", insert['new'], context_path))
                    elif node_type == "RETURN":
                        missing_errors.append(("MISSING_RETURN", insert['new'], context_path))
                    elif node_type == "CONST":
                        missing_errors.append(("MISSING_CONST_VALUE", insert['new'], context_path))
                    elif node_type == "OPERATION":
                        missing_errors.append(("MISSING_OPERATION", insert['new'], context_path))
                    elif node_type == "ARG":
                        missing_errors.append(("MISSING_ARGUMENT", insert['new'], context_path))
                    elif node_type == "VAR":
                        missing_errors.append(("MISSING_VARIABLE", insert['new'], context_path))
                    elif node_type == "ELSE:":
                        # Check if this is an elif (If inside Else) or a plain else
                        if "If" in context_path:
                            missing_errors.append(("MISSING_ELSE", insert['new'], context_path))

        return list(set(missing_errors))  # Remove duplicates

    def detect_unnecessary_deletions(self, patterns):
        """
        Detect unnecessary deletions in the code based on the nodes that are marked for deletion
        but have no corresponding insert operations for the same type and value.

        The detection groups nodes into broader categories:
        - FOR_LOOP
        - WHILE_LOOP
        - FUNCTION
        - STATEMENT (covers general statements like Assign, Return, Call)
        - CONDITIONAL (covers If statements)

        Each result includes:
        - The unnecessary construct type.
        - The value (if present) extracted from the node type after a ":".
        - The context (path) where the unnecessary deletion occurs.

        Args:
            patterns (list): A list of dictionaries containing the type of operation,
                             path, current value, and new value for transformations.

        Returns:
            list: A list of tuples in the format:
                  (unnecessary_statement, value (or None), context_path)
        """
        unnecessary_errors = []

        # Helper function to remove indices from path elements
        def structural_path_element(element):
            return element.split("[")[0]

        # Extract all insert nodes for comparison
        inserted_nodes = {i['new'] for i in patterns if i['type'] == 'insert'}

        # Analyze delete operations
        for delete in patterns:
            if delete['type'] == 'delete':
                node_type_with_value = structural_path_element(delete['current']).upper()
                context_path = " > ".join(map(str, delete['path'])) if delete.get('path') else ""

                # Handle cases where ":" is present
                if ":" in node_type_with_value:
                    node_type, value = node_type_with_value.split(":", 1)
                    node_type = node_type.strip().upper()
                    value = value.strip()
                else:
                    node_type = node_type_with_value
                    value = None

                # Check if the node is also being inserted elsewhere
                if delete['current'] not in inserted_nodes:
                    # Group broader categories
                    if node_type == "FOR":
                        unnecessary_errors.append(("UNNECESSARY_FOR_LOOP", value, context_path))
                    elif node_type == "WHILE":
                        unnecessary_errors.append(("UNNECESSARY_WHILE_LOOP", value, context_path))
                    elif node_type == "FUNCTION":  # Group Function and Return
                        unnecessary_errors.append(("UNNECESSARY_FUNCTION", value, context_path))
                    elif node_type == "RETURN":
                        unnecessary_errors.append(("UNNECESSARY_RETURN_IN_FUNCTION", value, context_path))
                    elif node_type == "IF":  # Group If into Conditional
                        unnecessary_errors.append(("UNNECESSARY_CONDITIONAL", value, context_path))
                    elif node_type == "CALL":
                        unnecessary_errors.append(("UNNECESSARY_CALL_STATEMENT", value, context_path))
                    elif node_type == "ASSIGN":
                        unnecessary_errors.append(("UNNECESSARY_ASSIGN_STATEMENT", value, context_path))
                    elif node_type == "CONST":
                        unnecessary_errors.append(("UNNECESSARY_CONST_VALUE", value, context_path))
                    elif node_type == "OPERATION":
                        unnecessary_errors.append(("UNNECESSARY_OPERATION", value, context_path))
                    elif node_type == "ARG":
                        unnecessary_errors.append(("UNNECESSARY_ARGUMENT", value, context_path))
                    elif node_type == "VAR":
                        unnecessary_errors.append(("UNNECESSARY_VAR", value, context_path))
                    elif node_type == "ELSE:":
                        # Unnecessary else branch
                        if "If" in context_path:
                            unnecessary_errors.append(("UNNECESSARY_ELSE", value, context_path))

        return list(set(unnecessary_errors))  # Remove duplicates

    def detect_incorrect_statement_positions(self, patterns):
        """
        Detect nodes that are deleted and re-appear elsewhere (inserted/updated),
        inserted where they previously existed elsewhere (insert+update),
        or updated in multiple places (simultaneous updates) — all indicating
        incorrect statement positioning.

        Covered node kinds: "Assign", "For", "While", "Call", "If", "Function",
        and "Return".

        Returns a list of tuples:
          (incorrect_statement_position_code, value_or_None, context_path)
        where `context_path` points to the *new/target* location (the insert/update).
        """

        # ---- helper: strip indices from a single path element ----
        def structural_path_element(element):
            return element.split("[")[0]

        # ---- helper: derive a normalized "kind" (e.g., 'CALL', 'FOR') from a label ----
        # e.g., 'Call: print' -> 'CALL', 'For' -> 'FOR', 'Var: i' -> 'VAR'
        def node_kind(label):
            base = (label or "").split("[")[0]
            return base.split(":", 1)[0].strip().upper()

        # ---- helper: extract human-readable value after ":" if it exists ----
        # e.g., 'CALL: PRINT' -> 'PRINT', 'ASSIGN: X' -> 'X'
        def extract_value(label):
            if ":" in label:
                return label.split(":", 1)[1].strip()
            return None

        # Build sets of (KIND, FULL_LABEL_UPPER, PATH_TUPLE)
        deleted_nodes = {
            (node_kind(d['current']), (d['current'] or "").upper(), tuple(d['path']))
            for d in patterns if d['type'] == 'delete'
        }
        inserted_nodes = {
            (node_kind(i['new']), (i['new'] or "").upper(), tuple(i['path']))
            for i in patterns if i['type'] == 'insert'
        }
        # For updates we key by the *new* label (where the node ended up)
        updated_nodes = {
            (node_kind(u['new']), (u['new'] or "").upper(), tuple(u['path']))
            for u in patterns if u['type'] == 'update'
        }

        incorrect_positions = []
        kind_to_code = {
            "FOR": "INCORRECT_STATEMENT_POSITION_FOR",
            "WHILE": "INCORRECT_STATEMENT_POSITION_WHILE",
            "IF": "INCORRECT_STATEMENT_POSITION_IF",
            "CALL": "INCORRECT_STATEMENT_POSITION_CALL",
            "ASSIGN": "INCORRECT_STATEMENT_POSITION_ASSIGN",
            "FUNCTION": "INCORRECT_STATEMENT_POSITION_FUNCTION",
            "RETURN": "INCORRECT_STATEMENT_POSITION_RETURN",
        }

        # ---- helper: append a detection result given (kind, label, target_path) ----
        def append_result(kind, label_upper, target_path_tuple):
            base_kind = kind.split(":", 1)[0].strip().upper()  # kind is already normalized (CALL/FOR/…)
            code = kind_to_code.get(base_kind)
            if not code:
                return
            value = extract_value(label_upper)  # human-readable value (after ':') or None
            context_path = " > ".join(structural_path_element(p) for p in target_path_tuple)
            incorrect_positions.append((code, value, context_path))

        # ------------------------------------------------------------
        # 1) DELETE + INSERT (existing behavior): node reappears elsewhere
        #    Compare like-with-like by (KIND, FULL_LABEL_UPPER)
        #    Report context at the *insert* location.
        # ------------------------------------------------------------
        for del_kind, del_label, del_path in deleted_nodes:
            for ins_kind, ins_label, ins_path in inserted_nodes:
                if del_kind == ins_kind and del_label == ins_label:
                    append_result(del_kind, del_label, ins_path)

        # Deduplicate results
        return list(set(incorrect_positions))

    def track_all_updates(self, patterns):
        """
        Track all updates in the code and categorize them based on the node type and context.

        The detection focuses on:
        - "Call" nodes → Constant Value Mismatch
        - Comparison operations → Incorrect Operation in Condition
        - Assignment operations → Incorrect Operation in Assign
        - Loop conditions (For/While) → Incorrect Number of Iterations
        - Variable updates → Ignored for now

        Args:
            patterns (list): A list of dictionaries containing the type of operation,
                             path, current value, and new value for transformations.

        Returns:
            list: A list of tuples in the format:
                  (update_category, current_value, new_value, context_path)
        """
        updates = []

        def _node_kind(label):
            base = (label or "").split("[")[0]
            return base.split(":", 1)[0].strip().upper()

        # Helper function to remove indices from path elements
        def structural_path_element(element):
            return element.split("[")[0]

        call_token = re.compile(ANNOTATION_CONTEXT_FUNCTION_CALL_NODE, re.IGNORECASE)

        _unnecessary_map = {
            "FOR": "UNNECESSARY_FOR_LOOP",
            "WHILE": "UNNECESSARY_WHILE_LOOP",
            "IF": "UNNECESSARY_IF_STATEMENT",
            "CALL": ANNOTATION_TAG_UNNECESSARY_CALL_STATEMENT,  # keep your existing constant
            "ASSIGN": "UNNECESSARY_ASSIGN_STATEMENT",
            "FUNCTIONDEF": "UNNECESSARY_FUNCTION_DEFINITION",
            "RETURN": "UNNECESSARY_RETURN",
            "OPERATION": "UNNECESSARY_OPERATION",
            "ARGUMENT": "UNNECESSARY_ARGUMENT",
            "VAR": "UNNECESSARY_VARIABLE",
        }


        # Analyze update operations
        for update in patterns:
            if update['type'] == 'update':
                path = update['path']
                node_type = structural_path_element(path[-1]).upper()
                context_path = " > ".join(structural_path_element(p) for p in path)
                current_value = update['current']
                new_value = update['new']

                cur_kind = _node_kind(current_value)
                new_kind = _node_kind(new_value)

                if cur_kind in _unnecessary_map and new_kind and current_value != new_value:
                    updates.append((_unnecessary_map[cur_kind], current_value, new_value, context_path))

                # Handle "missing construct" cases based on the new node type
                if new_kind in (
                "FOR", "WHILE", "IF", "CALL", "ASSIGN", "FUNCTIONDEF", "RETURN", "ARGUMENT"):
                    missing_map = {
                        "FOR": "MISSING_FOR_LOOP",
                        "WHILE": "MISSING_WHILE_LOOP",
                        "IF": "MISSING_IF_STATEMENT",
                        "CALL": "MISSING_CALL_STATEMENT",
                        "ASSIGN": "MISSING_ASSIGN_STATEMENT",
                    }
                    updates.append((missing_map[new_kind], new_value, context_path))


                if "COMPARE" in node_type:
                    updates.append((ANNOTATION_TAG_INCORRECT_OPERATION_IN_COMP, current_value, new_value, context_path))
                elif "OPERATION" in node_type:
                    updates.append((ANNOTATION_TAG_INCORRECT_OPERATION_IN_ASSIGN, current_value, new_value, context_path))
                elif "CONST" in node_type:
                    updates.append((ANNOTATION_TAG_CONST_VALUE_MISMATCH, current_value, new_value, context_path))
                elif "ASSIGN" in node_type:
                    updates.append(("NODE_TYPE_MISMATCH", current_value, new_value, context_path))
                elif "VAR" in node_type:
                    continue  # Skip variable changes for now

        return updates

    def detect_variable_mismatches(self, patterns):
        """
        Detect updates involving variables (Var) and check for consistency in their values.

        If the value of a variable changes in any of the update operations, it is flagged as a mismatch.

        Args:
            patterns (list): A list of dictionaries containing the type of operation,
                             path, current value, and new value for transformations.

        Returns:
            list: A list of tuples in the format:
                  ("VARIABLE_MISMATCH", variable_name, context_path)
        """
        variable_updates = {}

        # Helper function to remove indices from path elements
        def structural_path_element(element):
            return element.split("[")[0]

        # Collect all updates involving variables
        for pattern in patterns:
            if pattern['type'] == 'update':

                # Handle cases where ":" is present
                if ": " in pattern['current']:
                    # The space is important here as in the label of the Var node its 'Var: i'
                    cur_node_type, cur_var_value = pattern['current'].split(": ", 1)

                    if ": " in pattern['new']:
                        new_node_type, new_var_value = pattern['new'].split(": ", 1)

                        # Check if the last element is a Var node
                        if cur_node_type.upper() == "VAR" and new_node_type.upper() == "VAR":
                            context_path = " > ".join(structural_path_element(p) for p in pattern['path'])

                            # Track the variable updates
                            if cur_var_value not in variable_updates:
                                variable_updates[cur_var_value] = {'values': set(), 'context_paths': set()}

                            variable_updates[cur_var_value]['values'].add(new_var_value)
                            variable_updates[cur_var_value]['context_paths'].add(context_path)

        # Check for inconsistencies
        mismatches = []
        for var_name, details in variable_updates.items():
            if len(details['values']) > 1:  # Inconsistent updates
                for context_path in details['context_paths']:
                    mismatches.append(("VARIABLE_MISMATCH", var_name, context_path))

        return mismatches



