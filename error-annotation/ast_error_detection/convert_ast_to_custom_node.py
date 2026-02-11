# This file is part of ast_error_detection.
# Copyright (C) 2025 Badmavasan Kirouchenassamy & Eva Chouaki.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or any later version.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import ast
from .node import Node


def handle_comparison(comparison_node):
    # Mapping of AST comparison types to their string representation.
    comparison_ops = {
        ast.Gt: '>',
        ast.Lt: '<',
        ast.LtE: '<=',
        ast.GtE: '>=',
        ast.Eq: '==',
        ast.NotEq: '!=',
        ast.In: 'In',
        ast.NotIn: 'Not in'
    }

    operator = comparison_node.ops[0]
    operator_label = comparison_ops.get(type(operator), '')

    left_nodes = ast_to_custom_node(comparison_node.left)
    right_nodes = ast_to_custom_node(comparison_node.comparators[0])

    condition_children = left_nodes + right_nodes
    node_label = (
        f"Compare: {operator_label}"
        if operator_label not in ('In', 'Not in')
        else f"Call: {operator_label}"
    )

    return [Node(node_label, children=condition_children)]


def process_child_nodes(ast_node):
    children = []
    for child in ast.iter_child_nodes(ast_node):
        child_nodes = ast_to_custom_node(child)
        if child_nodes:
            children.extend(child_nodes)
    return children


def ast_to_custom_node(ast_node):
    """
    Transforms an AST node into a Tree using Node

    Returns:
        Node: Tree object representing ast_node
    """
    node_type = type(ast_node).__name__

    # Skip 'Load' or 'Store' node types.
    if node_type in {'Load', 'Store'}:
        return []

    # Handle 'Expr' nodes.
    elif node_type == 'Expr':
        return process_child_nodes(ast_node)

    # Handle 'UnaryOp' nodes.
    elif node_type == 'UnaryOp':
        if isinstance(ast_node.op, ast.USub):
            if isinstance(ast_node.operand, ast.Constant):
                return [Node(f"Const: -{ast_node.operand.value}")]
            elif isinstance(ast_node.operand, ast.Name):
                return [Node(f"Var: -{ast_node.operand.id}")]
        return []

    # Handle function definitions.
    elif isinstance(ast_node, ast.FunctionDef):
        children = process_child_nodes(ast_node)
        return [Node(f"Function: {ast_node.name}", children=children)]

    # Handle function arguments.
    elif isinstance(ast_node, ast.arg):
        return [Node(f"Arg: {ast_node.arg}")]

    # Handle function argument lists.
    elif isinstance(ast_node, ast.arguments):
        children = process_child_nodes(ast_node)
        if children:
            return [Node("arguments", children=children)]
        return []

    # Handle binary operations and augmented assignments.
    elif isinstance(ast_node, (ast.BinOp, ast.AugAssign)):
        op_labels = {
            ast.Mod: '%',
            ast.Add: '+',
            ast.Sub: '-',
            ast.Mult: '*',
            ast.Div: '/',
            ast.FloorDiv: '//',
            ast.Pow: '**',
        }
        op_type = type(ast_node.op)
        op_label = op_labels.get(op_type, 'Operation')

        children = []
        if isinstance(ast_node, ast.BinOp):
            left_nodes = ast_to_custom_node(ast_node.left)
            right_nodes = ast_to_custom_node(ast_node.right)
            children.extend(left_nodes + right_nodes)
        elif isinstance(ast_node, ast.AugAssign):
            target_nodes = ast_to_custom_node(ast_node.target)
            value_nodes = ast_to_custom_node(ast_node.value)
            children.extend(target_nodes + value_nodes)

        return [Node(f"Operation: {op_label}", children=children)]

    # Handle 'For' loops.
    elif isinstance(ast_node, ast.For):
        loop_var_nodes = ast_to_custom_node(ast_node.target)
        iter_nodes = ast_to_custom_node(ast_node.iter)
        condition_node = Node("Condition:", children=loop_var_nodes + iter_nodes)

        body_nodes = []
        for stmt in ast_node.body:
            body_nodes.extend(ast_to_custom_node(stmt))

        children = [condition_node, Node("Body:", children=body_nodes)]
        return [Node("For", children=children)]

    # Handle variable assignments.
    elif isinstance(ast_node, ast.Assign):
        nodes = []
        if isinstance(ast_node.targets[0], ast.Tuple) and isinstance(ast_node.value, ast.Tuple):
            for target, value in zip(ast_node.targets[0].elts, ast_node.value.elts):
                target_label = f"Var: {target.id}"
                value_label = (
                    f"Const: {value.value}" if isinstance(value, ast.Constant) else f"Var: {value.id}"
                )
                nodes.append(Node("Assign", children=[Node(target_label), Node(value_label)]))
        else:
            target_label = f"Var: {ast_node.targets[0].id}"
            value_nodes = ast_to_custom_node(ast_node.value)
            nodes.append(Node("Assign", children=[Node(target_label)] + value_nodes))
        return nodes

    # Handle conditional statements ('If') and loops ('While').
    elif isinstance(ast_node, (ast.If, ast.While)):
        children = []

        # Handle the test condition
        if isinstance(ast_node.test, ast.Compare):
            condition_nodes = handle_comparison(ast_node.test)
            children.append(Node("Condition:", children=condition_nodes))
        elif isinstance(ast_node.test, ast.BoolOp):
            boolop_children = []
            for value in ast_node.test.values:
                if isinstance(value, ast.Compare):
                    boolop_children.extend(handle_comparison(value))
                else:
                    boolop_children.extend(ast_to_custom_node(value))
            op_name = type(ast_node.test.op).__name__
            children.append(Node(f"Cond. set: {op_name}", children=boolop_children))
        elif (
                isinstance(ast_node.test, ast.UnaryOp)
                and isinstance(ast_node.test.op, ast.Not)
                and isinstance(ast_node.test.operand, ast.Compare)
        ):
            condition_nodes = handle_comparison(ast_node.test.operand)
            children.append(Node("Condition:", children=[Node("Call: Not")] + condition_nodes))
        else:
            test_nodes = ast_to_custom_node(ast_node.test)
            if test_nodes:
                children.append(Node("Condition:", children=test_nodes))

        # Handle the body
        body_nodes = []
        for stmt in ast_node.body:
            body_nodes.extend(ast_to_custom_node(stmt))
        if body_nodes:
            children.append(Node("Body:", children=body_nodes))

        # Handle the else part
        else_nodes = []
        for stmt in ast_node.orelse:
            else_nodes.extend(ast_to_custom_node(stmt))
        if else_nodes:
            children.append(Node("Else:", children=else_nodes))

        return [Node(node_type, children=children)]

    # Handle list constructs.
    elif isinstance(ast_node, ast.List):
        elements = []
        for elt in ast_node.elts:
            if isinstance(elt, ast.Constant):
                elements.append(str(elt.value))
            elif isinstance(elt, ast.Name):
                elements.append(elt.id)
        elements_str = ', '.join(elements)
        return [Node(f"Const: [{elements_str}]")]

    # Handle variable subscripts.
    elif isinstance(ast_node, ast.Subscript):
        value_nodes = ast_to_custom_node(ast_node.value)
        slice_nodes = ast_to_custom_node(ast_node.slice)
        return value_nodes + [Node("Sliced by", children=slice_nodes)]

    # Handle comparison constructs.
    elif isinstance(ast_node, ast.Compare):
        return handle_comparison(ast_node)

    # Default case for other node types.
    else:
        if isinstance(ast_node, ast.Constant):
            value = ast_node.value
            formatted_value = f"'{value}'" if isinstance(value, str) else str(value)
            return [Node(f"Const: {formatted_value}")]
        elif isinstance(ast_node, ast.Call):
            children = []
            for arg in ast_node.args:
                children.extend(ast_to_custom_node(arg))
            func_name = ast_node.func.id if hasattr(ast_node.func, "id") else "Unknown"
            return [Node(f"Call: {func_name}", children=children)]
        elif isinstance(ast_node, ast.Name):
            return [Node(f"Var: {ast_node.id}")]
        elif isinstance(ast_node, ast.Module):
            children = process_child_nodes(ast_node)
            return [Node("Module", children=children)]
        elif isinstance(ast_node, ast.Return):
            value_nodes = ast_to_custom_node(ast_node.value)
            return [Node("Return", children=value_nodes)]
        elif isinstance(ast_node, ast.alias):
            return [Node(f"alias: {ast_node.name}")]
        elif isinstance(ast_node, ast.ImportFrom):
            names_nodes = []
            for alias in ast_node.names:
                names_nodes.extend(ast_to_custom_node(alias))
            return [Node(f"ImportFrom: {ast_node.module}", children=names_nodes)]
        else:
            children = process_child_nodes(ast_node)
            return [Node(node_type, children=children)]
