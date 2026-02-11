# This file is part of ast_error_detection.
# Copyright (C) 2025 Badmavasan Kirouchenassamy & Eva Chouaki.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or any later version.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

def print_ast_nodes(nodes, indent=0):
    """
    Recursively print the labels of an AST's nodes with indentation.

    This function takes a list of nodes and prints their labels,
    indenting child nodes to visually represent the tree structure.

    Args:
        nodes (list[Node]): A list of nodes (e.g., the root node and its descendants).
        indent (int, optional): The current level of indentation.
                                Each level increases the indentation by four spaces.
                                Defaults to 0.

    Example of output of for i in range(5):\n\tprint("hello") :
    Module
        For
            Condition:
                Var: i
                Call: range
                    Const: 5
            Body:
                Call: print
                    Const: 'hello'
    """
    for node in nodes:
        print('    ' * indent + node.label)
        if node.children:
            print_ast_nodes(node.children, indent + 1)