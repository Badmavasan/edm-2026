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
from graphviz import Digraph

from ast_error_detection.convert_ast_to_custom_node import ast_to_custom_node


class ASTVisualizer:
    def __init__(self):
        self.graph = Digraph(format="png")
        self.node_counter = 0

    def visit(self, node):
        node_id = f"node{self.node_counter}"
        self.node_counter += 1

        label = type(node).__name__
        self.graph.node(node_id, label)

        for child_name, child in ast.iter_fields(node):
            if isinstance(child, list):
                for sub_child in child:
                    if isinstance(sub_child, ast.AST):
                        child_id = self.visit(sub_child)
                        self.graph.edge(node_id, child_id, label=child_name)
            elif isinstance(child, ast.AST):
                child_id = self.visit(child)
                self.graph.edge(node_id, child_id, label=child_name)

        return node_id

    def visualize(self, code):
        parsed_code = ast.parse(code)
        self.visit(parsed_code)
        return self.graph


def visualize_plain_ast_from_code(source_code, graph_file="plain_ast_visualization"):
    """
    Parses Python source code into an AST and visualizes it using Graphviz.
    This function visualizes the plain AST before any custom modifications.

    Args:
        source_code (str): The Python source code to parse and visualize.
        graph_file (str): The base name of the output file (without extension).
                          The resulting file will be saved as <graph_file>.png.

    Returns:
        None: The visualization is saved as a PNG file.
    """
    visualizer = ASTVisualizer()
    ast_graph = visualizer.visualize(source_code)
    ast_graph.render(graph_file, cleanup=True)
    print(f"Plain AST visualization saved as {graph_file}.png")


def visualize_custom_ast_from_code(source_code, graph_file="custom_ast_visualization"):
    """
    Parses Python source code into an AST, converts it into a custom tree using the `Node` class,
    and visualizes the tree using Graphviz.

    Args:
        source_code (str): The Python source code to parse and visualize.
        graph_file (str): The base name of the output file (without extension).
                          The resulting file will be saved as <graph_file>.png.

    Returns:
        None: The visualization is saved as a PNG file.
    """

    def visualize_node_tree(node, graph=None, parent_id=None):
        if graph is None:
            graph = Digraph(format="png")

        # Create a unique identifier for the current node
        node_id = str(id(node))
        graph.node(node_id, label=node.label)

        if parent_id is not None:
            graph.edge(parent_id, node_id)

        # Recursively process the children
        for child in node.children:
            visualize_node_tree(child, graph, node_id)

        return graph

    # Parse the source code into an AST
    ast_tree = ast.parse(source_code)

    # Convert the AST into a custom tree of Node objects
    custom_nodes = ast_to_custom_node(ast_tree)

    # Visualize each top-level Node
    graph = Digraph(format="png")

    graph.attr(bgcolor="transparent")

    for node in custom_nodes:
        visualize_node_tree(node, graph)

    graph.render(graph_file, cleanup=True)
    print(f"Custom AST visualization saved as {graph_file}.png")


