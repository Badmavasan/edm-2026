# This file is part of ast_error_detection.
# Copyright (C) 2025 Badmavasan Kirouchenassamy & Eva Chouaki.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or any later version.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.


class AnnotatedTree:
    """
    A class representing an annotated tree structure used in the Zhang-Shasha tree edit distance algorithm.

    This class attaches additional information to each node in a tree to facilitate the computation of the
    minimal edit distance between two trees. It does so by performing a post-order traversal and computing:

    - Post-order node IDs.
    - The Leftmost Leaf Descendant (LMD) of each node.
    - Keyroots, which are nodes that represent roots of subproblems in the Zhang-Shasha algorithm.

    Attributes:
        get_children (callable): A function that, given a node, returns a list of its children.
        root (Node): The root node of the tree.
        nodes (list): A list of nodes in the tree in post-order.
        ids (list): A list of integer IDs corresponding to each node, assigned in post-order.
        lmds (list): A list of leftmost leaf descendant indices (LMDs) for each node.
        keyroots (list): A list of keyroot indices used in the tree edit distance computation.
        nodes_path (list): A list of paths (lists of labels) for each node from the root.

    The AnnotatedTree is designed to work hand-in-hand with the Zhang-Shasha distance algorithm. The trees
    are processed so that each node has a unique post-order ID and an associated LMD, allowing for efficient
    dynamic programming computation of the edit distance.
    """

    def __init__(self, root, get_children):
        """
        Initialize the AnnotatedTree with the given root and get_children function.

        Args:
            root (Node): The root node of the tree.
            get_children (callable): A function or method that takes a node and returns a list of its children.
        """
        self.get_children = get_children
        self.root = root
        self.nodes = []
        self.ids = []
        self.lmds = []
        self.keyroots = []
        self.nodes_path = []  # Store the path from the root to each node
        self._build(root)

    def _build(self, node):
        """
        Build the annotated tree data structures by performing a post-order traversal from the given node.

        This method initializes counters and mappings, then delegates to the recursive `_compute_post_order`
        method to populate the `nodes`, `ids`, and `lmds`. Finally, it computes the keyroots based on the LMD
        information.
        """
        self._id_counter = 0
        self._lmd_mapping = {}
        self._compute_post_order(node)
        self.keyroots = self._compute_keyroots()

    def _compute_post_order(self, node):
        """
        Recursively compute post-order traversal to assign IDs, determine LMDs, and record paths.

        During the traversal:
          - Each node is assigned a post-order ID.
          - The leftmost leaf descendant (LMD) of each node is determined.
          - The path from the root to the node is recorded.
          - Results are appended to class-level lists: `nodes`, `ids`, and `lmds`.

        Args:
            node (Node): The current node being processed.

        Returns:
            int: The LMD index of the current node.
        """
        # Retrieve children of the current node
        children = self.get_children(node)
        lmd = None

        # Post-order: first process children
        for child in children:
            child_lmd = self._compute_post_order(child)
            if lmd is None:
                # The LMD of the first child encountered will be the LMD of this node
                lmd = child_lmd

        # Assign post-order ID to the current node
        node_id = self._id_counter
        self._id_counter += 1
        self.nodes.append(node)

        # Store the path from the root to this node
        path = node.get_path()
        self.nodes_path.append(path)

        self.ids.append(node_id)

        # If no children, this is a leaf node; its LMD is itself
        if lmd is None:
            lmd = node_id

        self.lmds.append(lmd)
        self._lmd_mapping[node_id] = lmd

        return lmd

    def _compute_keyroots(self):
        """
        Compute the keyroots for the tree.

        Keyroots are defined as follows:
        - For each leftmost leaf descendant index (LMD) in `self.lmds`, the last occurrence of that LMD
          identifies a keyroot.
        - Keyroots are then sorted by their indices.

        Returns:
            list: A sorted list of keyroot indices.
        """
        lmd_to_index = {}
        # Map each LMD to its last occurrence in the list
        for index, lmd in enumerate(self.lmds):
            lmd_to_index[lmd] = index

        # Keyroots are the final occurrences of each LMD, sorted by their indices
        keyroots = sorted(lmd_to_index.values())
        return keyroots

    def print_tree_structure(self, name):
        """
        Print a human-readable representation of the tree structure.

        This method is intended to help with debugging or understanding the current tree layout
        after it has been processed into an `AnnotatedTree`. It displays each node with:
        - Its index in the post-order traversal list.
        - The path from the root to this node.
        - The node's label.
        - The labels of its children in the current order.

        Args:
            name (str): A descriptive name for the tree, used as a heading in the output.

        Example:
            If the tree structure is:
                Module
                    For
                        Condition:
                            Var: i
                            Call: range
                                Const: 5
                        Body:
                            Call: print
                                Const: 'hello'

        The output might look like:
            --- Code1 Tree Structure ---
            Node 0: Path: ['Module', 'For[0]', 'Condition:[0]', 'Var: i[0]'] | Label: 'Var: i' | Children Order: []
            Node 1: Path: ['Module', 'For[0]', 'Condition:[0]', 'Call: range[1]'] | Label: 'Call: range' | Children Order: ['Const: 5']
            ...
        """
        print(f"--- {name} Tree Structure ---")
        for idx, node in enumerate(self.nodes):
            path = node.get_path()
            label = node.label
            children_labels = [child.label for child in node.children]
            print(f"Node {idx}: Path: {path} | Label: '{label}' | Children Order: {children_labels}")

