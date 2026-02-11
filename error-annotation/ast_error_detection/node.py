# This file is part of ast_error_detection.
# Copyright (C) 2025 Badmavasan Kirouchenassamy & Eva Chouaki.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or any later version.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import collections


class Node(object):
    """
    A custom tree node class designed to represent a node in an Abstract Syntax Tree (AST)
    with a human-readable label. This class can be used as input to the Zhang-Shasha tree edit
    distance algorithm. Each node maintains references to its parent, children, and an index
    position among its siblings.

    Typical usage involves:
    - Creating a Node with a label and optionally providing children.
    - Establishing a parent-child relationship automatically when children are added.
    - Accessing children, retrieving labels, and performing operations like subtree iteration.

    Attributes:
        label (str): A string label describing the node type or value.
                     For example: "For" for a loop node, "Call: print" for a function call node, etc.
        children (list): A list of `Node` objects representing the children of this node.
        parent (Node): A reference to the node's parent. This is set automatically when a node
                       is added as a child to another node.
        index (int): The index of this node among its siblings. This is set automatically when
                     nodes are added to a parent node.

    This structure is particularly useful for representing parsed ASTs in a simplified,
    labeled tree format suitable for tree difference computations.
    """

    def __init__(self, label, children=None, parent=None, index=None):
        """
        Initialize a Node with a given label and optional children, parent, and index.

        Args:
            label (str): The label for this node, describing the type or value it represents.
            children (list[Node], optional): A list of child Nodes. Defaults to None.
            parent (Node, optional): The parent Node of this node. Defaults to None.
            index (int, optional): The index of this node among its siblings. Defaults to None.

        When children are provided, this constructor updates each child's `parent` and `index`
        attributes to maintain the tree structure.
        """
        self.label = label
        self.children = []
        self.parent = parent
        self.index = index

        if children is not None:
            for idx, child in enumerate(children):
                # Assign parent and index to each child
                child.parent = self
                child.index = idx
                self.children.append(child)

    @staticmethod
    def get_children(node):
        """
        A static method that retrieves the children of a given node in a consistent manner.

        This method is designed to abstract away differences in data structures
        that might store children. It handles various scenarios:
        - If node is None, return an empty list.
        - If node is a list, return it directly.
        - If node has a `children` attribute, return that.
        - If node is callable, call it to get children.
        - Otherwise, return an empty list.

        Args:
            node (Node or other): The entity from which to retrieve children.

        Returns:
            list[Node]: The children of the node, or an empty list if none.
        """
        if node is None:
            return []
        if isinstance(node, list):
            return node
        if hasattr(node, 'children'):
            return node.children
        if callable(node):
            return node()
        return []

    @staticmethod
    def get_sorted_children(node):
        """
        Retrieve the children of a node and return them in a sorted order based on their labels.

        This can ensure a deterministic order of children, which may be important
        for certain algorithms like tree edit distance.

        Args:
            node (Node): The node whose children are to be retrieved and sorted.

        Returns:
            list[Node]: A sorted list of the node's children by their label.
        """
        return sorted(Node.get_children(node), key=lambda x: x.get_label())

    @staticmethod
    def get_parent(node):
        """
        Retrieve the label of the parent of the given node, if it exists.

        Args:
            node (Node): The node whose parent's label is needed.

        Returns:
            str or None: The label of the parent node, or None if there is no parent.
        """
        if node is not None and node.parent is not None:
            return node.parent.label
        else:
            return None

    def get_label(self):
        """
        Retrieve the label of this node. The label describes the node's type or value.

        This method is compatible with the zss (Zhang-Shasha) algorithm's expected interface.

        Returns:
            str or None: The node's label, or None if no label is set.
        """
        if self.label is None:
            return None
        return self.label

    def addkid(self, node, before=False):
        """
        Add a child node to this node's children.

        The child's `parent` is set to this node, and `index` is updated accordingly.

        Args:
            node (Node): The child node to add.
            before (bool, optional): If True, insert the child at the beginning of the children list.
                                     Otherwise, append it at the end. Defaults to False.

        Returns:
            Node: This node (for chaining calls if desired).
        """
        if before:
            self.children.insert(0, node)
        else:
            self.children.append(node)
        node.parent = self
        # Reassign indices to maintain correct ordering
        for idx, child in enumerate(self.children):
            child.index = idx
        return self

    def get(self, label):
        """
        Retrieve a descendant node by label.

        Performs a search of the subtree rooted at this node for a node whose label matches
        the given label. Returns the first match found.

        Args:
            label (str): The label to search for.

        Returns:
            Node or None: The node with the given label if found, otherwise None.
        """
        if self.label == label:
            return self
        for c in self.children:
            if label in c:
                return c.get(label)
        return None

    def iter(self):
        """
        Iterate over this node and all its descendants in a pre-order traversal.

        Yields:
            Node: Each node in the subtree rooted at this node.
        """
        queue = collections.deque()
        queue.append(self)
        while len(queue) > 0:
            n = queue.popleft()
            for c in n.children:
                queue.append(c)
            yield n

    def get_path(self):
        """
        Compute the path from the root of the tree down to this node.

        The path includes the node's label and its index if the parent has multiple children.
        This can be useful for debugging or for identifying the node's position within the tree.

        Returns:
            list[str]: The path from the root to this node, each element being a label
                       (with index if applicable).
        """
        path = []
        node = self
        while node is not None:
            label = node.label
            # Include [index] if the node has a parent
            if node.parent:
                label = f"{label}[{node.index}]"
            path.insert(0, label)
            node = node.parent
        return path

    def __contains__(self, b):
        """
        Check whether a given string or Node is present in the subtree rooted at this node.

        Args:
            b (str or Node): The label or Node to search for.

        Returns:
            int: 1 if found, 0 otherwise.

        Raises:
            TypeError: If b is neither a str nor a Node.
        """
        if isinstance(b, str) and self.label == b:
            return 1
        elif not isinstance(b, str) and self.label == b.label:
            return 1
        elif (isinstance(b, str) and self.label != b) or (not isinstance(b, str) and self.label != b.label):
            return sum(b in c for c in self.children)
        raise TypeError("Object %s is not of type str or Node" % repr(b))

    def __eq__(self, b):
        """
        Check equality between this node and another object.

        Two nodes are considered equal if they have the same label.

        Args:
            b: The object to compare against.

        Returns:
            bool: True if b is a Node with the same label, False otherwise.
        """
        if b is None:
            return False
        if not isinstance(b, Node):
            return False
        return self.label == b.label

    def __ne__(self, b):
        """
        Check inequality between this node and another object.

        Inverse of __eq__.

        Args:
            b: The object to compare against.

        Returns:
            bool: True if not equal, False otherwise.
        """
        return not self.__eq__(b)

    def __repr__(self):
        """
        Return a string representation of the Node instance.

        This includes the node's class and label, which can be useful for debugging.
        """
        return super(Node, self).__repr__()[:-1] + " %s>" % self.label

    def __str__(self):
        """
        Return a string representation of the subtree rooted at this node.

        The output includes the number of children and the label of the node on the first line,
        followed by a recursive representation of each child.
        """
        s = "%d:%s" % (len(self.children), self.label)
        s = '\n'.join([s] + [str(c) for c in self.children])
        return s

