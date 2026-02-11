# This file is part of ast_error_detection.
# Copyright (C) 2025 Badmavasan Kirouchenassamy & Eva Chouaki.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or any later version.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

from .annotated_tree import AnnotatedTree
from numpy import zeros

def insert_cost(node):
    """
    Compute the cost of inserting a node into the tree.

    This function is used by the Zhang-Shasha tree edit distance algorithm to determine
    how expensive it is to insert a node into one tree to match a node in the other tree.

    Args:
        node (Node): The node to be inserted.

    Returns:
        int: The cost of inserting the given node. A cost of 1 means every insertion has a uniform cost.
    """
    return 1


def remove_cost(node):
    """
    Compute the cost of removing a node from the tree.

    This function is used by the Zhang-Shasha tree edit distance algorithm to determine
    how expensive it is to remove a node from one tree to match the other tree.

    Args:
        node (Node): The node to be removed.

    Returns:
        int: The cost of removing the given node. A cost of 1 means every removal has a uniform cost.
    """
    return 1


def update_cost(node1, node2):
    """
    Compute the cost of updating one node to match another node.

    This function is used by the Zhang-Shasha tree edit distance algorithm. If the two nodes have
    the same label, the update cost is zero (they match perfectly). Otherwise, changing node1
    to node2 costs 1.

    Args:
        node1 (Node): The original node.
        node2 (Node): The node to which we are updating node1.

    Returns:
        int: 0 if both nodes have the same label (no cost to update), otherwise 1.
    """
    return 0 if node1.label == node2.label else 1

def distance(A, B, get_children):
    A = AnnotatedTree(A, get_children)
    B = AnnotatedTree(B, get_children)
    size_a = len(A.nodes)
    size_b = len(B.nodes)
    treedists = zeros((size_a, size_b), float)
    operations = [[[] for _ in range(size_b)] for _ in range(size_a)]

    def treedist(i, j):
        Al = A.lmds
        Bl = B.lmds
        An = A.nodes
        Bn = B.nodes

        m = i - Al[i] + 2
        n = j - Bl[j] + 2
        fd = zeros((m, n), float)
        partial_ops = [[[] for _ in range(n)] for _ in range(m)]

        ioff = Al[i] - 1
        joff = Bl[j] - 1

        fd[0][0] = 0
        for x in range(1, m):
            node = An[x + ioff]
            fd[x][0] = fd[x - 1][0] + remove_cost(node)
            op = {
                'type': 'delete',
                'path': node.get_path(),
                'current': node.label,
                'new': None
            }
            partial_ops[x][0] = partial_ops[x - 1][0] + [op]

        for y in range(1, n):
            node = Bn[y + joff]
            fd[0][y] = fd[0][y - 1] + insert_cost(node)
            op = {
                'type': 'insert',
                'path': node.get_path(),
                'current': None,
                'new': node.label
            }
            partial_ops[0][y] = partial_ops[0][y - 1] + [op]

        for x in range(1, m):
            for y in range(1, n):
                node1 = An[x + ioff]
                node2 = Bn[y + joff]

                if Al[i] == Al[x + ioff] and Bl[j] == Bl[y + joff]:
                    cost_remove = fd[x - 1][y] + remove_cost(node1)
                    cost_insert = fd[x][y - 1] + insert_cost(node2)
                    cost_update = fd[x - 1][y - 1] + update_cost(node1, node2)

                    costs = [cost_remove, cost_insert, cost_update]
                    min_cost = min(costs)
                    fd[x][y] = min_cost
                    idx = costs.index(min_cost)

                    if idx == 0:
                        op = {
                            'type': 'delete',
                            'path': node1.get_path(),
                            'current': node1.label,
                            'new': None
                        }
                        partial_ops[x][y] = partial_ops[x - 1][y] + [op]
                    elif idx == 1:
                        op = {
                            'type': 'insert',
                            'path': node2.get_path(),
                            'current': None,
                            'new': node2.label
                        }
                        partial_ops[x][y] = partial_ops[x][y - 1] + [op]
                    else:
                        if fd[x][y] == fd[x-1][y-1]:
                            op_type = 'match'
                        else:
                            op_type = 'update'
                        op = {
                            'type': op_type,
                            'path': node1.get_path(),
                            'current': node1.label,
                            'new': node2.label
                        }
                        partial_ops[x][y] = partial_ops[x - 1][y - 1] + [op]
                    treedists[x + ioff][y + joff] = fd[x][y]
                    operations[x + ioff][y + joff] = partial_ops[x][y]
                else:
                    p = Al[x + ioff] - 1 - ioff
                    q = Bl[y + joff] - 1 - joff
                    cost_remove = fd[x - 1][y] + remove_cost(node1)
                    cost_insert = fd[x][y - 1] + insert_cost(node2)
                    cost_subtree = fd[p][q] + treedists[x + ioff][y + joff]

                    costs = [cost_remove, cost_insert, cost_subtree]
                    min_cost = min(costs)
                    fd[x][y] = min_cost
                    idx = costs.index(min_cost)

                    if idx == 0:
                        op = {
                            'type': 'delete',
                            'path': node1.get_path(),
                            'current': node1.label,
                            'new': None
                        }
                        partial_ops[x][y] = partial_ops[x - 1][y] + [op]
                    elif idx == 1:
                        op = {
                            'type': 'insert',
                            'path': node2.get_path(),
                            'current': None,
                            'new': node2.label
                        }
                        partial_ops[x][y] = partial_ops[x][y - 1] + [op]
                    else:
                        partial_ops[x][y] = partial_ops[p][q] + operations[x + ioff][y + joff]

    for i in A.keyroots:
        for j in B.keyroots:
            treedist(i, j)

    return treedists[-1][-1], operations[-1][-1]
