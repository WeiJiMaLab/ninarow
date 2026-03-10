import json
from pathlib import Path

from graphviz import Digraph
import numpy as np


def _pattern_string_to_board_positions(pattern_string):
    """Convert pattern.to_string() to list of 0-indexed board positions."""
    return [i for i in range(len(pattern_string)) if pattern_string[-i - 1] == "1"]


def tree_to_json(root, path=None, max_depth=None, indent=2):
    """
    Export the search tree to a readable JSON format for the TreeSearch Vue component.

    Each node has: id, self (board positions of active player), opponent (positions of
    other player), and optionally children.

    Args:
        root: Root node from NInARowBestFirstSearch.get_tree()
        path: If provided, write JSON to this file path.
        max_depth: If provided, limit tree depth.
        indent: JSON indent for readability (default 2).

    Returns:
        dict: The tree as a nested dict. If path is given, also writes to file.
    """
    from fourbynine import get_other_player

    root_board = root.get_board()
    root_self_player = root_board.active_player()
    root_opponent_player = get_other_player(root_self_player)

    counter = [0]

    def node_to_dict(node, depth, on_pv=True):
        if max_depth is not None and depth > max_depth:
            return None

        node_id = f"n{counter[0]}"
        counter[0] += 1

        board = node.get_board()
        self_positions = _pattern_string_to_board_positions(
            board.get_pieces(root_self_player).to_string()
        )
        opponent_positions = _pattern_string_to_board_positions(
            board.get_pieces(root_opponent_player).to_string()
        )

        best_move_pos = (
            node.get_best_move().board_position if node.get_children() else None
        )

        children = []
        for child in node.get_children():
            if max_depth is not None and depth >= max_depth:
                continue
            child_on_pv = (
                on_pv
                and best_move_pos is not None
                and child.get_move().board_position == best_move_pos
            )
            child_dict = node_to_dict(child, depth + 1, on_pv=child_on_pv)
            if child_dict is not None:
                children.append(child_dict)

        value = node.get_value()
        v = float(value)
        if np.isinf(v):
            value_out = "win" if v > 0 else "lose"
        else:
            value_out = round(v, 2)

        out = {
            "id": node_id,
            "self": self_positions,
            "opponent": opponent_positions,
            "value": value_out,
            "pv": on_pv,
        }
        if depth > 0:
            out["move"] = node.get_move().board_position
        if children:
            out["children"] = children
        return out

    tree_dict = node_to_dict(root, 0, on_pv=True)
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(tree_dict, f, indent=indent)
    return tree_dict


def tree_to_graphviz(root, max_depth=None):
    """
    Returns a Graphviz Digraph object visualizing the tree rooted at `root`.
    Optionally limits the depth of traversal.
    """
    dot = Digraph(comment="Search Tree")
    dot.attr(
        "node",
        shape="box",
        fontname="Courier New",
        margin="0.2,0.2"
    )

    visited = {}
    counter = [0]

    def add_node_rec(node, depth):
        if max_depth is not None and depth > max_depth:
            return None

        if node not in visited:
            node_id = f"n{counter[0]}"
            counter[0] += 1
            visited[node] = node_id

            board_str = (
                node.get_board().to_string()
                .replace("\r\n", "\n")
                .replace("o", "●")
                .replace("x", "○")
                .replace(" ", "·")
                .replace("|", "")
                .replace("-", "")
                .replace("+", "")
                .replace("\n", r"\l")
            )
            value = node.get_value()
            label = f"{board_str}\\lValue: {np.round(value, 2)}\\l"
            dot.node(node_id, label=label)
        else:
            node_id = visited[node]

        for child in node.get_children():
            if max_depth is not None and depth >= max_depth:
                continue
            child_id = add_node_rec(child, depth + 1)
            if child_id is not None:
                dot.edge(node_id, child_id, label=f"{child.get_move().board_position}", fontname="Courier")
        return node_id

    add_node_rec(root, depth=0)
    return dot

def get_moves(node, depth=0):
    """
    Recursively collects all move positions in the tree, except for the root.
    Returns a flat list of board_position values.
    """
    moves = [] if depth == 0 else [node.get_move().board_position]
    for child in node.get_children():
        moves.extend(get_moves(child, depth + 1))
    return moves

def get_all_paths(node, path=None, depth=0):
    """
    Returns all root-to-leaf paths as lists of board positions.
    Each path is a list of board_positions corresponding to moves.
    """
    if path is None:
        path = []
    if not list(node.get_children()):
        return [path]
    paths = []
    for child in node.get_children():
        paths.extend(get_all_paths(child, path + [child.get_move().board_position], depth + 1))
    return paths

def get_principal_variation(node):
    pv = []
    while node.get_children():
        best_move = node.get_best_move().board_position
        pv.append(best_move)
        node = next(child for child in node.get_children() if child.get_move().board_position == best_move)
    return pv

def get_max_depth(node):
    """
    Returns the maximum depth of the tree from the given node.
    
    Args:
        node: The root node of the tree
        
    Returns:
        The depth of the deepest node in the tree (using node.get_depth())
    """
    if not node.get_children():
        return 0
    
    max_child_depth = max(get_max_depth(child) + 1 for child in node.get_children())
    return max_child_depth