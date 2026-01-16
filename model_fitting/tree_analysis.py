from graphviz import Digraph
import numpy as np

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