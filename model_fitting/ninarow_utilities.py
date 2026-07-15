import random
from fourbynine import NInARowBestFirstSearch


def search_from_position(position, heuristic, noise_enabled=True, seed=None):
    """
    Given a position and a heuristic, execute a search from the given position
    and return the best move.

    Args:
        position: The position (board) to search from.
        heuristic: The heuristic to use to evaluate the position.
        noise_enabled: If true, enable noise, else disable it.
        seed: The seed for the RNG in the heuristic.

    Returns:
        The best move from the given position as evaluated by the heuristic.
    """
    if seed:
        heuristic.seed_generator(seed)
    else:
        heuristic.seed_generator(random.randint(0, 2**64))
    heuristic.set_noise_enabled(noise_enabled)
    bfs = NInARowBestFirstSearch(heuristic, position)
    bfs.complete_search()
    return bfs.get_tree()


def bads_parameters_to_model_parameters(params):
    """
    Expands a truncated set of BADS parameters into a full set of heuristic params
    for constructing a heuristic.

    Args:
        params: The BADS parameters to convert (of length 10)
              [pruning, stopping_prob, feature_drop, lapse, c_opp, w_center, 2IAR_CON, 2IAR_DIS, 3IAR, 4IAR]

    Returns:
        The corresponding heuristic parameters (of length 58)
    """
    if (len(params) != 10):
        raise Exception(
            "Parameter file must contain 10 parameters: {}".format(params))
    params = list(map(float, params))
    out = [10000.0, params[0], params[1], params[3], 1, 1, params[5]]
    # Feature weights: params[6:] = [2IAR_CON, 2IAR_DIS, 3IAR, 4IAR]
    out.extend([x for x in params[6:]] * 4)
    out.append(0)
    out.extend([x * params[4] for x in params[6:]] * 4)
    out.append(0)
    out.extend([params[2]] * 17)
    return out
