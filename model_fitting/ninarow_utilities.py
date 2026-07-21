import numpy as np
import argparse
import random
from pathlib import Path
from fourbynine import NInARowBestFirstSearch
import csv

# Input files for heuristic-quality evaluation live alongside this script, so the
# script is portable across checkouts (no hardcoded /scratch path).
HQ_INPUTS = Path(__file__).resolve().parent / "heuristic_quality_inputs"


def search_from_position(position, heuristic, noise_enabled=True, seed=None):
    """
    Given a position and a heuristic, execute a search from the given position
    and return the best move.
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


def get_heuristic_quality(params):
    """
    Given model parameters (of specifically length 58), evaluate the correlation of the
    parameters with a pre-derived set of optimal parameters.

    Returns:
        A number in [-1, 1]: correlation of the given parameters with optimal parameters.
    """
    feature_counts = np.loadtxt(HQ_INPUTS / "optimal_feature_vals.txt")[:, -35:]
    optimal_move_values = np.loadtxt(HQ_INPUTS / "opt_hvh.txt")[:, -36:]
    # columns are player id, color, cross-validation group, number of pieces, chosen move, response time in ms
    move_stats_hvh = np.loadtxt(HQ_INPUTS / "move_stats_hvh.txt", dtype=int)
    num_pieces_hvh = move_stats_hvh[:, 3]

    mask = ~np.isnan(optimal_move_values)
    optimal_move_values[mask] = np.vectorize(
        lambda x: -1 if x < -5000 else (1 if x > 5000 else 0))(optimal_move_values[mask])

    player_color = move_stats_hvh[:, 1]
    optimal_board_values = np.full_like(
        player_color, fill_value=np.nan, dtype=float)
    optimal_board_values[player_color == 0] = np.nanmax(
        optimal_move_values[player_color == 0, :], axis=1)
    optimal_board_values[player_color == 1] = - \
        np.nanmin(optimal_move_values[player_color == 1, :], axis=1)

    params = np.array(params)
    f3inarow = (params[9]+params[28])/2
    heuristic_values = np.tanh(0.4*np.sum((-2*player_color+1)[:, None]*feature_counts
                                          * params[None, 6:41]/f3inarow, axis=1))
    return np.corrcoef(heuristic_values, optimal_board_values)[0, 1]


def main():
    parser = argparse.ArgumentParser(
        description="Parse a parameters file and evaluate its heuristic quality "
                    "(correlation with optimal parameters).")
    parser.add_argument("-p", "--params", required=True, type=str,
                        help="The file containing the parameters for the model.")
    parser.add_argument("-o", "--output", required=False, type=str,
                        help="Filename to write the statistic to as a CSV. Optional.")
    args = parser.parse_args()
    from parsers import parse_bads_parameter_file_to_model_parameters
    params = parse_bads_parameter_file_to_model_parameters(args.params)
    stats = get_heuristic_quality(params)
    print(stats)
    if args.output:
        with open(args.output, 'w') as f:
            writer = csv.writer(f)
            writer.writerow([stats])


if __name__ == "__main__":
    main()
