"""Generate synthetic choice data by sampling the TreeSearch model at known params.

For each board position we draw one move from the model's stochastic policy via
``TreeSearch.predict``. The move index (0..35) is written as a single-bit bitboard
``2**index`` so the result passes ``MultiThreadedFitter.check_dataframe`` and can be
refit by the standard pipeline unchanged.
"""

import numpy as np
import pandas as pd

from recovery_common import build_model

from tree_search import fourbynine_board, fourbynine_pattern


def generate_synthetic_dataset(params, boards, seed=0, verbose=False):
    """Sample one synthetic move per board row at the given parameter vector.

    Args:
        params: length-12 parameter vector in the 6-template model's order.
        boards: DataFrame with at least black/white/color columns.
        seed: pins the heuristic RNG so generation is reproducible.

    Returns:
        DataFrame with black/white/move/color (+ passthrough trial_id/n_pieces),
        where ``move`` is a single-bit bitboard sampled from the model.
    """
    model = build_model(verbose=verbose)
    model.set_params(np.asarray(params, dtype=np.float64))
    # set_params seeds randomly; override for reproducibility.
    model.heuristic.seed_generator(int(seed))

    moves = np.empty(len(boards), dtype=np.int64)
    for i, row in enumerate(boards.itertuples()):
        board = fourbynine_board(
            fourbynine_pattern(int(row.black)), fourbynine_pattern(int(row.white))
        )
        position = model.predict(board)
        moves[i] = 1 << int(position)

    out = pd.DataFrame(
        {
            "black": boards["black"].astype("int64").values,
            "white": boards["white"].astype("int64").values,
            "move": moves,
            "color": boards["color"].values,
        }
    )
    for passthrough in ("trial_id", "n_pieces"):
        if passthrough in boards.columns:
            out[passthrough] = boards[passthrough].values
    return out
