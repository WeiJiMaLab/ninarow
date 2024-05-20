#ifndef NINAROW_VECTORIZED_FEATURE_EVALUATOR_H_INCLUDED
#define NINAROW_VECTORIZED_FEATURE_EVALUATOR_H_INCLUDED

#include <Eigen/Dense>
#include <unordered_map>

#include "ninarow_heuristic_feature.h"
#include "player.h"

namespace NInARow {

/**
 * Counts the number of overlapping bits between a given bitset and a vector of
 * known bitsets in an efficient, vectorized way. Uses size_t instead of actual
 * bits to keep track of total overlap counts in the final evaluation.
 *
 * @tparam N The maximum length of all of the bitsets in the known vector of
 * bitsets.
 */
template <std::size_t N>
class VectorizedBitsetCounter {
 private:
  /**
   * Represents the known vector of bitsets - an M dimensional vector of length
   * N, where M is the number of bitsets that have been registered for
   * evaluation.
   */
  Eigen::Matrix<std::size_t, Eigen::Dynamic, N> bitset_matrix;

  /**
   * Converts a bitset to a one-dimensional vector of size_ts
   *
   * @param bitset The set of bits to convert.
   *
   * @return A vector of size_t, where each set element of the bitset
   * corresponds to a 1 in the vector.
   */
  static Eigen::Vector<std::size_t, N> bitset_to_vector(
      const std::bitset<N> &bitset) {
    Eigen::Vector<std::size_t, N> vector;
    for (std::size_t i = 0; i < N; ++i) {
      vector(i) = static_cast<std::size_t>(bitset[i]);
    }
    return vector;
  }

  /**
   * Converts a list of bitsets to a matrix.
   *
   * @param bitsets The list of bitsets to convert.
   *
   * @return A matrix of size_t, where each row of the matrix corresponds to a
   * bitset from the input.
   */
  static Eigen::Matrix<std::size_t, Eigen::Dynamic, N> bitsets_to_matrix(
      const std::vector<std::bitset<N>> &bitsets) {
    Eigen::Matrix<std::size_t, Eigen::Dynamic, N> matrix;
    matrix.conservativeResize(bitsets.size(), Eigen::NoChange);
    for (std::size_t i = 0; i < bitsets.size(); ++i) {
      matrix.row(i) = bitset_to_vector(bitsets[i]);
    }
    return matrix;
  }

 public:
  /**
   * Constructor.
   */
  VectorizedBitsetCounter() : bitset_matrix(0, N) {}

  /**
   * Adds a bitset into our known pool. After this function is called, each
   * query will return an additional line representing the bit overlap count
   * with this bitset.
   *
   * @param bitset The bitset to add.
   */
  void register_bitset(const std::bitset<N> &bitset) {
    bitset_matrix.conservativeResize(bitset_matrix.rows() + 1, Eigen::NoChange);
    bitset_matrix.row(bitset_matrix.rows() - 1) = bitset_to_vector(bitset);
  }

  /**
   * Queries all of the added bitsets against a list of new bitsets. Returns a
   * vector of vectors, where each top-level vector corresponds to a single
   * bitset passed in, and each element of the subvectors corresponds to a count
   * of the overlapping bits between each line of our registered bitsets and the
   * given bitset.
   *
   * @param bitsets The bitsets to query against.
   *
   * @return A list of lists of bit overlap counts, where each element
   * corresponds to the bit overlap count for each registered bitset against
   * each given bitset.
   */
  std::vector<std::vector<std::size_t>> query(
      std::vector<std::bitset<N>> bitsets) const {
    const Eigen::Matrix<std::size_t, Eigen::Dynamic, Eigen::Dynamic>
        count_results = bitsets_to_matrix(bitsets) * bitset_matrix.transpose();
    std::vector<std::vector<std::size_t>> output;
    for (std::size_t i = 0; i < bitsets.size(); ++i) {
      const auto start_address =
          count_results.data() + i * count_results.cols();
      output.push_back({start_address, start_address + count_results.cols()});
    }
    return output;
  }
};

/**
 * Registers a number of features that can all be evaluated simultaneously and
 * efficiently on given boards.
 *
 * @tparam Board The board that the feature will evaluate.
 */
template <typename Board>
class VectorizedFeatureEvaluator {
 private:
  /**
   * The number of features we're tracking.
   */
  std::size_t feature_count;

  /**
   * A counter representing the set of all of the pieces corresponding to all of
   * the features we're tracking. (A feature comprises pieces and spaces.) Each
   * line of this counter represents one feature's pieces.
   */
  VectorizedBitsetCounter<Board::get_board_size()> feature_pieces_bitsets;

  /**
   * A counter representing the set of all of the spaces corresponding to all of
   * the features we're tracking. (A feature comprises pieces and spaces.) Each
   * line of this counter represents one feature's spaces.
   */
  VectorizedBitsetCounter<Board::get_board_size()> feature_spaces_bitsets;

 public:
  /**
   * Constructor.
   */
  VectorizedFeatureEvaluator()
      : feature_count(0), feature_pieces_bitsets(), feature_spaces_bitsets() {}

  /**
   * Adds a new feature to the evaluator.
   *
   * @param feature The feature to add.
   *
   * @return The total number of features this evaluator is tracking.
   */
  std::size_t register_feature(const HeuristicFeature<Board> &feature) {
    feature_pieces_bitsets.register_bitset(feature.pieces.positions);
    feature_spaces_bitsets.register_bitset(feature.spaces.positions);
    return feature_count++;
  }

  /**
   * Given a list of boards and a player, count the number of pieces that the
   * player has on each board which overlap with each of our registered
   * features' pieces.
   *
   * @param boards The boards to evaluate.
   * @param player The player whose pieces we are evaluating.
   *
   * @return A list of counts representing the number of pieces that the
   * player has on the board that overlap with each feature in order.
   */
  std::vector<std::vector<std::size_t>> query_pieces(
      const std::vector<Board> &boards, Player player) const {
    std::vector<std::bitset<Board::get_board_size()>> positions;
    positions.reserve(boards.size());
    for (const auto &board : boards) {
      positions.push_back(board.get_pieces(player).positions);
    }
    return feature_pieces_bitsets.query(positions);
  }

  /**
   * Given a list of boards, count the number of spaces on each board which
   * overlap with each of our registered features' spaces.
   *
   * @param boards The boards to evaluate.
   *
   * @return A list of counts representing the amount of overlap between
   * between the board's spaces and each feature's spaces.
   */
  std::vector<std::vector<std::size_t>> query_spaces(
      const std::vector<Board> &boards) const {
    std::vector<std::bitset<Board::get_board_size()>> spaces;
    spaces.reserve(boards.size());
    for (const auto &board : boards) {
      spaces.push_back(board.get_spaces().positions);
    }
    return feature_spaces_bitsets.query(spaces);
  }

  /**
   * Helper functions for calling query_pieces/spaces on single board inputs
   * easily.
   *
   * @{
   */
  std::vector<std::size_t> query_pieces(const Board &board,
                                        Player player) const {
    return query_pieces(std::vector<Board>{board}, player)[0];
  }

  std::vector<std::size_t> query_spaces(const Board &board) const {
    return query_spaces(std::vector<Board>{board})[0];
  }
  /**
   * @}
   */
};
}  // namespace NInARow

#endif  // NINAROW_VECTORIZED_FEATURE_EVALUATOR_H_INCLUDED
