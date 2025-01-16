// fourbynine.i - SWIG interface
%module(directors="1") fourbynine
%{
#include "game_tree_node.h"
#include "bfs_node.h"
#include "ninarow_bfs.h"
#include "ninarow_board.h"
#include "ninarow_move.h"
#include "ninarow_pattern.h"
#include "ninarow_heuristic.h"
#include "ninarow_heuristic_feature.h"
#include "ninarow_vectorized_feature_evaluator.h"
#include "fourbynine_features.h"
#include "player.h"
#include "searches.h"

// incorrect logic in old commit causes
// the size_t definition to FAIL for compiling.
// size_t should always be unsigned long int.
#if defined(_WIN32) || defined(_WIN64)
typedef unsigned long long int size_t;
#else
typedef unsigned long int size_t;
#endif
%}

%include "stdint.i"
%include "std_string.i"
%include "std_vector.i"
%include "std_shared_ptr.i"
%include "exception.i"

%exception {
  try {
    $function
  } catch(const std::exception& e) {
    std::cerr << e.what() << std::endl;
    SWIG_exception(SWIG_RuntimeError, e.what());
  } catch(...) {
    SWIG_exception(SWIG_UnknownError, "Unknown exception thrown!");
  }
}

%shared_ptr(NInARow::Heuristic<NInARow::Board<4, 9, 4>>);
%shared_ptr(Node<NInARow::Board<4, 9, 4>>);
%shared_ptr(BFSNode<NInARow::Board<4, 9, 4>>);
%shared_ptr(AbstractSearch<NInARow::Heuristic<NInARow::Board<4, 9, 4>>>);
%shared_ptr(Search<NInARow::Heuristic<NInARow::Board<4, 9, 4>>, BFSNode<NInARow::Board<4, 9, 4>>>);
%shared_ptr(NInARow::NInARowBestFirstSearch<NInARow::Heuristic<NInARow::Board<4, 9, 4>>>);

// Parse the original header files
%include "game_tree_node.h"
%include "bfs_node.h"
%include "ninarow_bfs.h"
%include "ninarow_board.h"
%include "ninarow_move.h"
%include "ninarow_pattern.h"
%include "ninarow_heuristic.h"
%include "ninarow_heuristic_feature.h"
%include "ninarow_vectorized_feature_evaluator.h"
%include "fourbynine_features.h"
%include "player.h"
%include "searches.h"

// Instantiate some templates

%template(fourbynine_board) NInARow::Board<4, 9, 4>;
%template(fourbynine_move) NInARow::Move<4, 9, 4>;
%template(fourbynine_pattern) NInARow::Pattern<4, 9, 4>;
%template(fourbynine_heuristic) NInARow::Heuristic<NInARow::Board<4, 9, 4>>;
%template(fourbynine_heuristic_feature) NInARow::HeuristicFeature<NInARow::Board<4, 9, 4>>;
%template(fourbynine_heuristic_feature_with_metadata) NInARow::HeuristicFeatureWithMetadata<NInARow::Board<4, 9, 4>>;
%template(fourbynine_game_tree_node) Node<NInARow::Board<4, 9, 4>>;
%template(fourbynine_bfs_node) BFSNode<NInARow::Board<4, 9, 4>>;
%template(fourbynine_feature_evaluator) NInARow::VectorizedFeatureEvaluator<NInARow::Board<4, 9, 4>>;
%template(DoubleVector) std::vector<double>;
namespace std {
  %template(SizeTVector) vector<size_t>;
  %template(SizeTVectorVector) vector<std::vector<size_t>>;
}
%template(MoveVector) std::vector<NInARow::Move<4, 9, 4>>;
%template(BoardVector) std::vector<NInARow::Board<4, 9, 4>>;
%template(NodeVector) std::vector<std::shared_ptr<Node<NInARow::Board<4, 9, 4>>>>;
%template(BFSNodeVector) std::vector<std::shared_ptr<BFSNode<NInARow::Board<4, 9, 4>>>>;
%template(FeatureGroupWeightVector) std::vector<NInARow::FeatureGroupWeight>;
%template(FeatureWithMetadataVector) std::vector<NInARow::HeuristicFeatureWithMetadata<NInARow::Board<4, 9, 4>>>;

%feature("director") AbstractSearch;
%template(AbstractSearch) AbstractSearch<NInARow::Heuristic<NInARow::Board<4, 9, 4>>>;

%feature("director") BestFirstSearch;
%template(BestFirstSearch) Search<NInARow::Heuristic<NInARow::Board<4, 9, 4>>, BFSNode<NInARow::Board<4, 9, 4>>>;

%feature("director") NInARowBestFirstSearch;
%template(NInARowBestFirstSearch) NInARow::NInARowBestFirstSearch<NInARow::Heuristic<NInARow::Board<4, 9, 4>>>;

