cmake_minimum_required(VERSION 3.15)
project(NInARow)

include (FetchContent)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)

# Make the default build type Release. If user or another
# project sets a different value than use that
if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "Setting build type to default -- Release")
  set(CMAKE_BUILD_TYPE "Release" CACHE
      STRING "Choose the type of build." FORCE)
endif()
message(STATUS "NInARow Build Type: ${CMAKE_BUILD_TYPE}")

# Detect architecture
if(APPLE)
    execute_process(
        COMMAND uname -m
        OUTPUT_VARIABLE ARCHITECTURE
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    message(STATUS "Detected architecture: ${ARCHITECTURE}")
    
    # Set specific flags for Apple Silicon
    if("${ARCHITECTURE}" STREQUAL "arm64")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -arch arm64")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -arch arm64")
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -arch arm64")
    endif()
endif()

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.13.0.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

FetchContent_Declare(
      Eigen
      GIT_REPOSITORY    https://gitlab.com/libeigen/eigen.git
      GIT_TAG           master
    )
FetchContent_MakeAvailable(Eigen)

# Find Python and SWIG
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(SWIG REQUIRED)

# Find Boost library
find_package(Boost REQUIRED)

# Add Boost include directories
include_directories(${Boost_INCLUDE_DIRS})

include(${SWIG_USE_FILE})
set(CMAKE_SWIG_FLAGS "")
include_directories(${Python3_INCLUDE_DIRS})

set(CXX_FLAGS)

list(APPEND CXX_FLAGS "-fPIC" "-Wall" "-Wextra" "-Werror" "-Wno-unused" "-fexceptions")

enable_testing()

add_executable(tests
  ${CMAKE_SOURCE_DIR}/../tests/cpp/bfs_node_ut.cpp
  ${CMAKE_SOURCE_DIR}/../tests/cpp/ninarow_bfs_ut.cpp
  ${CMAKE_SOURCE_DIR}/../tests/cpp/ninarow_board_ut.cpp
  ${CMAKE_SOURCE_DIR}/../tests/cpp/ninarow_heuristic_ut.cpp
  ${CMAKE_SOURCE_DIR}/../tests/cpp/ninarow_heuristic_feature_ut.cpp
  ${CMAKE_SOURCE_DIR}/../tests/cpp/ninarow_vectorized_feature_evaluator_ut.cpp
  ${CMAKE_SOURCE_DIR}/../tests/cpp/ninarow_move_ut.cpp
  ${CMAKE_SOURCE_DIR}/../tests/cpp/ninarow_pattern_ut.cpp
)
target_include_directories(tests PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(tests
  GTest::gtest_main
  Eigen3::Eigen
)

include(GoogleTest)
gtest_discover_tests(tests)

SET_SOURCE_FILES_PROPERTIES(fourbynine.i PROPERTIES CPLUSPLUS ON)
SWIG_ADD_LIBRARY(swig_fourbynine TYPE SHARED LANGUAGE python OUTPUT_DIR "../model_fitting" OUTFILE_DIR "./" SOURCES fourbynine.i)
set_property(TARGET swig_fourbynine PROPERTY SWIG_USE_TARGET_INCLUDE_DIRECTORIES TRUE)
 
include(CheckTypeSize)

# Check the size of size_t
check_type_size("size_t" SIZE_T_SIZE)
message(STATUS "Detected size_t size: ${SIZE_T_SIZE}")

# Handle architecture-specific SWIG configuration
if(APPLE AND "${ARCHITECTURE}" STREQUAL "arm64")
    # For Apple Silicon (arm64), we need to ensure proper 64-bit handling
    target_compile_definitions(swig_fourbynine PRIVATE SWIGWORDSIZE64)
    target_compile_definitions(swig_fourbynine PRIVATE __aarch64__)
    target_compile_definitions(swig_fourbynine PRIVATE SWIG_PYTHON_SAFE_CSTR=1)
    target_compile_definitions(swig_fourbynine PRIVATE SWIG_PYTHON_CAST_MODE_PYTHON)
    set_target_properties(swig_fourbynine PROPERTIES 
        COMPILE_FLAGS "-arch arm64"
        LINK_FLAGS "-arch arm64"
    )
elseif(NOT WIN32)
    # For other Unix-like systems, check size_t size
    if (NOT SIZE_T_SIZE EQUAL 8)
        target_compile_definitions(swig_fourbynine PRIVATE SWIGWORDSIZE64)
    endif()
endif()

IF (NOT WIN32)
  set_target_properties(swig_fourbynine PROPERTIES SUFFIX ".so")
ENDIF()

target_include_directories(swig_fourbynine PUBLIC "./")

# On macOS, Python extensions must NOT link against libpython.
# Python symbols are resolved at runtime from the interpreter via -undefined dynamic_lookup.
# Explicitly linking Python3::Python causes duplicate symbols and segfaults, especially with conda.
# On Linux/Windows, explicit linking is required.
if(NOT APPLE)
  target_link_libraries(swig_fourbynine Python3::Python)
endif()
target_link_libraries(swig_fourbynine Eigen3::Eigen)


# Set Python module properties
set_target_properties(swig_fourbynine PROPERTIES
    PREFIX "_"
    SUFFIX ".so"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../model_fitting"
)

add_custom_command(TARGET swig_fourbynine POST_BUILD 
  COMMAND "${CMAKE_COMMAND}" -E copy 
     "$<TARGET_FILE:swig_fourbynine>"
     "../model_fitting" 
  COMMENT "Copying SWIG library to output directory")
