#!/bin/bash
printf "\e[32mBeginning build of n-in-a-row package\e[0m\n"
rm -rf build

# -----------------------------
# 1. Detect environment
# -----------------------------
if [[ "$OSTYPE" == "darwin"* ]]; then
    ARCH=$(uname -m)
    echo "Detected macOS ($ARCH)"
    env=1
else
    HOSTNAME=$(hostname)
    # Check for known clusters first (both use local miniforge)
    if [[ "$HOSTNAME" == *"greene"* ]]; then
        echo "🧠 Detected Greene HPC cluster environment"
        env=2
    elif [[ "$HOSTNAME" == cs* ]]; then
        echo "🔥 Detected Torch cluster environment"
        env=5
    # Check for local miniforge installation
    elif [[ -d "/scratch/$USER/conda/miniforge3" ]]; then
        echo "🐍 Detected local miniforge installation"
        env=2
    elif [[ -n "$SLURM_CLUSTER_NAME" ]]; then
        echo "🧮 Detected generic SLURM cluster: $SLURM_CLUSTER_NAME"
        env=3
    else
        echo "Select environment:"
        echo "(1) Mac"
        echo "(2) Greene Cluster (NYU)"
        echo "(3) Generic Linux Cluster"
        echo "(4) Windows"
        echo "(5) Torch Cluster (NYU)"
        read -p "Enter number: " env
    fi
fi

# -----------------------------
# 2. Install / activate dependencies
# -----------------------------
case $env in
    1)
        echo "Installing dependencies for Mac..."
        brew install cmake swig boost
        ;;

    2)
        echo "Setting up for Greene Cluster..."
        source /scratch/$USER/conda/miniforge3/etc/profile.d/conda.sh
        conda activate env || {
            echo "Conda environment 'env' not found. Please create it first:"
            echo "  conda create -n env python cmake swig boost"
            exit 1
        }

        echo "✅ Using conda environment: $CONDA_PREFIX"
        which cmake || { echo "CMake not found. Run: conda install cmake"; exit 1; }
        which swig || { echo "SWIG not found. Run: conda install swig"; exit 1; }

        export BOOST_ROOT="$CONDA_PREFIX"
        export CMAKE_PREFIX_PATH="$CONDA_PREFIX:$CMAKE_PREFIX_PATH"
        ;;

    3)
        echo "Setting up for Generic Linux Cluster..."
        module load cmake 2>/dev/null || echo "⚠️ No cmake module found."
        module load swig 2>/dev/null || echo "⚠️ No swig module found."
        module load boost 2>/dev/null || echo "⚠️ No boost module found."

        if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
            source "$HOME/miniconda3/etc/profile.d/conda.sh"
            conda activate base
        fi
        ;;

    5)
        echo "Setting up for Torch cluster..."
        source /scratch/$USER/conda/miniforge3/etc/profile.d/conda.sh
        conda activate env || {
            echo "Conda environment 'env' not found. Please create it first:"
            echo "  conda create -n env python cmake swig boost"
            exit 1
        }

        echo "✅ Using conda environment: $CONDA_PREFIX"
        which cmake || { echo "CMake not found. Run: conda install cmake"; exit 1; }
        which swig || { echo "SWIG not found. Run: conda install swig"; exit 1; }

        export BOOST_ROOT="$CONDA_PREFIX"
        export CMAKE_PREFIX_PATH="$CONDA_PREFIX:$CMAKE_PREFIX_PATH"
        ;;

    4)
        echo "Windows setup instructions (manual install):"
        echo "- Install CMake using Windows installer."
        echo "- Download and unzip SWIG, set SWIG_DIR and SWIG_EXECUTABLE env vars."
        echo "- Download Boost and set BOOST_ROOT (e.g., C:\\Boost)."
        exit 0
        ;;

    *)
        echo "Invalid selection. Exiting."
        exit 1
        ;;
esac

# -----------------------------
# 3. Configure and build
# -----------------------------
echo "Creating build directory..."
mkdir -p build
cd build

echo "Running CMake..."
PY_EXEC=$(which python)
echo "Using Python executable: $PY_EXEC"
cmake -DPython3_EXECUTABLE=$PY_EXEC -Dgtest_discover_tests=OFF ..

# Limit build parallelism safely on login nodes
cmake --build . --config Release

# -----------------------------
# 4. Run Python install test
# -----------------------------
if [ -f "../model_fitting/install_test.py" ]; then
    echo "Running Python installation test..."
    python ../model_fitting/install_test.py || echo "⚠️ Python test script failed (check dependencies)."
else
    echo "No install_test.py found, skipping."
fi

# -----------------------------
# 5. Optional C++ tests
# -----------------------------
if [ -f "./tests" ]; then
    if [ "$env" -eq 2 ] || [ "$env" -eq 5 ]; then
        echo "Skipping C++ tests by default on Greene/Torch login nodes (heavy compute)."
    else
        read -p "Run compiled C++ tests? (y/n): " run_tests
        if [ "$run_tests" == "y" ]; then
            echo "Running tests..."
            ./tests || echo "⚠️ Some tests failed"
        else
            echo "Skipping C++ tests."
        fi
    fi
else
    echo "No C++ test binary found."
fi

# -----------------------------
# 6. Optional Python package installation
# -----------------------------
if [ -f "../model_fitting/requirements.txt" ]; then
    read -p "Install Python packages for model fitting? (y/n): " install_packages
    if [ "$install_packages" == "y" ]; then
        echo "Installing Python dependencies..."
        cd ../model_fitting
        pip install -r requirements.txt
        cd ../build
    else
        echo "Skipping Python package installation."
    fi
fi

# -----------------------------
# 7. Final checks
# -----------------------------
echo "Verifying C++ extension imports..."
python - <<'EOF'
try:
    import fourbynine
    print("✅ fourbynine module imported successfully")
except ImportError as e:
    print("⚠️ Could not import fourbynine:", e)
EOF

echo -e "\n----------------------------------------"
printf "\e[32m🎉 Build Complete.\e[0m\n"
echo "To fit a model:  cd model_fitting && python model_fit.py <path_to_game_csv>"
echo "----------------------------------------"
