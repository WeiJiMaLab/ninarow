#!/bin/bash
printf "\e[32mBeginning build of n-in-a-row package\e[0m\n"
rm -rf build

# -----------------------------
# 1. Detect environment
# -----------------------------
if [[ "$OSTYPE" == "darwin"* ]]; then
    ARCH=$(uname -m)
    BREW_PREFIX=$(brew --prefix)
    echo "Detected macOS ($ARCH), Homebrew prefix: $BREW_PREFIX"
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
        brew install cmake boost pcre2
        
        # Install SWIG 4.3.0 to a local directory to avoid system conflicts and RPATH issues
        SWIG_LOCAL_DIR="$HOME/.ninarow/swig-4.3.0"
        SWIG_EXEC="$SWIG_LOCAL_DIR/bin/swig"
        
        # Check if local SWIG 4.3.0 is already installed AND functional
        SWIG_VERSION=$("$SWIG_EXEC" -version 2>/dev/null | head -1 | grep -o "4\.[0-9]\+\.[0-9]\+" || echo "")
        SWIG_WORKS=0
        if [ "$SWIG_VERSION" = "4.3.0" ] && "$SWIG_EXEC" -swiglib >/dev/null 2>&1; then
            SWIG_WORKS=1
        fi

        if [ "$SWIG_WORKS" -eq 1 ]; then
            echo "✅ Local SWIG 4.3.0 is functional at $SWIG_LOCAL_DIR"
        else
            echo "Installing SWIG 4.3.0 manually to $SWIG_LOCAL_DIR (avoids $function macro issues in 4.4.1)..."
            mkdir -p "$SWIG_LOCAL_DIR"
            SWIG_SRC_DIR="/tmp/swig-4.3.0"
            SWIG_TAR="$SWIG_SRC_DIR.tar.gz"
            
            # Download if not already present
            if [ ! -d "$SWIG_SRC_DIR" ]; then
                echo "Downloading SWIG 4.3.0 source..."
                curl -L -o "$SWIG_TAR" https://sourceforge.net/projects/swig/files/swig/swig-4.3.0/swig-4.3.0.tar.gz/download
                tar -xzf "$SWIG_TAR" -C /tmp
                rm -f "$SWIG_TAR"
            fi
            
            # Build and install to local directory
            cd "$SWIG_SRC_DIR"
            echo "Configuring SWIG 4.3.0..."
            ./configure --prefix="$SWIG_LOCAL_DIR"
            echo "Building SWIG 4.3.0..."
            make -j$(sysctl -n hw.ncpu 2>/dev/null || echo 4)
            echo "Installing SWIG 4.3.0..."
            make install
            
            # Explicitly fix dylib references on macOS so SWIG can load its dependencies
            if [[ "$OSTYPE" == "darwin"* ]]; then
                echo "Fixing SWIG dylib references for Homebrew libraries..."
                PCRE_PREFIX=$(brew --prefix pcre2 2>/dev/null || brew --prefix)

                # Add Homebrew lib dir as an rpath so libpcre2 is found
                install_name_tool -add_rpath "$BREW_PREFIX/lib" "$SWIG_EXEC" 2>/dev/null || true

                # Hard-wire @rpath/libpcre2-8.0.dylib to its absolute Homebrew path
                install_name_tool -change "@rpath/libpcre2-8.0.dylib" \
                    "$PCRE_PREFIX/lib/libpcre2-8.0.dylib" "$SWIG_EXEC" 2>/dev/null || true

                # Hard-wire @rpath/libc++.1.dylib to the macOS system path.
                # dyld looks for libc++ in rpath entries (e.g. pcre2/lib) where it
                # doesn't exist; pointing directly at /usr/lib avoids this entirely.
                install_name_tool -change "@rpath/libc++.1.dylib" \
                    "/usr/lib/libc++.1.dylib" "$SWIG_EXEC" 2>/dev/null || true

                # Verify it's functional now
                if ! "$SWIG_EXEC" -swiglib >/dev/null 2>&1; then
                    echo "⚠️ SWIG still has library loading issues after dylib fix."
                    echo "   Inspect with: otool -L $SWIG_EXEC"
                fi
            fi
            
            cd - > /dev/null
            echo "✅ SWIG 4.3.0 installed successfully to $SWIG_LOCAL_DIR"
        fi
        echo "Using SWIG: $("$SWIG_EXEC" -version | head -1)"
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
# On macOS, prefer Homebrew Python for the SWIG extension (avoids conda-related import segfault).
if [[ "$OSTYPE" == "darwin"* ]]; then
  for p in "$BREW_PREFIX/bin/python3" /usr/local/bin/python3; do
    [ -x "$p" ] && PY_EXEC=$p && break
  done
fi
PY_EXEC=${PY_EXEC:-$(which python3 2>/dev/null || which python)}
SWIG_EXEC=${SWIG_EXEC:-$(which swig)}
echo "Using Python executable: $PY_EXEC"
echo "Using SWIG: $SWIG_EXEC"
# macOS: avoid SWIG extension segfault on import by using -undefined dynamic_lookup
CMAKE_EXTRA=()
[[ "$OSTYPE" == "darwin"* ]] && CMAKE_EXTRA=(-DCMAKE_SHARED_LINKER_FLAGS="-undefined dynamic_lookup")
cmake -DPython3_EXECUTABLE=$PY_EXEC -DSWIG_EXECUTABLE=$SWIG_EXEC -Dgtest_discover_tests=OFF .. "${CMAKE_EXTRA[@]}"

# Limit build parallelism safely on login nodes
cmake --build . --config Release

# -----------------------------
# 4. Run Python install test
# -----------------------------
if [ -f "../model_fitting/tests/test_installation.py" ]; then
    echo "Running Python installation test..."
    "$PY_EXEC" ../model_fitting/tests/test_installation.py || echo "⚠️ Python test script failed (check dependencies)."
else
    echo "No test_installation.py found, skipping."
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
cd ../model_fitting
"$PY_EXEC" - <<'EOF'
try:
    import fourbynine
    print("✅ fourbynine module imported successfully")
except ImportError as e:
    print("⚠️ Could not import fourbynine:", e)
    print("   Make sure _swig_fourbynine.so exists in this directory")
EOF
cd ../build

echo -e "\n----------------------------------------"
printf "\e[32m🎉 Build Complete.\e[0m\n"
echo "To fit a model:  cd model_fitting && python model_fit.py <path_to_game_csv>"
echo "----------------------------------------"
