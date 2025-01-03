#!/bin/bash
echo "Are you running on (1) Linux/Mac, (2) Cluster, or (3) Windows?"
read -p "Enter the number corresponding to your environment: " env

case $env in
    1)
        echo "Installing dependencies for Linux/Mac..."
        sudo apt-get install cmake
        sudo apt-get install swig
        ;;
    2)
        echo "Installing dependencies for Cluster..."
        apt-get install cmake
        apt-get install swig
        ;;
    3)
        echo "Please install cmake using the Windows installer, and download and unzip SWIG to a directory."
        echo "Add the following environment variables for all users:"
        echo "- SWIG_DIR : <path to unzipped SWIG directory>"
        echo "- SWIG_EXECUTABLE : <path to swig.exe in unzipped SWIG directory>"
        echo "Finally, add the SWIG directory (SWIG_DIR above) to your PATH system variable."
        ;;
    *)
        echo "Invalid option. Exiting."
        exit 1
        ;;
esac

echo "Creating build directory..."
mkdir -p build
cd build

echo "Running cmake..."
cmake ..
cmake --build .

if [ "$env" -eq 3 ]; then
    echo "Running cmake for Release build on Windows..."
    cmake --build . --config Release
fi

read -p "Do you want to run tests? (y/n): " run_tests
if [ "$run_tests" == "y" ]; then
    echo "Running tests..."
    ./tests
else
    echo "Skipping tests."
fi

read -p "Do you want to install the required Python packages for model fitting? (y/n): " install_packages
if [ "$install_packages" == "y" ]; then
    echo "Installing required Python packages..."
    cd ../model_fitting
    pip install -r requirements.txt
    cd ../build
else
    echo "Skipping Python package installation."
fi

echo -e "\n\n\n\n\n"
echo "----------------------------------------"
echo -e "\e[32mHooray! Successfully Finished Installation of N-in-a-row Package!\e[0m"

echo "To fit a model, from the model_fitting directory run:"
echo "python model_fit.py <path_to_game_csv>"

echo "Contributors should run utils/precommit.sh from the utils/ directory before committing."
echo "Documentation can be found at https://weijimalab.github.io/ninarow/"
