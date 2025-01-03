# Authorial Credit

This is a rewrite of Bas van Opheusden's [Four-in-a-row implementation](https://github.com/basvanopheusden/fourinarow). Please consider him the author of this code for citation purposes.

This repository was written for compatibility with Python 3 by Tyler Seip and is actively maintained by the members of Wei Ji Ma Lab.

# Dependencies

This repository requires Python 3.

**For Local Linux/Mac**, To install the build system (cmake) and SWIG, use the following commands:

```sh
sudo apt-get install cmake
sudo apt-get install swig
```

**For Cluster**, if you are using a Singularity instance, do not use `sudo`. Simply activate your Singularity instance and run

```sh
apt-get install cmake
apt-get install swig
```

**For Windows**, install cmake using the Windows installer, and download and unzip SWIG to a directory. Add the following environment variables for all users:

- `SWIG_DIR` : `<path to unzipped SWIG directory>`
- `SWIG_EXECUTABLE` : `<path to swig.exe in unzipped SWIG directory>`

Finally, add the SWIG directory (`SWIG_DIR` above) to your PATH system variable.

# Build Instructions

From a fresh checkout, create a build directory:

```sh
mkdir build
```

Then from the build directory:

```sh
cd build
cmake ..
cmake --build .
```

**For Windows**, if you do not have the debug Python libraries installed, specify the Release build at build-time:

```sh
cmake --build . --config Release
```

To run tests, execute `./tests` in the build output.

To install the required Python packages, from the `model_fitting` directory run:

```sh
pip install -r requirements.txt
```

To fit a model, from the `model_fitting` directory run:

```sh
python model_fit.py <path_to_game_csv>
```

# Commit Instructions

Contributors should run `utils/precommit.sh` from the `utils/` directory before committing. The dependencies for this script are documented in the script itself.

# Documentation

Documentation can be found at [https://weijimalab.github.io/ninarow/](https://weijimalab.github.io/ninarow/)
