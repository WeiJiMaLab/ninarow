# Authorial Credit

This is a rewrite of Bas van Opheusden's [Four-in-a-row implementation](https://github.com/basvanopheusden/fourinarow). Please consider him the author of this code for citation purposes.

This repository was written for compatibility with Python 3 by Tyler Seip and is actively maintained by the members of Wei Ji Ma Lab.

# Dependencies

This repository requires Python 3.
To build and stall the required packages, run the bash script at `build.sh`.

```sh
./build.sh
```
The script will build a folder called `build` which contains the necessary SWIG modules.
It will also automatically prompt you to run tests and install the needed Python packages. 
If anything fails, you can always install and test manually following the instructions in 
`legacy/README.md`.

To fit a model, from the `model_fitting` directory run:
```sh
python model_fit.py <path_to_game_csv>
```

# Commit Instructions

Contributors should run `utils/precommit.sh` from the `utils/` directory before committing. The dependencies for this script are documented in the script itself.

# Documentation

Documentation can be found at [https://weijimalab.github.io/ninarow/](https://weijimalab.github.io/ninarow/)
