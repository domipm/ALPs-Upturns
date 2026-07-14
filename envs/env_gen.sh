# BASH SCRIPT TO GENERATE CONDA ENV YAML FILE WITH DEPENDENCIES, ETC.
conda env export --no-builds | grep -v "^prefix: " > ../envs/env_$CONDA_DEFAULT_ENV.yaml