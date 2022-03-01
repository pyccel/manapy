SCRIPT_DIR=$(dirname -- "$(realpath -- "$0")")
python3 -m pytest -v -m "not parallel"
mpirun -n 4 python3 -m pytest  -v -m "parallel"
