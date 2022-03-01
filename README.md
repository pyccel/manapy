[![Build Status](https://travis-ci.org/pyccel/manapy.svg?branch=master)](https://travis-ci.org/pyccel/manapy)


**MANAPY** is a Python 3 Library for Finite Volume using unstructured grids.

## Requirements
-----

***Python3**:
```bash
sudo apt-get install python3 python3-dev
```

***pip3**:
```bash
sudo apt-get install python3-pip
```

*All *non-Python* dependencies can be installed by following the [instructions for the pyccel library](https://github.com/pyccel/pyccel#Requirements)

## Installing the library
-----

***Standard mode**:
```bash
python3 -m pip install .
```
   
***Development mode**:
```bash
python3 -m pip install --user -e .
```

## Uninstall
-----
***Whichever the install mode**:
```bash
python3 -m pip uninstall manapy
```


## pyccelize functions

```python
./run_pyccel.sh
```

## Running tests
-----
```bash
python3 -m pytest  manapy -m "not parallel"
```


To use Mumps solver

```sh
sudo apt install libmumps-ptscotch-dev && pip install pymumps
```


To use petsc4py solver

https://petsc.org/release/install/