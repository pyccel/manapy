# Goals

1. Implicit Finite Volumes: different discretization order (1, 2)
2. Unstructured (triangular) Finite Elements
    * Finite Elements: arbitrary polynomial order 
    * Triangular Bézier Finite Elements

# Low-level

1. data structure for Domain Decomposition (DDM) using MPI (but also in sequential mode)
2. Matrix over the DDM:
    * depending on the used discretization method; discretization order
3. Matrix operations: dot (with a vector), ...
4. Vector

# Mid-level
1. Discrete Function Space
2. Field: is an element of a discrete function space 

# High-level

## Finite Volumes 

* no notion of Function space
* however, the Function Space is implicitly used when interpolating the unknown function

```python
# Abstract model
V  = ScalarFunctionSpace('V', domain)
u  = element_of(V, name='u')
```

* we shall not use Bilinear forms 
* we may use a Linear form, where the integral is computed over an element

```python
# Abstract model
domain = Square()
element = element_of(Triangulation(domain))

V  = ... 
u  = element_of(V, name='u')

# defining an expression for u
# aa is a Tuple/Vector
expr = integral(element, dot(grad(u), aa))
```

### how to define a flux?
Two solutions:
1. the user defines or construct what a flux is
2. there pre-defined fluxes, and the user can call the flux operator over an expression


```python
expr = ...
expr = expr + flux(expr)

# discretization step
expr1_h = discretize(expr1, flux='scheme1', **kwargs)
expr2_h = discretize(expr2, flux='scheme2', **kwargs)
```


## Unstructured Finite Elements
