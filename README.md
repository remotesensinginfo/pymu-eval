# pymu-eval

A pure-Python implementation of the [muParser](https://github.com/beltoforion/muparser) mathematical expression syntax parser and RPN bytecode evaluator (`import pymu_eval`), designed to compute mathematical expressions over scalars, **NumPy `ndarray`s**, **`xarray.DataArray`s**, and **`xarray.Dataset`s**.

## Documentation & Examples

The documentation is built with **Sphinx** and configured for **Read the Docs** via [.readthedocs.yaml](.readthedocs.yaml) and [docs/conf.py](docs/conf.py):

- **Sphinx Documentation Sources (`docs/`)**:
  - [docs/index.rst](docs/index.rst) ([Markdown version](docs/index.md)): Documentation overview & quick start
  - [docs/syntax_guide.rst](docs/syntax_guide.rst) ([Markdown version](docs/syntax_guide.md)): Operators, precedence, built-in functions, conditionals, and integer mode
  - [docs/numpy_xarray_guide.rst](docs/numpy_xarray_guide.rst) ([Markdown version](docs/numpy_xarray_guide.md)): NumPy & `xarray` evaluation guide
  - [docs/api_reference.rst](docs/api_reference.rst) ([Markdown version](docs/api_reference.md)): Autodoc API reference (`Parser`, `ParserInt`, `VariableRef`, `ParserError`, `eval_expr`)
  - [docs/examples.rst](docs/examples.rst): Annotated examples and Colab notebook reference
- **Examples & Interactive Tutorial (`examples/`)**:
  - [examples/pymu_eval_tutorial.ipynb](examples/pymu_eval_tutorial.ipynb): Annotated Jupyter Notebook (ready to run in **Google Colab** or local JupyterLab) with plots and real-world NumPy & `xarray` workflows.
  - [examples/01_basic_syntax.py](examples/01_basic_syntax.py): Core syntax, variables, constants, custom operators/functions, 5-point differentiation, and integer/bitwise parsing.
  - [examples/02_numpy_arrays.py](examples/02_numpy_arrays.py): 1D/2D NumPy array broadcasting, piecewise functions with `? :`, and array derivatives.
  - [examples/03_xarray_datasets.py](examples/03_xarray_datasets.py): Earth observation & meteorology workflows (cloud-masked NDVI, EVI, multi-threshold classification, and VPD) on `xarray.Dataset`s.

### Building Documentation Locally

```bash
sphinx-build -W -b html docs docs/_build/html
# or using Make:
make -C docs html
```

## Features

- **Full muParser Syntax Compatibility**:
  - Built-in binary operators with `muParser` precedence & associativity:
    - Assignment: `=`
    - Ternary conditional (`if-then-else`): `cond ? then_expr : else_expr` (supports both lazy scalar evaluation and vectorized element-wise `np.where` / `xr.where` evaluation)
    - Logical operators: `||`, `&&`
    - Comparison operators: `<=`, `>=`, `!=`, `==`, `<`, `>`
    - Arithmetic operators: `+`, `-`, `*`, `/`, `^` (right-associative power)
  - Built-in unary prefix (`-`, `+`) and custom prefix/infix, postfix, and binary operators.
  - Built-in functions:
    - Trigonometric: `sin`, `cos`, `tan`, `cot`, `asin`, `acos`, `atan`, `atan2`
    - Hyperbolic: `sinh`, `cosh`, `tanh`, `coth`, `asinh`, `acosh`, `atanh`
    - Logarithmic & exponential: `ln`, `log`, `log2`, `log10`, `exp`, `sqrt`
    - Rounding & sign: `rint`, `sign`, `abs`
    - Variadic (element-wise across arguments): `sum`, `avg`, `min`, `max`
    - Random: `rnd`
  - Built-in constants: `_pi`, `_e` (plus `pi` and `e` in `eval_expr` / `evaluate`).
  - String constants, string functions, user-data callbacks, variable factories, custom value recognizers, and locale customization (`SetDecSep`, `SetThousandsSep`, `SetArgSep`).
  - RPN bytecode compiler and constant/polynomial expression optimizer (`EnableOptimizer`).
  - 5-point 4th-order numerical differentiation (`diff` / `Diff`) for scalars, NumPy arrays, and `xarray.DataArray`s.
  - Integer parser (`ParserInt`) supporting hex (`0xff`), binary (`#101`), and bitwise operators (`&`, `|`, `<<`, `>>`, `%`, `!`).
- **Dual Naming Convention**:
  - Supports both Pythonic `snake_case` (`set_expr`, `eval`, `define_var`, `define_fun`, ...) and classic C++ `muParser` `CamelCase` (`SetExpr`, `Eval`, `DefineVar`, `DefineFun`, ...).

## Installation

```bash
pip install pymu-eval
# With optional xarray support:
pip install "pymu-eval[xarray]"

# Or from a local checkout:
pip install -e .
pip install -e ".[xarray]"
```

## Quick Start

### Evaluating with NumPy Arrays

```python
import numpy as np
from pymu_eval import Parser, eval_expr

a = np.linspace(0.1, 1.0, 5)
b = np.linspace(1.0, 2.0, 5)

# One-shot evaluation
result = eval_expr("a > 0.5 ? sin(a) + b^2 : cos(a)", a=a, b=b)

# Reusable compiled bytecode parser
p = Parser("2*a + 3*b - max(a, b)")
p.define_var("a", a)
p.define_var("b", b)
res = p.eval()
```

### Evaluating with xarray Datasets and DataArrays

```python
import numpy as np
import xarray as xr
from pymu_eval import Parser, eval_expr

ds = xr.Dataset(
    {
        "red": (("y", "x"), np.array([[0.1, 0.2], [0.15, 0.3]])),
        "nir": (("y", "x"), np.array([[0.5, 0.6], [0.45, 0.25]])),
        "cloud": (("y", "x"), np.array([[0, 0], [1, 0]])),
    },
    coords={"y": [10.0, 20.0], "x": [100.0, 110.0]},
)

# Compute masked NDVI directly from an xarray.Dataset
ndvi = eval_expr("cloud == 0 ? (nir - red) / (nir + red) : -9999", ds)
```

## Running Tests

```bash
pytest
```
