# Syntax Guide

`pymu-eval` implements the exact grammar, operator precedence, and function semantics of C++ `muParser` v2.3.6.

---

## 1. Built-in Constants

| Constant | Value | Availability | Description |
| :--- | :--- | :--- | :--- |
| `_pi` | $3.14159265358979323846$ | `Parser`, `eval_expr` | $\pi$ (ratio of circumference to diameter) |
| `_e` | $2.71828182845904523536$ | `Parser`, `eval_expr` | Euler's number $e$ |
| `pi` | $3.14159265358979323846$ | `eval_expr` / `evaluate` | Convenience alias for `_pi` (unless overridden by a variable named `pi`) |
| `e` | $2.71828182845904523536$ | `eval_expr` / `evaluate` | Convenience alias for `_e` (unless overridden by a variable named `e`) |

You can register additional constants with `parser.define_const(name, value)` or `parser.DefineConst(name, value)`.

---

## 2. Built-in Binary Operators & Precedence

Binary operators are listed below from **lowest** to **highest** precedence (matching [`EOprtPrecedence`](../pymu_eval/definitions.py)):

| Operator | Precedence | Associativity | Description |
| :--- | :---: | :---: | :--- |
| `=` | `-1` | Right | Variable assignment (`a = b = 5`) |
| `? :` | `0` | Right | Ternary conditional (`cond ? then_expr : else_expr`) |
| `\|\|` | `1` | Left | Logical OR (non-zero is true; returns `1.0` or `0.0`) |
| `&&` | `2` | Left | Logical AND (non-zero is true; returns `1.0` or `0.0`) |
| `\|` | `3` | Left | Bitwise OR (enabled in `ParserInt` or when defined) |
| `&` | `4` | Left | Bitwise AND (enabled in `ParserInt` or when defined) |
| `!=`, `==` | `5` | Left | Inequality and equality comparison (`1.0` or `0.0`) |
| `<`, `<=`, `>`, `>=` | `6` | Left | Relational comparisons (`1.0` or `0.0`) |
| `+`, `-` | `7` | Left | Addition and subtraction |
| `*`, `/` | `8` | Left | Multiplication and division |
| `^` | `9` | Right | Exponentiation (`2^3^2 == 2^(3^2) == 512`) |

### Prefix (Infix) Unary Operators

| Operator | Precedence | Description |
| :--- | :---: | :--- |
| `-` | `7` (`prINFIX`) | Unary negation (`-x`). Note: matching C++ `muParser`, unary `-` has addition/subtraction precedence, so `-2^2 == -(2^2) == -4` and `2^-2 == 2^(-2) == 0.25`. |
| `+` | `7` (`prINFIX`) | Unary identity (`+x`). |

---

## 3. Built-in Mathematical Functions

### Single-Argument Functions

| Function | Formula / Description | Domain Notes |
| :--- | :--- | :--- |
| `sin(x)` | $\sin(x)$ (radians) | All real $x$ |
| `cos(x)` | $\cos(x)$ (radians) | All real $x$ |
| `tan(x)` | $\tan(x)$ (radians) | $x \neq \frac{\pi}{2} + k\pi$ |
| `cot(x)` | $\cot(x) = 1 / \tan(x)$ | $\tan(x) \neq 0$ |
| `asin(x)` | $\arcsin(x)$ in $[-\pi/2, \pi/2]$ | $x \in [-1, 1]$ |
| `acos(x)` | $\arccos(x)$ in $[0, \pi]$ | $x \in [-1, 1]$ |
| `atan(x)` | $\arctan(x)$ in $[-\pi/2, \pi/2]$ | All real $x$ |
| `sinh(x)` | $\sinh(x)$ | All real $x$ |
| `cosh(x)` | $\cosh(x)$ | All real $x$ |
| `tanh(x)` | $\tanh(x)$ | All real $x$ |
| `coth(x)` | $\coth(x) = \cosh(x)/\sinh(x)$ | $x \neq 0$ |
| `asinh(x)` | $\operatorname{arsinh}(x)$ | All real $x$ |
| `acosh(x)` | $\operatorname{arcosh}(x)$ | $x \ge 1$ |
| `atanh(x)` | $\operatorname{artanh}(x)$ | $x \in (-1, 1)$ |
| `log2(x)` | $\log_2(x)$ | $x > 0$ |
| `log10(x)` | $\log_{10}(x)$ | $x > 0$ |
| `log(x)` | $\ln(x)$ (natural logarithm, matching `muParser` v2.3+) | $x > 0$ |
| `ln(x)` | $\ln(x)$ (natural logarithm) | $x > 0$ |
| `exp(x)` | $e^x$ | All real $x$ |
| `sqrt(x)` | $\sqrt{x}$ | $x \ge 0$ |
| `sign(x)` | $-1$ if $x < 0$, $1$ if $x > 0$, $0$ if $x == 0$ | All real $x$ |
| `rint(x)` | Round to nearest integer: $\lfloor x + 0.5 \rfloor$ | All real $x$ |
| `abs(x)` | Absolute value $|x|$ | All real $x$ |
| `rnd()` | Random number in $[0, 1)$ | 0 arguments |

### Two-Argument Functions

| Function | Description |
| :--- | :--- |
| `atan2(y, x)` | Four-quadrant inverse tangent $\operatorname{atan2}(y, x)$ in $[-\pi, \pi]$ |

### Variadic Functions (1 or More Arguments)

Variadic functions accept any number of comma-separated arguments (`sum(a, b, c, ...)`). When arguments are NumPy arrays or `xarray.DataArray`s, they operate **element-wise across the provided arguments** (broadcasting automatically):

| Function | Description |
| :--- | :--- |
| `sum(x1, x2, ...)` | Element-wise sum: $x_1 + x_2 + \dots + x_n$ |
| `avg(x1, x2, ...)` | Element-wise arithmetic mean: $\frac{1}{n}\sum_{i=1}^n x_i$ |
| `min(x1, x2, ...)` | Element-wise minimum across all arguments |
| `max(x1, x2, ...)` | Element-wise maximum across all arguments |

---

## 4. Ternary Conditional (`? :`)

The `if-then-else` operator has the syntax:

```text
condition ? true_branch : false_branch
```

- **Nesting**: Ternary operators are right-associative and can be nested arbitrarily or enclosed in parentheses:
  ```text
  x < 0 ? -1 : x == 0 ? 0 : 1
  ```
- **Scalar Evaluation**: Short-circuits and evaluates only the active branch.
- **Array / xarray Evaluation**: Evaluates element-wise using `numpy.where(condition != 0, true_branch, false_branch)` or `xarray.where(condition != 0, true_branch, false_branch)`, preserving `xarray` dimensions and coordinates.

---

## 5. Multi-Expression Evaluation

Multiple expressions can be separated by commas at the top level of an expression string:

```python
from pymu_eval import Parser

p = Parser()
a = p.define_var("a", 1.0)
b = p.define_var("b", 2.0)
p.set_expr("a = b + 3, b = a * 2, a + b")

# eval() returns the last expression's value (15.0)
last_val = p.eval()

# eval_multi() returns a list of all expression results: [5.0, 10.0, 15.0]
all_vals = p.eval_multi()
```

---

## 6. Integer Parser (`ParserInt`)

[`ParserInt`](../pymu_eval/parser.py) is a specialized subclass for integer arithmetic, hexadecimal/binary literals, and bitwise logic:

- **Hexadecimal literals**: `0xff`, `0x10` (prefixed with `0x`)
- **Binary literals**: `#1010`, `#1111` (prefixed with `#`)
- **Bitwise & Integer Operators**:
  - `|` (bitwise OR), `&` (bitwise AND)
  - `<<` (left bit-shift), `>>` (right bit-shift)
  - `%` (modulo), `/` (integer division truncating toward zero)
  - `!` (logical NOT prefix operator)
  - `&&`, `||` (logical AND / OR)
