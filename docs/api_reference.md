# API Reference

All classes and functions are importable directly from the top-level `pymu_eval` package (`pip install pymu-eval`):

```python
from pymu_eval import (
    Parser,
    ParserBase,
    ParserInt,
    ParserError,
    VariableRef,
    Variable,
    eval_expr,
    evaluate,
    ECmdCode,
    ETypeCode,
    EErrorCodes,
    EOprtPrecedence,
    EOprtAssociativity,
)
```

---

## 1. Convenience Function: `eval_expr` / `evaluate`

```python
eval_expr(
    expr: str,
    variables: Mapping[str, Any] | xr.Dataset | None = None,
    *,
    constants: Mapping[str, Any] | None = None,
    functions: Mapping[str, Callable] | None = None,
    **kwargs: Any,
) -> Any
```

One-shot helper that compiles and evaluates a `muParser` expression over scalars, `numpy.ndarray`s, `xarray.DataArray`s, or an `xarray.Dataset`.

### Parameters
- **`expr`** (`str`): Mathematical expression string using `muParser` syntax.
- **`variables`** (`Mapping[str, Any]` or `xarray.Dataset`, optional):
  - If a dict/mapping is passed, each key is bound as a parser variable.
  - If an `xarray.Dataset` is passed, all `data_vars` and `coords` in the dataset are automatically bound as variables.
- **`constants`** (`Mapping[str, Any]`, optional): Additional named constants to register via `define_const`.
- **`functions`** (`Mapping[str, Callable]`, optional): Custom Python callables to register via `define_fun`.
- **`**kwargs`**: Additional variables passed by keyword (can be combined with `variables`).

### Returns
- Scalar `float`, `numpy.ndarray`, `xarray.DataArray`, `xarray.Dataset`, or a `tuple` of results when `expr` contains multiple comma-separated expressions.

---

## 2. `Parser` & `ParserBase`

```python
class Parser(ParserBase):
    def __init__(self, expr: str = "") -> None: ...
```

The floating-point and array parser class. Supports both `snake_case` and `CamelCase` method names.

### Expression & Evaluation Methods

| Pythonic (`snake_case`) | C++ (`CamelCase`) | Description |
| :--- | :--- | :--- |
| `set_expr(expr: str)` | `SetExpr(expr: str)` | Set the expression string and mark the parser for re-compilation. |
| `get_expr() -> str` | `GetExpr() -> str` | Return the current expression string. |
| `eval() -> Any` | `Eval() -> Any` | Evaluate the expression and return the result (or last result if comma-separated). |
| `eval_multi() -> list[Any]` | `EvalMulti() -> list[Any]` | Evaluate all comma-separated sub-expressions and return a list of results. |
| `get_num_results() -> int` | `GetNumResults() -> int` | Return the number of sub-expressions on the stack after compilation. |
| `diff(var, pos=None, eps=None)` | `Diff(var, pos=None, eps=None)` | Compute the 5-point 4th-order numerical derivative with respect to `var` (variable name `str` or [`VariableRef`](../pymu_eval/token_reader.py)). Works on scalars, `np.ndarray`, and `xr.DataArray`. |

### Variable & Constant Management

| Pythonic (`snake_case`) | C++ (`CamelCase`) | Description |
| :--- | :--- | :--- |
| `define_var(name: str, value: Any = 0.0) -> VariableRef` | `DefineVar(name, value)` | Define or update a variable. Accepts scalars, `np.ndarray`, `xr.DataArray`, `xr.Dataset`, or an existing `VariableRef`. Returns the [`VariableRef`](../pymu_eval/token_reader.py). |
| `define_vars(variables: Mapping \| xr.Dataset)` | `DefineVars(variables)` | Define multiple variables from a dictionary or an `xarray.Dataset`. |
| `define_dataset(dataset: xr.Dataset, include_coords: bool = True)` | `DefineDataset(dataset, include_coords=True)` | Bind all `data_vars` (and optionally `coords`) of an `xarray.Dataset` as parser variables. |
| `remove_var(name: str)` | `RemoveVar(name: str)` | Remove a single variable by name. |
| `clear_var()` | `ClearVar()` | Remove all defined variables. |
| `get_var() -> dict[str, VariableRef]` | `GetVar()` | Return the dictionary of defined variables. |
| `get_used_var() -> dict[str, VariableRef]` | `GetUsedVar()` | Parse the expression (without evaluating) and return variables referenced by the expression. |
| `define_const(name: str, val: Any)` | `DefineConst(name, val)` | Define a numeric constant. |
| `define_str_const(name: str, val: str)` | `DefineStrConst(name, val)` | Define a string constant. |
| `clear_const()` | `ClearConst()` | Clear all defined constants. |
| `set_var_factory(factory: Callable, user_data: Any = None)` | `SetVarFactory(factory, user_data=None)` | Register a callback `factory(name) -> VariableRef \| float` invoked automatically when an undefined variable identifier is encountered. |

### Custom Functions & Operators

| Pythonic (`snake_case`) | C++ (`CamelCase`) | Description |
| :--- | :--- | :--- |
| `define_fun(name, fun, allow_opt=True, argc=None)` | `DefineFun(name, fun, allow_opt=True, argc=None)` | Register a custom function. Arity is auto-inspected from `fun`'s signature (`*args` registers a variadic function). |
| `define_fun_userdata(name, fun, user_data, allow_opt=True, argc=None)` | `DefineFunUserData(...)` | Register a custom function whose first parameter receives `user_data`. |
| `define_str_fun(name, fun, argc=None, allow_opt=True)` | `DefineStrFun(...)` | Register a function whose first argument is a string literal (e.g., `fn("label", x, y)`). |
| `define_oprt(name, fun, prec=0, associativity=oaLEFT, allow_opt=False)` | `DefineOprt(...)` | Register a custom binary operator with explicit precedence and associativity. |
| `define_infix_oprt(name, fun, prec=prINFIX, allow_opt=True)` | `DefineInfixOprt(...)` | Register a custom unary prefix operator (e.g., `!a`, `~a`). |
| `define_postfix_oprt(name, fun, allow_opt=True)` | `DefinePostfixOprt(...)` | Register a custom unary postfix operator (e.g., `a!`, `10m`, `5k`). |
| `has_fun(name) -> bool` | `HasFun(name) -> bool` | Check if a function is registered. |
| `has_oprt(name) -> bool` | `HasOprt(name) -> bool` | Check if a binary operator is registered. |
| `has_infix_oprt(name) -> bool` | `HasInfixOprt(name) -> bool` | Check if a prefix operator is registered. |
| `has_postfix_oprt(name) -> bool` | `HasPostfixOprt(name) -> bool` | Check if a postfix operator is registered. |

### Optimizer & Localization

| Pythonic (`snake_case`) | C++ (`CamelCase`) | Description |
| :--- | :--- | :--- |
| `enable_optimizer(enable: bool = True)` | `EnableOptimizer(enable=True)` | Enable or disable compile-time constant folding and algebraic bytecode optimizations. |
| `set_arg_sep(sep: str)` | `SetArgSep(sep: str)` | Set the function argument separator character (default `','`). |
| `set_dec_sep(sep: str)` | `SetDecSep(sep: str)` | Set the decimal separator character (default `'.'`). |
| `set_thousands_sep(sep: str = "")` | `SetThousandsSep(sep="")` | Set the thousands group separator character (default `''`). |
| `reset_locale()` | `ResetLocale()` | Reset separators to defaults (`arg_sep=','`, `dec_sep='.'`, `thousands_sep=''`). |

---

## 3. `VariableRef` (aliased as `Variable`)

```python
class VariableRef:
    def __init__(self, value: Any = 0.0, name: str = "") -> None: ...
    @property
    def value(self) -> Any: ...
    @value.setter
    def value(self, val: Any) -> None: ...
```

A mutable reference container (analogous to `value_type*` in C++ `muParser`). Updating `var_ref.value = new_array` immediately updates the value seen by `parser.eval()` without recompiling the expression bytecode.

---

## 4. `ParserError`

```python
class ParserError(Exception): ...
```

Raised on syntax, definition, or expression errors.

### Methods & Properties
- `err.code` / `err.GetCode() -> EErrorCodes`: The [`EErrorCodes`](../pymu_eval/definitions.py) enum value (e.g., `EErrorCodes.ecUNEXPECTED_OPERATOR`).
- `err.msg` / `err.GetMsg() -> str`: Formatted human-readable error message.
- `err.expr` / `err.GetExpr() -> str`: The expression being parsed when the error occurred.
- `err.token` / `err.GetToken() -> str`: The offending token string.
- `err.pos` / `err.GetPos() -> int`: 0-indexed character position in the expression where the error occurred.
