"""
Mathematical and operator implementations supporting scalars, NumPy arrays,
xarray DataArrays, and xarray Datasets.
Mirrors muParserTemplateMagic.h and muParserInt.cpp.
"""

# pylint: disable=invalid-name,too-many-public-methods

import math
import random
from typing import Any
import numpy as np

try:
    import xarray as xr
    HAS_XARRAY = True
except ImportError:  # pragma: no cover
    xr = None  # type: ignore
    HAS_XARRAY = False

from .definitions import EErrorCodes
from .errors import ParserError

INT32_MIN = -2147483648
INT32_MAX = 2147483647


def is_xarray(val: Any) -> bool:
    """Return True if val is an xarray DataArray or Dataset."""
    if not HAS_XARRAY:
        return False
    return isinstance(val, (xr.DataArray, xr.Dataset))


def is_array_like(val: Any) -> bool:
    """Return True if val is a non-scalar NumPy ndarray or xarray object."""
    if is_xarray(val):
        return getattr(val, "ndim", 1) > 0 or isinstance(val, xr.Dataset)
    if isinstance(val, np.ndarray):
        return val.ndim > 0
    return False


def _is_any_array(*vals: Any) -> bool:
    """Return True if any argument is an xarray object or NumPy ndarray."""
    return any(is_xarray(v) or isinstance(v, np.ndarray) for v in vals)


def _to_scalar_or_keep(val: Any, was_scalar: bool) -> Any:
    """Convert 0-d numpy output back to Python float if input was scalar."""
    if was_scalar and not is_xarray(val) and not isinstance(val, np.ndarray):
        return float(val)
    if was_scalar and isinstance(val, np.generic):
        return float(val)
    return val


def _apply_ufunc_1(ufunc: Any, v: Any) -> Any:
    """Apply a 1-arg numpy ufunc preserving array types and returning float for scalars."""
    scalar = not _is_any_array(v)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        res = ufunc(v)
    return _to_scalar_or_keep(res, scalar)


class MathImpl:
    """
    Standard floating-point and array math implementation.
    Mirrors mu::MathImpl<double> while supporting NumPy and xarray objects.
    """

    CONST_PI: float = 3.141592653589793238462643
    CONST_E: float = 2.718281828459045235360287

    @staticmethod
    def Sin(v: Any) -> Any:
        """Compute sine of v."""
        return _apply_ufunc_1(np.sin, v)

    @staticmethod
    def Cos(v: Any) -> Any:
        """Compute cosine of v."""
        return _apply_ufunc_1(np.cos, v)

    @staticmethod
    def Tan(v: Any) -> Any:
        """Compute tangent of v."""
        return _apply_ufunc_1(np.tan, v)

    @staticmethod
    def Cot(v: Any) -> Any:
        """Compute cotangent of v."""
        scalar = not _is_any_array(v)
        with np.errstate(divide="ignore", invalid="ignore"):
            res = 1.0 / np.tan(v)
        return _to_scalar_or_keep(res, scalar)

    @staticmethod
    def ASin(v: Any) -> Any:
        """Compute arc sine of v."""
        return _apply_ufunc_1(np.arcsin, v)

    @staticmethod
    def ACos(v: Any) -> Any:
        """Compute arc cosine of v."""
        return _apply_ufunc_1(np.arccos, v)

    @staticmethod
    def ATan(v: Any) -> Any:
        """Compute arc tangent of v."""
        return _apply_ufunc_1(np.arctan, v)

    @staticmethod
    def ATan2(v1: Any, v2: Any) -> Any:
        """Compute two-argument arc tangent atan2(v1, v2)."""
        scalar = not _is_any_array(v1, v2)
        with np.errstate(divide="ignore", invalid="ignore"):
            res = np.arctan2(v1, v2)
        return _to_scalar_or_keep(res, scalar)

    @staticmethod
    def Sinh(v: Any) -> Any:
        """Compute hyperbolic sine of v."""
        return _apply_ufunc_1(np.sinh, v)

    @staticmethod
    def Cosh(v: Any) -> Any:
        """Compute hyperbolic cosine of v."""
        return _apply_ufunc_1(np.cosh, v)

    @staticmethod
    def Tanh(v: Any) -> Any:
        """Compute hyperbolic tangent of v."""
        return _apply_ufunc_1(np.tanh, v)

    @staticmethod
    def Coth(v: Any) -> Any:
        """Compute hyperbolic cotangent of v."""
        scalar = not _is_any_array(v)
        with np.errstate(divide="ignore", invalid="ignore"):
            res = 1.0 / np.tanh(v)
        return _to_scalar_or_keep(res, scalar)

    @staticmethod
    def ASinh(v: Any) -> Any:
        """Compute inverse hyperbolic sine of v."""
        return _apply_ufunc_1(np.arcsinh, v)

    @staticmethod
    def ACosh(v: Any) -> Any:
        """Compute inverse hyperbolic cosine of v."""
        return _apply_ufunc_1(np.arccosh, v)

    @staticmethod
    def ATanh(v: Any) -> Any:
        """Compute inverse hyperbolic tangent of v."""
        return _apply_ufunc_1(np.arctanh, v)

    @staticmethod
    def Log(v: Any) -> Any:
        """Compute natural logarithm of v."""
        return _apply_ufunc_1(np.log, v)

    @staticmethod
    def Log2(v: Any) -> Any:
        """Compute base-2 logarithm of v."""
        return _apply_ufunc_1(np.log2, v)

    @staticmethod
    def Log10(v: Any) -> Any:
        """Compute base-10 logarithm of v."""
        return _apply_ufunc_1(np.log10, v)

    @staticmethod
    def Exp(v: Any) -> Any:
        """Compute exponential exp(v)."""
        return _apply_ufunc_1(np.exp, v)

    @staticmethod
    def Abs(v: Any) -> Any:
        """Compute absolute value of v."""
        return _apply_ufunc_1(np.abs, v)

    @staticmethod
    def Sqrt(v: Any) -> Any:
        """Compute square root of v."""
        return _apply_ufunc_1(np.sqrt, v)

    @staticmethod
    def Rint(v: Any) -> Any:
        """Round v to nearest integer."""
        scalar = not _is_any_array(v)
        with np.errstate(invalid="ignore"):
            res = np.floor(v + 0.5)
        return _to_scalar_or_keep(res, scalar)

    @staticmethod
    def Sign(v: Any) -> Any:
        """Compute sign of v (-1, 0, or 1)."""
        scalar = not _is_any_array(v)
        if scalar:
            if v < 0:
                return -1.0
            if v > 0:
                return 1.0
            if v == 0:
                return 0.0
            return float("nan")
        with np.errstate(invalid="ignore"):
            res = np.sign(v)
            if _is_any_array(res):
                return res.astype(float)
            return res

    @staticmethod
    def Pow(v1: Any, v2: Any) -> Any:
        """Compute v1 raised to the power v2."""
        scalar = not _is_any_array(v1, v2)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            if scalar:
                return float(np.power(float(v1), float(v2)))
            # Ensure integer arrays raised to negative powers work as float
            if isinstance(v1, np.ndarray) and np.issubdtype(v1.dtype, np.integer):
                v1 = v1.astype(float)
            elif (
                is_xarray(v1)
                and isinstance(v1, xr.DataArray)
                and np.issubdtype(v1.dtype, np.integer)
            ):
                v1 = v1.astype(float)
            return v1 ** v2

    @staticmethod
    def Rnd() -> float:
        """Return a uniform random float in [0, 1)."""
        return random.random()

    @staticmethod
    def UnaryMinus(v: Any) -> Any:
        """Negate v."""
        return -v

    @staticmethod
    def UnaryPlus(v: Any) -> Any:
        """Unary plus on v."""
        return +v

    @staticmethod
    def Sum(*args: Any) -> Any:
        """Compute sum of arguments."""
        if not args:
            raise ParserError("too few arguments for function sum.")
        res = args[0]
        for a in args[1:]:
            res = res + a
        return res

    @staticmethod
    def Avg(*args: Any) -> Any:
        """Compute arithmetic mean of arguments."""
        if not args:
            raise ParserError("too few arguments for function avg.")
        res = args[0]
        for a in args[1:]:
            res = res + a
        return res / float(len(args))

    @staticmethod
    def Min(*args: Any) -> Any:
        """Compute element-wise minimum across arguments."""
        if not args:
            raise ParserError("too few arguments for function min.")
        res = args[0]
        for a in args[1:]:
            if not _is_any_array(res, a):
                res = res if res < a else a
            else:
                res = np.minimum(res, a)
        return res

    @staticmethod
    def Max(*args: Any) -> Any:
        """Compute element-wise maximum across arguments."""
        if not args:
            raise ParserError("too few arguments for function max.")
        res = args[0]
        for a in args[1:]:
            if not _is_any_array(res, a):
                res = res if res > a else a
            else:
                res = np.maximum(res, a)
        return res

    # Binary operator helpers for both scalars and arrays/xarray
    @staticmethod
    def Div(v1: Any, v2: Any) -> Any:
        """Divide v1 by v2."""
        scalar = not _is_any_array(v1, v2)
        with np.errstate(divide="ignore", invalid="ignore"):
            if scalar:
                return float(np.true_divide(float(v1), float(v2)))
            return v1 / v2

    @staticmethod
    def _bool_to_num(res: Any) -> Any:
        if _is_any_array(res):
            return res.astype(float)
        return 1.0 if res else 0.0

    @staticmethod
    def LE(v1: Any, v2: Any) -> Any:
        """Return 1.0 where v1 <= v2, else 0.0."""
        with np.errstate(invalid="ignore"):
            return MathImpl._bool_to_num(v1 <= v2)

    @staticmethod
    def GE(v1: Any, v2: Any) -> Any:
        """Return 1.0 where v1 >= v2, else 0.0."""
        with np.errstate(invalid="ignore"):
            return MathImpl._bool_to_num(v1 >= v2)

    @staticmethod
    def NEQ(v1: Any, v2: Any) -> Any:
        """Return 1.0 where v1 != v2, else 0.0."""
        with np.errstate(invalid="ignore"):
            return MathImpl._bool_to_num(v1 != v2)

    @staticmethod
    def EQ(v1: Any, v2: Any) -> Any:
        """Return 1.0 where v1 == v2, else 0.0."""
        with np.errstate(invalid="ignore"):
            return MathImpl._bool_to_num(v1 == v2)

    @staticmethod
    def LT(v1: Any, v2: Any) -> Any:
        """Return 1.0 where v1 < v2, else 0.0."""
        with np.errstate(invalid="ignore"):
            return MathImpl._bool_to_num(v1 < v2)

    @staticmethod
    def GT(v1: Any, v2: Any) -> Any:
        """Return 1.0 where v1 > v2, else 0.0."""
        with np.errstate(invalid="ignore"):
            return MathImpl._bool_to_num(v1 > v2)

    @staticmethod
    def LAND(v1: Any, v2: Any) -> Any:
        """Logical AND of v1 and v2."""
        if _is_any_array(v1, v2):
            return ((v1 != 0) & (v2 != 0)).astype(float)
        return 1.0 if ((v1 != 0) and (v2 != 0)) else 0.0

    @staticmethod
    def LOR(v1: Any, v2: Any) -> Any:
        """Logical OR of v1 and v2."""
        if _is_any_array(v1, v2):
            return ((v1 != 0) | (v2 != 0)).astype(float)
        return 1.0 if ((v1 != 0) or (v2 != 0)) else 0.0

    @staticmethod
    def Where(cond: Any, then_val: Any, else_val: Any) -> Any:
        """Select then_val where cond != 0, else else_val."""
        cond_bool = cond != 0
        if is_xarray(cond) or is_xarray(then_val) or is_xarray(else_val):
            return xr.where(cond_bool, then_val, else_val)
        if _is_any_array(cond, then_val, else_val):
            return np.where(cond_bool, then_val, else_val)
        return then_val if cond_bool else else_val


class ParserIntMathImpl:
    """
    Integer math implementation for ParserInt.
    Mirrors mu::ParserInt methods in muParserInt.h / muParserInt.cpp.
    """

    @staticmethod
    def Round(v: Any) -> Any:
        """Round v to signed 32-bit integer."""
        if _is_any_array(v):
            r = np.where(v >= 0, v + 0.5, v - 0.5)
            return r.astype(np.int64)
        vf = float(v)
        if math.isnan(vf) or math.isinf(vf):
            raise ParserError(EErrorCodes.ecDOMAIN_ERROR)
        r = vf + (0.5 if vf >= 0 else -0.5)
        if not float(INT32_MIN) - 1.0 < r < float(INT32_MAX) + 1.0:
            raise ParserError(EErrorCodes.ecDOMAIN_ERROR)
        return int(r)

    @classmethod
    def Abs(cls, v: Any) -> Any:
        """Integer absolute value."""
        if _is_any_array(v):
            return np.abs(cls.Round(v))
        return float(cls.Round(abs(float(v))))

    @classmethod
    def Sign(cls, v: Any) -> Any:
        """Integer sign (-1, 0, or 1)."""
        rv = cls.Round(v)
        if _is_any_array(rv):
            return np.sign(rv)
        return -1.0 if rv < 0 else (1.0 if rv > 0 else 0.0)

    @classmethod
    def Ite(cls, v1: Any, v2: Any, v3: Any) -> Any:
        """Integer if-then-else."""
        r1 = cls.Round(v1)
        r2 = cls.Round(v2)
        r3 = cls.Round(v3)
        if is_xarray(r1) or is_xarray(r2) or is_xarray(r3):
            return xr.where(r1 == 1, r2, r3)
        if _is_any_array(r1, r2, r3):
            return np.where(r1 == 1, r2, r3)
        return float(r2 if r1 == 1 else r3)

    @classmethod
    def Add(cls, v1: Any, v2: Any) -> Any:
        """Integer addition with 32-bit overflow check."""
        r1, r2 = cls.Round(v1), cls.Round(v2)
        if _is_any_array(r1, r2):
            return r1 + r2
        r = r1 + r2
        if r > INT32_MAX or r < INT32_MIN:
            raise ParserError(EErrorCodes.ecDOMAIN_ERROR)
        return float(r)

    @classmethod
    def Sub(cls, v1: Any, v2: Any) -> Any:
        """Integer subtraction with 32-bit overflow check."""
        r1, r2 = cls.Round(v1), cls.Round(v2)
        if _is_any_array(r1, r2):
            return r1 - r2
        r = r1 - r2
        if r > INT32_MAX or r < INT32_MIN:
            raise ParserError(EErrorCodes.ecDOMAIN_ERROR)
        return float(r)

    @classmethod
    def Mul(cls, v1: Any, v2: Any) -> Any:
        """Integer multiplication with 32-bit overflow check."""
        r1, r2 = cls.Round(v1), cls.Round(v2)
        if _is_any_array(r1, r2):
            return r1 * r2
        r = r1 * r2
        if r > INT32_MAX or r < INT32_MIN:
            raise ParserError(EErrorCodes.ecDOMAIN_ERROR)
        return float(r)

    @classmethod
    def Div(cls, v1: Any, v2: Any) -> Any:
        """Integer division truncating toward zero."""
        r1, r2 = cls.Round(v1), cls.Round(v2)
        if _is_any_array(r1, r2):
            if np.any(r2 == 0):
                raise ParserError(EErrorCodes.ecDIV_BY_ZERO)
            return np.trunc(r1 / r2).astype(np.int64)
        if r2 == 0:
            raise ParserError(EErrorCodes.ecDIV_BY_ZERO)
        if r1 == INT32_MIN and r2 == -1:
            raise ParserError(EErrorCodes.ecDOMAIN_ERROR)
        # C++ integer division truncates toward zero
        return float(int(r1 / r2))

    @classmethod
    def Mod(cls, v1: Any, v2: Any) -> Any:
        """Integer modulo matching C++ remainder semantics."""
        r1, r2 = cls.Round(v1), cls.Round(v2)
        if _is_any_array(r1, r2):
            if np.any(r2 == 0):
                raise ParserError(EErrorCodes.ecDIV_BY_ZERO)
            return np.fmod(r1, r2)
        if r2 == 0:
            raise ParserError(EErrorCodes.ecDIV_BY_ZERO)
        if r1 == INT32_MIN and r2 == -1:
            raise ParserError(EErrorCodes.ecDOMAIN_ERROR)
        return float(r1 - int(r1 / r2) * r2)

    @classmethod
    def Shr(cls, v1: Any, v2: Any) -> Any:
        """Integer right bit-shift."""
        r1, shift = cls.Round(v1), cls.Round(v2)
        if _is_any_array(r1, shift):
            if np.any((shift < 0) | (shift >= 32)):
                raise ParserError(EErrorCodes.ecDOMAIN_ERROR)
            return r1 >> shift
        if shift < 0 or shift >= 32:
            raise ParserError(EErrorCodes.ecDOMAIN_ERROR)
        # Unsigned 32-bit shift right then cast to signed 32-bit int
        u = (r1 & 0xFFFFFFFF) >> shift
        if u & 0x80000000:
            u -= 0x100000000
        return float(u)

    @classmethod
    def Shl(cls, v1: Any, v2: Any) -> Any:
        """Integer left bit-shift."""
        r1, shift = cls.Round(v1), cls.Round(v2)
        if _is_any_array(r1, shift):
            if np.any((shift < 0) | (shift >= 32)):
                raise ParserError(EErrorCodes.ecDOMAIN_ERROR)
            return r1 << shift
        if shift < 0 or shift >= 32:
            raise ParserError(EErrorCodes.ecDOMAIN_ERROR)
        u = ((r1 & 0xFFFFFFFF) << shift) & 0xFFFFFFFF
        if u & 0x80000000:
            u -= 0x100000000
        return float(u)

    @classmethod
    def BitAnd(cls, v1: Any, v2: Any) -> Any:
        """Bitwise AND."""
        r1, r2 = cls.Round(v1), cls.Round(v2)
        if _is_any_array(r1, r2):
            return r1 & r2
        return float(r1 & r2)

    @classmethod
    def BitOr(cls, v1: Any, v2: Any) -> Any:
        """Bitwise OR."""
        r1, r2 = cls.Round(v1), cls.Round(v2)
        if _is_any_array(r1, r2):
            return r1 | r2
        return float(r1 | r2)

    @classmethod
    def And(cls, v1: Any, v2: Any) -> Any:
        """Logical AND on integers."""
        r1, r2 = cls.Round(v1), cls.Round(v2)
        if _is_any_array(r1, r2):
            return ((r1 != 0) & (r2 != 0)).astype(np.int64)
        return 1.0 if (bool(r1) and bool(r2)) else 0.0

    @classmethod
    def Or(cls, v1: Any, v2: Any) -> Any:
        """Logical OR on integers."""
        r1, r2 = cls.Round(v1), cls.Round(v2)
        if _is_any_array(r1, r2):
            return ((r1 != 0) | (r2 != 0)).astype(np.int64)
        return 1.0 if (bool(r1) or bool(r2)) else 0.0

    @classmethod
    def Less(cls, v1: Any, v2: Any) -> Any:
        """Integer less-than comparison."""
        r1, r2 = cls.Round(v1), cls.Round(v2)
        if _is_any_array(r1, r2):
            return (r1 < r2).astype(np.int64)
        return 1.0 if r1 < r2 else 0.0

    @classmethod
    def Greater(cls, v1: Any, v2: Any) -> Any:
        """Integer greater-than comparison."""
        r1, r2 = cls.Round(v1), cls.Round(v2)
        if _is_any_array(r1, r2):
            return (r1 > r2).astype(np.int64)
        return 1.0 if r1 > r2 else 0.0

    @classmethod
    def LessEq(cls, v1: Any, v2: Any) -> Any:
        """Integer less-or-equal comparison."""
        r1, r2 = cls.Round(v1), cls.Round(v2)
        if _is_any_array(r1, r2):
            return (r1 <= r2).astype(np.int64)
        return 1.0 if r1 <= r2 else 0.0

    @classmethod
    def GreaterEq(cls, v1: Any, v2: Any) -> Any:
        """Integer greater-or-equal comparison."""
        r1, r2 = cls.Round(v1), cls.Round(v2)
        if _is_any_array(r1, r2):
            return (r1 >= r2).astype(np.int64)
        return 1.0 if r1 >= r2 else 0.0

    @classmethod
    def Equal(cls, v1: Any, v2: Any) -> Any:
        """Integer equality comparison."""
        r1, r2 = cls.Round(v1), cls.Round(v2)
        if _is_any_array(r1, r2):
            return (r1 == r2).astype(np.int64)
        return 1.0 if r1 == r2 else 0.0

    @classmethod
    def NotEqual(cls, v1: Any, v2: Any) -> Any:
        """Integer inequality comparison."""
        r1, r2 = cls.Round(v1), cls.Round(v2)
        if _is_any_array(r1, r2):
            return (r1 != r2).astype(np.int64)
        return 1.0 if r1 != r2 else 0.0

    @classmethod
    def Not(cls, v: Any) -> Any:
        """Logical NOT on integer."""
        rv = cls.Round(v)
        if _is_any_array(rv):
            return (rv == 0).astype(np.int64)
        return 1.0 if not rv else 0.0

    @classmethod
    def Pow(cls, v1: Any, v2: Any) -> Any:
        """Integer power."""
        r1, r2 = cls.Round(v1), cls.Round(v2)
        if _is_any_array(r1, r2):
            return np.power(r1, r2)
        return float(math.pow(float(r1), float(r2)))

    @classmethod
    def UnaryMinus(cls, v: Any) -> Any:
        """Integer unary negation with INT32_MIN overflow check."""
        r = cls.Round(v)
        if _is_any_array(r):
            return -r
        if r == INT32_MIN:
            raise ParserError(EErrorCodes.ecDOMAIN_ERROR)
        return float(-r)

    @staticmethod
    def Sum(*args: Any) -> Any:
        """Compute sum of arguments."""
        return MathImpl.Sum(*args)

    @staticmethod
    def Min(*args: Any) -> Any:
        """Compute minimum of arguments."""
        return MathImpl.Min(*args)

    @staticmethod
    def Max(*args: Any) -> Any:
        """Compute maximum of arguments."""
        return MathImpl.Max(*args)
