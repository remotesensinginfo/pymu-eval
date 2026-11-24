"""
Regression tests for fixes from the code review: coordinate refresh on parser reuse,
ParserInt shift/rounding semantics, NaN handling in min/max, precedence constants,
bytecode result counts, and error message wording.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

try:
    import xarray as xr
except ImportError:
    xr = None  # type: ignore[assignment]

import pymu_eval as mu
from pymu_eval import (
    ERROR_MESSAGES,
    EErrorCodes,
    EOprtPrecedence,
    Parser,
    ParserError,
    ParserInt,
    ParserIntMathImpl,
)

needs_xarray = pytest.mark.skipif(xr is None, reason="xarray not installed")


# ---------------------------------------------------------------------------
# Coordinate refresh when reusing a parser with a new Dataset
# ---------------------------------------------------------------------------


@needs_xarray
class TestDatasetCoordRefresh:
    @staticmethod
    def _ds(x_vals):
        return xr.Dataset({"a": ("x", [1.0, 2.0])}, coords={"x": x_vals})

    def test_eval_refreshes_coords_on_reuse(self):
        p = Parser("a + x")
        np.testing.assert_allclose(p.eval(self._ds([10.0, 20.0])).values, [11.0, 22.0])
        res = p.eval(self._ds([100.0, 200.0]))
        np.testing.assert_allclose(res.values, [101.0, 202.0])
        np.testing.assert_allclose(res["x"].values, [100.0, 200.0])

    def test_define_vars_refreshes_coords_on_reuse(self):
        p = Parser("a + x")
        p.define_vars(self._ds([10.0, 20.0]))
        p.eval()
        p.define_vars(self._ds([100.0, 200.0]))
        np.testing.assert_allclose(p.eval().values, [101.0, 202.0])

    def test_user_variable_not_overwritten_by_coord(self):
        p = Parser("a + x")
        p.define_var("x", 5.0)
        np.testing.assert_allclose(p.eval(self._ds([10.0, 20.0])).values, [6.0, 7.0])

    def test_explicit_definition_takes_ownership_from_coord(self):
        p = Parser("a + x")
        p.eval(self._ds([10.0, 20.0]))
        p.set_var("x", 5.0)
        np.testing.assert_allclose(p.eval(self._ds([100.0, 200.0])).values, [6.0, 7.0])

    def test_copy_keeps_coord_tracking(self):
        p = Parser("a + x")
        p.eval(self._ds([10.0, 20.0]))
        q = p.copy()
        q.set_expr("a + x")
        np.testing.assert_allclose(q.eval(self._ds([100.0, 200.0])).values, [101.0, 202.0])


# ---------------------------------------------------------------------------
# ParserInt semantics
# ---------------------------------------------------------------------------


class TestParserIntFixes:
    @pytest.mark.parametrize(
        "v1, v2, expected",
        [(-8, 1, -4.0), (8, 1, 4.0), (-1, 31, -1.0), (-2147483648, 1, -1073741824.0)],
    )
    def test_shr_is_arithmetic_for_scalars(self, v1, v2, expected):
        assert ParserIntMathImpl.Shr(v1, v2) == expected

    def test_shr_scalar_matches_array(self):
        vals = np.array([-8.0, -7.0, 7.0, 8.0])
        arr = ParserInt("x>>1").eval(x=vals)
        scal = [ParserInt("x>>1").eval(x=float(v)) for v in vals]
        np.testing.assert_array_equal(arr, scal)

    def test_shl_scalar_matches_array_with_32bit_wrap(self):
        vals = np.array([1.0, -1.0, 1073741824.0])
        arr = ParserInt("x<<1").eval(x=vals)
        scal = [ParserInt("x<<1").eval(x=float(v)) for v in vals]
        np.testing.assert_array_equal(arr, scal)
        assert scal[2] == -2147483648.0

    def test_array_nan_raises_domain_error(self):
        with pytest.raises(ParserError) as exc:
            ParserInt("x+1").eval(x=np.array([1.0, np.nan]))
        assert exc.value.GetCode() == EErrorCodes.ecDOMAIN_ERROR

    def test_array_out_of_int32_range_raises(self):
        with pytest.raises(ParserError) as exc:
            ParserInt("x+1").eval(x=np.array([3.0e9]))
        assert exc.value.GetCode() == EErrorCodes.ecDOMAIN_ERROR

    def test_array_rounding_matches_scalar(self):
        vals = np.array([-2.5, -1.5, -0.4, 0.4, 1.5, 2.5])
        arr = ParserIntMathImpl.Round(vals)
        np.testing.assert_array_equal(arr, [ParserIntMathImpl.Round(float(v)) for v in vals])

    def test_array_pow_negative_exponent(self):
        res = ParserInt("x^y").eval(x=np.array([2.0, 4.0]), y=np.array([-1.0, 2.0]))
        np.testing.assert_allclose(res, [0.5, 16.0])

    @needs_xarray
    def test_xarray_metadata_preserved(self):
        da = xr.DataArray([1.0, 2.0, 3.0], dims="t", coords={"t": [10, 20, 30]})
        res = ParserInt("x*2 + (x>1)").eval(x=da)
        assert isinstance(res, xr.DataArray)
        np.testing.assert_array_equal(res["t"].values, [10, 20, 30])
        np.testing.assert_array_equal(res.values, [2, 5, 7])

    @needs_xarray
    def test_xarray_if_function(self):
        da = xr.DataArray([0.0, 1.0], dims="t")
        res = ParserInt("if(x, 10, 20)").eval(x=da)
        assert isinstance(res, xr.DataArray)
        np.testing.assert_array_equal(res.values, [20, 10])

    @needs_xarray
    def test_xarray_div_by_zero_raises(self):
        da = xr.DataArray([1.0, 0.0], dims="t")
        with pytest.raises(ParserError) as exc:
            ParserInt("4/x").eval(x=da)
        assert exc.value.GetCode() == EErrorCodes.ecDIV_BY_ZERO


# ---------------------------------------------------------------------------
# min / max NaN handling
# ---------------------------------------------------------------------------


class TestMinMaxNaN:
    @pytest.mark.parametrize("expr", ["min(x, 1)", "min(1, x)", "max(x, 1)", "max(1, x)"])
    def test_scalar_nan_propagates(self, expr):
        assert math.isnan(Parser(expr).eval(x=float("nan")))

    @pytest.mark.parametrize("expr", ["min(x, 1)", "max(x, 1)"])
    def test_scalar_matches_array(self, expr):
        arr = Parser(expr).eval(x=np.array([np.nan, 0.5, 2.0]))
        scal = [Parser(expr).eval(x=v) for v in (float("nan"), 0.5, 2.0)]
        np.testing.assert_array_equal(arr, scal)

    def test_scalar_result_is_float(self):
        assert isinstance(Parser("min(x, 1)").eval(x=0.5), float)


# ---------------------------------------------------------------------------
# Precedence constants, bytecode result count, error messages
# ---------------------------------------------------------------------------


class TestMiscFixes:
    def test_precedence_constants_are_not_enum_aliases(self):
        assert EOprtPrecedence.prINFIX == 7
        assert EOprtPrecedence.prPOSTFIX == 7
        assert mu.prINFIX == 7 and mu.prPOSTFIX == 7
        assert "prINFIX" not in EOprtPrecedence.__members__
        assert EOprtPrecedence(7).name == "prMUL_DIV"
        assert Parser("-2^2").eval() == -4.0

    def test_set_byte_code_restores_num_results(self):
        src = Parser("1, 2, 3")
        src.eval()
        dst = Parser()
        dst.set_byte_code(src.get_byte_code())
        assert dst.get_num_results() == 3
        assert dst.eval_multi() == [1.0, 2.0, 3.0]
        assert dst.eval() == 3.0

    def test_error_messages_have_no_known_typos(self):
        text = " ".join(ERROR_MESSAGES.values())
        for typo in ("to small", "identic ", "identifer", "non string", "non printable"):
            assert typo not in text

    def test_error_messages_end_with_punctuation(self):
        for code, msg in ERROR_MESSAGES.items():
            assert msg.rstrip().endswith((".", "!")), code.name

    def test_version_matches_package_metadata(self):
        from importlib.metadata import PackageNotFoundError, version

        try:
            assert version("pymu-eval") == mu.__version__
        except PackageNotFoundError:
            pytest.skip("pymu-eval not installed")
