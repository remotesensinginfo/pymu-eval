"""
Pytest test suite for evaluating muParser mathematical expressions
using NumPy ndarrays, xarray DataArrays, and xarray Datasets.
"""

import numpy as np
import pytest
import xarray as xr

import pymu_eval as mu
from pymu_eval import Parser, ParserError, ParserInt, eval_expr, evaluate


# ===========================================================================
# NumPy ndarray Tests
# ===========================================================================


class TestNumPyEvaluation:
    def test_numpy_basic_arithmetic_and_powers(self):
        a = np.array([1.0, 2.0, 3.0, 4.0])
        b = np.array([2.0, 3.0, 4.0, 5.0])

        p = Parser()
        p.define_var("a", a)
        p.define_var("b", b)

        p.set_expr("2*a + 3*b - a^2 + b^3 / 2")
        res = p.eval()
        expected = 2 * a + 3 * b - a**2 + (b**3) / 2
        np.testing.assert_allclose(res, expected)

        # Test negative exponent on integer numpy array
        int_arr = np.array([1, 2, 4, 8])
        p.set_var("a", int_arr)
        p.set_expr("a^-1")
        np.testing.assert_allclose(p.eval(), 1.0 / int_arr)

        # Test 0 exponent on numpy array preserves array shape
        p.set_expr("a^0")
        res_zero = p.eval()
        assert isinstance(res_zero, np.ndarray)
        assert res_zero.shape == int_arr.shape
        np.testing.assert_allclose(res_zero, np.ones_like(int_arr, dtype=float))

    def test_numpy_all_builtin_math_functions(self):
        x = np.linspace(0.2, 0.8, 10)
        y = np.linspace(1.2, 2.5, 10)

        cases = [
            ("sin(x) + cos(x)", np.sin(x) + np.cos(x)),
            ("tan(x) * cot(x)", np.ones_like(x)),
            ("asin(x) + acos(x)", np.arcsin(x) + np.arccos(x)),
            ("atan(x)", np.arctan(x)),
            ("atan2(y, x)", np.arctan2(y, x)),
            ("sinh(x) + cosh(x)", np.sinh(x) + np.cosh(x)),
            ("tanh(x) * coth(x)", np.ones_like(x)),
            ("asinh(x) + acosh(y) + atanh(x)", np.arcsinh(x) + np.arccosh(y) + np.arctanh(x)),
            ("log(y) + ln(y) + log2(y) + log10(y)", 2 * np.log(y) + np.log2(y) + np.log10(y)),
            ("exp(x) + sqrt(y)", np.exp(x) + np.sqrt(y)),
            ("abs(-x) + sign(x - 0.5)", np.abs(-x) + np.sign(x - 0.5)),
            ("rint(y)", np.floor(y + 0.5)),
        ]

        for expr, expected in cases:
            res = eval_expr(expr, x=x, y=y)
            np.testing.assert_allclose(res, expected, err_msg=f"Failed on {expr}")

    def test_numpy_variadic_functions(self):
        a = np.array([[1.0, 10.0], [20.0, 2.0]])
        b = np.array([[5.0, 4.0], [15.0, 8.0]])
        c = np.array([[3.0, 12.0], [10.0, 6.0]])

        np.testing.assert_allclose(
            eval_expr("sum(a, b, c)", a=a, b=b, c=c),
            a + b + c,
        )
        np.testing.assert_allclose(
            eval_expr("avg(a, b, c)", a=a, b=b, c=c),
            (a + b + c) / 3.0,
        )
        np.testing.assert_allclose(
            eval_expr("min(a, b, c)", a=a, b=b, c=c),
            np.minimum(np.minimum(a, b), c),
        )
        np.testing.assert_allclose(
            eval_expr("max(a, b, c)", a=a, b=b, c=c),
            np.maximum(np.maximum(a, b), c),
        )
        # Mixed array and scalar in variadic functions
        np.testing.assert_allclose(
            eval_expr("min(a, 5)", a=a),
            np.minimum(a, 5.0),
        )
        np.testing.assert_allclose(
            eval_expr("max(a, 5)", a=a),
            np.maximum(a, 5.0),
        )

    def test_numpy_comparisons_and_logical_operators(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([5.0, 4.0, 3.0, 2.0, 1.0])

        np.testing.assert_allclose(eval_expr("a < b", a=a, b=b), [1, 1, 0, 0, 0])
        np.testing.assert_allclose(eval_expr("a <= b", a=a, b=b), [1, 1, 1, 0, 0])
        np.testing.assert_allclose(eval_expr("a > b", a=a, b=b), [0, 0, 0, 1, 1])
        np.testing.assert_allclose(eval_expr("a >= b", a=a, b=b), [0, 0, 1, 1, 1])
        np.testing.assert_allclose(eval_expr("a == b", a=a, b=b), [0, 0, 1, 0, 0])
        np.testing.assert_allclose(eval_expr("a != b", a=a, b=b), [1, 1, 0, 1, 1])

        np.testing.assert_allclose(
            eval_expr("(a >= 2) && (b >= 2)", a=a, b=b),
            [0, 1, 1, 1, 0],
        )
        np.testing.assert_allclose(
            eval_expr("(a == 1) || (b == 1)", a=a, b=b),
            [1, 0, 0, 0, 1],
        )

    def test_numpy_ternary_conditional_and_nested(self):
        a = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        b = np.array([10.0, 20.0, 30.0, 40.0, 50.0])

        # Element-wise ternary
        res = eval_expr("a > 0 ? a * b : -b", a=a, b=b)
        expected = np.where(a > 0, a * b, -b)
        np.testing.assert_allclose(res, expected)

        # Nested element-wise ternary (piecewise function)
        res_nested = eval_expr("a < 0 ? -100 : (a == 0 ? 0 : a * 100)", a=a)
        expected_nested = np.where(a < 0, -100.0, np.where(a == 0, 0.0, a * 100.0))
        np.testing.assert_allclose(res_nested, expected_nested)

        # Scalar condition with array branches
        res_scalar_cond = eval_expr("1 < 2 ? a + b : a - b", a=a, b=b)
        np.testing.assert_allclose(res_scalar_cond, a + b)

    def test_numpy_2d_and_3d_broadcasting(self):
        cube = np.arange(24, dtype=float).reshape((2, 3, 4))
        layer = np.ones((3, 4), dtype=float) * 2.0

        res = eval_expr("cube * layer + _pi", cube=cube, layer=layer)
        assert res.shape == (2, 3, 4)
        np.testing.assert_allclose(res, cube * layer + np.pi)

    def test_numpy_diff(self):
        x = np.linspace(-2.0, 2.0, 20)
        p = Parser("sin(x) + x^2")
        p.define_var("x", x)
        deriv = p.diff("x", x)
        np.testing.assert_allclose(deriv, np.cos(x) + 2 * x, rtol=1e-5, atol=1e-6)


# ===========================================================================
# xarray DataArray & Dataset Tests
# ===========================================================================


class TestXarrayEvaluation:
    @pytest.fixture
    def sample_dataset(self) -> xr.Dataset:
        rng = np.random.default_rng(42)
        lat = np.linspace(-10.0, 10.0, 5)
        lon = np.linspace(100.0, 120.0, 6)
        time = ["2026-01-01", "2026-06-01"]

        red = xr.DataArray(
            rng.uniform(0.05, 0.3, size=(2, 5, 6)),
            dims=("time", "lat", "lon"),
            coords={"time": time, "lat": lat, "lon": lon},
            name="red",
        )
        nir = xr.DataArray(
            rng.uniform(0.2, 0.8, size=(2, 5, 6)),
            dims=("time", "lat", "lon"),
            coords={"time": time, "lat": lat, "lon": lon},
            name="nir",
        )
        blue = xr.DataArray(
            rng.uniform(0.02, 0.15, size=(2, 5, 6)),
            dims=("time", "lat", "lon"),
            coords={"time": time, "lat": lat, "lon": lon},
            name="blue",
        )
        cloud_mask = xr.DataArray(
            rng.integers(0, 2, size=(2, 5, 6)),
            dims=("time", "lat", "lon"),
            coords={"time": time, "lat": lat, "lon": lon},
            name="cloud_mask",
        )
        return xr.Dataset(
            {
                "red": red,
                "nir": nir,
                "blue": blue,
                "cloud_mask": cloud_mask,
            }
        )

    def test_xarray_dataarray_arithmetic_and_functions(self, sample_dataset: xr.Dataset):
        red = sample_dataset["red"]
        nir = sample_dataset["nir"]

        ndvi = eval_expr("(nir - red) / (nir + red)", red=red, nir=nir)
        assert isinstance(ndvi, xr.DataArray)
        assert ndvi.dims == ("time", "lat", "lon")
        xr.testing.assert_allclose(ndvi, (nir - red) / (nir + red))

        # Trig & log functions on DataArray
        trig_res = eval_expr("sin(red)^2 + cos(red)^2 + log(nir + 1)", red=red, nir=nir)
        assert isinstance(trig_res, xr.DataArray)
        xr.testing.assert_allclose(trig_res, 1.0 + np.log(nir + 1))

    def test_xarray_dataset_direct_evaluation(self, sample_dataset: xr.Dataset):
        # Pass the entire xr.Dataset directly to eval_expr / Parser.eval
        evi_expr = "2.5 * ((nir - red) / (nir + 6 * red - 7.5 * blue + 1))"
        evi = eval_expr(evi_expr, sample_dataset)

        expected_evi = 2.5 * (
            (sample_dataset["nir"] - sample_dataset["red"])
            / (
                sample_dataset["nir"]
                + 6 * sample_dataset["red"]
                - 7.5 * sample_dataset["blue"]
                + 1
            )
        )
        assert isinstance(evi, xr.DataArray)
        xr.testing.assert_allclose(evi, expected_evi)

        # Also test stateful Parser with Dataset
        p = Parser(evi_expr)
        evi2 = p.eval(sample_dataset)
        xr.testing.assert_allclose(evi2, expected_evi)

    def test_xarray_dataset_coordinates_in_expression(self, sample_dataset: xr.Dataset):
        # Coordinates ("lat", "lon") from the Dataset should be usable in expressions
        res = eval_expr("nir * cos(lat * _pi / 180)", sample_dataset)
        expected = sample_dataset["nir"] * np.cos(sample_dataset.coords["lat"] * np.pi / 180.0)
        assert isinstance(res, xr.DataArray)
        xr.testing.assert_allclose(res, expected)

    def test_xarray_ternary_masking_and_conditionals(self, sample_dataset: xr.Dataset):
        # Compute NDVI where cloud_mask == 0, else -9999
        expr = "cloud_mask == 0 ? (nir - red) / (nir + red) : -9999"
        masked_ndvi = evaluate(expr, sample_dataset)

        raw_ndvi = (sample_dataset["nir"] - sample_dataset["red"]) / (
            sample_dataset["nir"] + sample_dataset["red"]
        )
        expected = xr.where(sample_dataset["cloud_mask"] == 0, raw_ndvi, -9999.0)
        assert isinstance(masked_ndvi, xr.DataArray)
        xr.testing.assert_allclose(masked_ndvi, expected)

    def test_xarray_variadic_functions_and_broadcasting(self, sample_dataset: xr.Dataset):
        res_max = eval_expr("max(red, nir, blue)", sample_dataset)
        expected_max = np.maximum(
            np.maximum(sample_dataset["red"], sample_dataset["nir"]),
            sample_dataset["blue"],
        )
        xr.testing.assert_allclose(res_max, expected_max)

        res_avg = eval_expr("avg(red, nir, blue)", sample_dataset)
        expected_avg = (
            sample_dataset["red"] + sample_dataset["nir"] + sample_dataset["blue"]
        ) / 3.0
        xr.testing.assert_allclose(res_avg, expected_avg)

    def test_xarray_assignment_and_multi_expressions(self, sample_dataset: xr.Dataset):
        p = Parser()
        p.define_vars(sample_dataset)
        p.define_var("ndvi", 0.0)
        p.define_var("savi", 0.0)

        p.set_expr(
            "ndvi = (nir - red) / (nir + red), "
            "savi = 1.5 * (nir - red) / (nir + red + 0.5), "
            "max(ndvi, savi)"
        )
        results = p.eval_multi()
        assert len(results) == 3

        expected_ndvi = (sample_dataset["nir"] - sample_dataset["red"]) / (
            sample_dataset["nir"] + sample_dataset["red"]
        )
        expected_savi = (
            1.5
            * (sample_dataset["nir"] - sample_dataset["red"])
            / (sample_dataset["nir"] + sample_dataset["red"] + 0.5)
        )
        xr.testing.assert_allclose(results[0], expected_ndvi)
        xr.testing.assert_allclose(results[1], expected_savi)
        xr.testing.assert_allclose(results[2], np.maximum(expected_ndvi, expected_savi))
        xr.testing.assert_allclose(p["ndvi"], expected_ndvi)
        xr.testing.assert_allclose(p["savi"], expected_savi)

    def test_xarray_whole_dataset_as_single_variable(self, sample_dataset: xr.Dataset):
        # Bind an entire xr.Dataset to a single variable name "ds"
        ds_sub = sample_dataset[["red", "nir", "blue"]]
        res = eval_expr("2 * ds + 1", ds=ds_sub)
        assert isinstance(res, xr.Dataset)
        xr.testing.assert_allclose(res, 2 * ds_sub + 1)

    def test_xarray_diff(self, sample_dataset: xr.Dataset):
        red = sample_dataset["red"]
        p = Parser("red^3 + sin(red)")
        p.define_var("red", red)
        deriv = p.diff("red", red)
        assert isinstance(deriv, xr.DataArray)
        xr.testing.assert_allclose(deriv, 3 * red**2 + np.cos(red), rtol=1e-5, atol=1e-6)
