"""
Comprehensive pytest suite porting muParser's C++ test suite (muParserTest.cpp)
to verify full syntax, operator precedence, functions, error codes, optimizer,
localization, and integer parser compatibility.
"""

import copy
import math
import pytest

import pymu_eval as mu
from pymu_eval import (
    ECmdCode,
    EErrorCodes,
    EOprtAssociativity,
    EOprtPrecedence,
    MathImpl,
    Parser,
    ParserError,
    ParserInt,
    Variable,
)


# ---------------------------------------------------------------------------
# Helper callbacks mirroring muParserTest.h
# ---------------------------------------------------------------------------
def f0():
    return 42.0


def f1of1(v):
    return v


def f1of2(v1, v2):
    return v1


def f2of2(v1, v2):
    return v2


def f1of3(v1, v2, v3):
    return v1


def f2of3(v1, v2, v3):
    return v2


def f3of3(v1, v2, v3):
    return v3


def f1of4(v1, v2, v3, v4):
    return v1


def f2of4(v1, v2, v3, v4):
    return v2


def f3of4(v1, v2, v3, v4):
    return v3


def f4of4(v1, v2, v3, v4):
    return v4


def f1of5(v1, v2, v3, v4, v5):
    return v1


def f2of5(v1, v2, v3, v4, v5):
    return v2


def f3of5(v1, v2, v3, v4, v5):
    return v3


def f4of5(v1, v2, v3, v4, v5):
    return v4


def f5of5(v1, v2, v3, v4, v5):
    return v5


def plus2(v):
    return v + 2.0


def times3(v):
    return v * 3.0


def sqr(v):
    return v * v


def sign_fn(v):
    return -v


def add_fn(v1, v2):
    return v1 + v2


def land_fn(v1, v2):
    return float(int(v1) & int(v2))


def first_arg(*args):
    if not args:
        raise ParserError("too few arguments for function FirstArg.")
    return args[0]


def last_arg(*args):
    if not args:
        raise ParserError("too few arguments for function LastArg.")
    return args[-1]


def ping():
    return 10.0


def value_of(s: str):
    return 123.0


def str_len(s: str):
    return float(len(s))


def str_to_float(s: str):
    try:
        return float(s.strip())
    except ValueError:
        return 0.0


def _parse_int_prefix(s: str) -> int:
    s = s.strip()
    try:
        return int(s)
    except ValueError:
        return 0


def str_fun1(v1: str):
    return float(_parse_int_prefix(v1))


def str_fun2(v1: str, v2):
    return float(_parse_int_prefix(v1) + v2)


def str_fun3(v1: str, v2, v3):
    return float(_parse_int_prefix(v1) + v2 + v3)


def str_fun4(v1: str, v2, v3, v4):
    return float(_parse_int_prefix(v1) + v2 + v3 + v4)


def str_fun5(v1: str, v2, v3, v4, v5):
    return float(_parse_int_prefix(v1) + v2 + v3 + v4 + v5)


def str_fun6(v1: str, v2, v3, v4, v5, v6):
    return float(_parse_int_prefix(v1) + v2 + v3 + v4 + v5 + v6)


def mult_str_fun(s: str, *args):
    if not args:
        raise ParserError("too few arguments for function multstrfun.")
    return float(sum(args))


def mega(v):
    return v * 1e6


def milli(v):
    return v / 1e3


def nano(v):
    return v * 1e-9


def fun_ud0(data):
    return float(data)


def fun_ud1(data, v):
    return float(data) + v


def fun_ud2(data, v1, v2):
    return float(data) + v1 + v2


def fun_ud10(data, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10):
    return float(data) + v1 + v2 + v3 + v4 + v5 + v6 + v7 + v8 + v9 + v10


def str_fun_ud3(data, v1: str, v2, v3):
    return float(data) + _parse_int_prefix(v1) + v2 + v3


def sum_ud(data, *args):
    if not args:
        raise ParserError("too few arguments for function sum.")
    return float(data) + sum(args)


def is_hex_val(expr: str, pos_box: list, val_box: list) -> int:
    if len(expr) < 2 or not expr.startswith("0x"):
        return 0
    hex_chars = []
    for ch in expr[2:]:
        if ch in "0123456789abcdefABCDEF":
            hex_chars.append(ch)
        else:
            break
    if not hex_chars:
        return 0
    val_box[0] = float(int("".join(hex_chars), 16))
    pos_box[0] += 2 + len(hex_chars)
    return 1


def create_configured_parser() -> tuple[Parser, dict[str, Variable]]:
    p = Parser()
    p.DefineConst("pi", MathImpl.CONST_PI)
    p.DefineConst("e", MathImpl.CONST_E)
    p.DefineConst("const", 1.0)
    p.DefineConst("const1", 2.0)
    p.DefineConst("const2", 3.0)

    p.DefineStrConst("str1", "1.11")
    p.DefineStrConst("str2", "2.22")

    var_a = Variable(1.0, "a")
    var_b = Variable(2.0, "b")
    var_c = Variable(3.0, "c")
    var_d = Variable(-2.0, "d")

    p.DefineVar("a", var_a)
    p.DefineVar("aa", var_b)
    p.DefineVar("b", var_b)
    p.DefineVar("c", var_c)
    p.DefineVar("d", var_d)

    p.AddValIdent(is_hex_val)

    p.DefineFun("ping", ping)
    p.DefineFun("f0", f0)
    p.DefineFun("f1of1", f1of1)
    p.DefineFun("f1of2", f1of2)
    p.DefineFun("f2of2", f2of2)
    p.DefineFun("f1of3", f1of3)
    p.DefineFun("f2of3", f2of3)
    p.DefineFun("f3of3", f3of3)
    p.DefineFun("f1of4", f1of4)
    p.DefineFun("f2of4", f2of4)
    p.DefineFun("f3of4", f3of4)
    p.DefineFun("f4of4", f4of4)
    p.DefineFun("f1of5", f1of5)
    p.DefineFun("f2of5", f2of5)
    p.DefineFun("f3of5", f3of5)
    p.DefineFun("f4of5", f4of5)
    p.DefineFun("f5of5", f5of5)

    p.DefineOprt("add", add_fn, 0)
    p.DefineOprt("++", add_fn, 0)
    p.DefineOprt("&", land_fn, EOprtPrecedence.prLAND)

    p.DefineFun("min", MathImpl.Min)
    p.DefineFun("max", MathImpl.Max)
    p.DefineFun("sum", MathImpl.Sum)
    p.DefineFun("valueof", value_of)
    p.DefineFun("atof", str_to_float)
    p.DefineFun("strlen", str_len)
    p.DefineFun("strfun1", str_fun1)
    p.DefineFun("strfun2", str_fun2)
    p.DefineFun("strfun3", str_fun3)
    p.DefineFun("strfun4", str_fun4)
    p.DefineFun("strfun5", str_fun5)
    p.DefineFun("strfun6", str_fun6)
    p.DefineFun("multstrfun", mult_str_fun)
    p.DefineFun("lastArg", last_arg)
    p.DefineFun("firstArg", first_arg)
    p.DefineFun("order", first_arg)

    p.DefineFunUserData("funud0_8", fun_ud0, 8)
    p.DefineFunUserData("funud1_16", fun_ud1, 16)
    p.DefineFunUserData("funud2_24", fun_ud2, 24)
    p.DefineFunUserData("funud10_32", fun_ud10, 32)
    p.DefineFunUserData("funud0_9", fun_ud0, 9)
    p.DefineFunUserData("funud1_17", fun_ud1, 17)
    p.DefineFunUserData("funud2_25", fun_ud2, 25)
    p.DefineFunUserData("funud10_33", fun_ud10, 33)
    p.DefineFunUserData("strfunud3_10", str_fun_ud3, 10)
    p.DefineFunUserData("sumud_100", sum_ud, 100)

    p.DefineInfixOprt("$", sign_fn, int(EOprtPrecedence.prPOW) + 1)
    p.DefineInfixOprt("~", plus2)
    p.DefineInfixOprt("~~", plus2)
    p.DefinePostfixOprt("{m}", milli)
    p.DefinePostfixOprt("{M}", mega)
    p.DefinePostfixOprt("m", milli)
    p.DefinePostfixOprt("n", nano)
    p.DefinePostfixOprt("meg", mega)
    p.DefinePostfixOprt("#", times3)
    p.DefinePostfixOprt("'", sqr)

    return p, {"a": var_a, "b": var_b, "c": var_c, "d": var_d}


def check_eqn(expr: str, expected: float, should_pass: bool = True) -> None:
    p1, _ = create_configured_parser()
    try:
        p1.SetExpr(expr)
        val0 = p1.Eval()
        val1 = p1.Eval()
        assert val0 == val1 or (math.isnan(val0) and math.isnan(val1))

        p4 = copy.copy(p1)
        val2 = p4.Eval()

        p5 = Parser(p4)
        p5.EnableOptimizer(False)
        val3 = p5.Eval()

        p4.SetExpr(expr)
        multi_res = p4.EvalMulti()
        val4 = multi_res[-1]
        multi_res2 = p4.EvalMulti()
        val5 = multi_res2[-1]

        vals = [val0, val1, val2, val3, val4, val5]
        close_enough = True
        for v in vals:
            tol = abs(v * 1e-5) if v != 0 else 1e-10
            close_enough = close_enough and (abs(expected - v) <= max(tol, 1e-7))
            close_enough = close_enough and math.isfinite(v)

        if should_pass:
            assert close_enough, f"Expr {expr!r}: expected {expected}, got {vals}"
        else:
            assert not close_enough, f"Expr {expr!r}: expected NOT {expected}, got {vals}"
    except ParserError as exc:
        if should_pass:
            pytest.fail(f"Expr {expr!r} raised unexpected ParserError: {exc.GetMsg()}")


def check_throw(expr: str, expected_code: EErrorCodes) -> None:
    p = Parser()
    p.DefineVar("a", 1.0)
    p.DefineVar("b", 1.0)
    p.DefineVar("c", 1.0)
    p.DefinePostfixOprt("{m}", milli)
    p.DefinePostfixOprt("m", milli)
    p.DefineFun("ping", ping)
    p.DefineFun("valueof", value_of)
    p.DefineFun("strfun1", str_fun1)
    p.DefineFun("strfun2", str_fun2)
    p.DefineFun("strfun3", str_fun3)
    p.DefineFun("strfun4", str_fun4)
    p.DefineFun("strfun5", str_fun5)
    p.DefineFun("strfun6", str_fun6)
    p.DefineFun("strlen", str_len)
    p.DefineFun("multstrfun", mult_str_fun)

    with pytest.raises(ParserError) as exc_info:
        p.SetExpr(expr)
        p.Eval()
    assert exc_info.value.GetCode() == expected_code, (
        f"Expr {expr!r}: expected {expected_code.name}, got {exc_info.value.GetCode().name} ({exc_info.value.GetMsg()})"
    )


def check_eqn_int(expr: str, expected: float, should_pass: bool = True) -> None:
    p = ParserInt()
    p.DefineConst("const1", 1.0)
    p.DefineConst("const2", 2.0)
    p.DefineVar("a", 1.0)
    p.DefineVar("b", 2.0)
    p.DefineVar("c", 3.0)
    try:
        p.SetExpr(expr)
        val0 = p.Eval()
        val1 = p.Eval()
        assert val0 == val1
        if should_pass:
            assert val0 == expected, f"Int Expr {expr!r}: expected {expected}, got {val0}"
        else:
            assert val0 != expected
    except ParserError as exc:
        if should_pass:
            pytest.fail(f"Int Expr {expr!r} raised unexpected ParserError: {exc.GetMsg()}")


def check_throw_int(expr: str, expected_code: EErrorCodes) -> None:
    p = ParserInt()
    with pytest.raises(ParserError) as exc_info:
        p.SetExpr(expr)
        p.Eval()
    assert exc_info.value.GetCode() == expected_code


# ===========================================================================
# Test Suites mirroring muParserTest.cpp
# ===========================================================================


class TestMuParserNames:
    def test_const_names(self):
        p = Parser()
        long_name = "a" * 150
        for bad_name in [long_name, "0a", "9a", "+a", "-a", "a-", "a*", "a?"]:
            with pytest.raises(ParserError):
                p.DefineConst(bad_name, 1.0)

        for good_name in ["a", "a_min", "a_min0", "a_min9"]:
            p.DefineConst(good_name, 1.0)

    def test_var_names(self):
        p = Parser()
        p.ClearConst()
        long_name = "a" * 150
        for bad_name in [
            long_name,
            "123abc",
            "9a",
            "0a",
            "+a",
            "-a",
            "?a",
            "!a",
            "a+",
            "a-",
            "a*",
            "a?",
        ]:
            with pytest.raises(ParserError):
                p.DefineVar(bad_name, 1.0)

        for good_name in ["a", "a_min", "a_min0", "a_min9"]:
            p.DefineVar(good_name, 1.0)

        with pytest.raises(ParserError):
            p.DefineVar("a_min9", None)

    def test_postfix_oprt_names(self):
        p = Parser()
        long_name = "a" * 150
        for bad in [long_name, "(k", "9+"]:
            with pytest.raises(ParserError):
                p.DefinePostfixOprt(bad, f1of1)
        with pytest.raises(ParserError):
            p.DefinePostfixOprt("+", None)

        for good in [
            "-a",
            "?a",
            "_",
            "#",
            "&&",
            "||",
            "&",
            "|",
            "++",
            "--",
            "?>",
            "?<",
            "**",
            "xor",
            "and",
            "or",
            "not",
            "!",
        ]:
            p.DefinePostfixOprt(good, f1of1)

    def test_bin_oprt_names(self):
        p = Parser()
        long_name = "a" * 150
        with pytest.raises(ParserError):
            p.DefineOprt(long_name, f1of2)

        for builtin_op in ["+", "-", "*", "/", "^", "&&", "||"]:
            with pytest.raises(ParserError):
                p.DefineOprt(builtin_op, f1of2)

        p.EnableBuiltInOprt(False)
        for builtin_op in ["+", "-", "*", "/", "^", "&&", "||"]:
            p.DefineOprt(builtin_op, f1of2)


class TestMuParserSyntax:
    def test_syntax_errors(self):
        check_throw("1,", EErrorCodes.ecUNEXPECTED_EOF)
        check_throw("a,", EErrorCodes.ecUNEXPECTED_EOF)
        check_throw("sin(8),", EErrorCodes.ecUNEXPECTED_EOF)
        check_throw("1++sin(2)", EErrorCodes.ecUNARY_PLUS_IN_FRONT_OF_FUNCTION)
        check_throw("1-+sin(2)", EErrorCodes.ecUNARY_PLUS_IN_FRONT_OF_FUNCTION)
        check_throw('1++strlen("333")', EErrorCodes.ecUNARY_PLUS_IN_FRONT_OF_FUNCTION)
        check_throw("1+++sin(2)", EErrorCodes.ecUNEXPECTED_OPERATOR)
        check_throw("1+-+sin(2)", EErrorCodes.ecUNEXPECTED_OPERATOR)
        check_throw("(sin(8)),", EErrorCodes.ecUNEXPECTED_EOF)
        check_throw("a{m},", EErrorCodes.ecUNEXPECTED_EOF)

    def test_syntax_expressions(self):
        check_eqn("(1+ 2*a)", 3.0, True)
        check_eqn("sqrt((4))", 2.0, True)
        check_eqn("sqrt((2)+2)", 2.0, True)
        check_eqn("sqrt(2+(2))", 2.0, True)
        check_eqn("sqrt(a+(3))", 2.0, True)
        check_eqn("sqrt((3)+a)", 2.0, True)
        check_eqn("order(1,2)", 1.0, True)
        check_eqn("(2+", 0.0, False)
        check_eqn("2++4", 0.0, False)
        check_eqn("2+-4", 0.0, False)
        check_eqn("(2+)", 0.0, False)
        check_eqn("--2", 0.0, False)
        check_eqn("ksdfj", 0.0, False)
        check_eqn("()", 0.0, False)
        check_eqn("5+()", 0.0, False)
        check_eqn("sin(cos)", 0.0, False)
        check_eqn("5t6", 0.0, False)
        check_eqn("5 t 6", 0.0, False)
        check_eqn("8*", 0.0, False)
        check_eqn(",3", 0.0, False)
        check_eqn("3,5", 0.0, False)
        check_eqn("sin(8,8)", 0.0, False)
        check_eqn("(7,8)", 0.0, False)
        check_eqn("sin)", 0.0, False)
        check_eqn("a)", 0.0, False)
        check_eqn("pi)", 0.0, False)
        check_eqn("sin(())", 0.0, False)
        check_eqn("sin()", 0.0, False)


class TestMuParserVarConst:
    def test_var_change(self):
        p = Parser()
        v = Variable(0.0, "a")
        p.DefineVar("a", v)
        p.SetExpr("a")
        v.set(1.0)
        assert p.Eval() == pytest.approx(1.0)
        v.set(2.0)
        assert p.Eval() == pytest.approx(2.0)

        p.SetExpr("2*a")
        v.set(2.0)
        assert p.Eval() == pytest.approx(4.0)
        v.set(3.0)
        assert p.Eval() == pytest.approx(6.0)

    def test_const_and_var_distinction(self):
        check_eqn("const", 1.0, True)
        check_eqn("const1", 2.0, True)
        check_eqn("const2", 3.0, True)
        check_eqn("2*const", 2.0, True)
        check_eqn("2*const1", 4.0, True)
        check_eqn("2*const2", 6.0, True)
        check_eqn("2*const+1", 3.0, True)
        check_eqn("2*const1+1", 5.0, True)
        check_eqn("2*const2+1", 7.0, True)
        check_eqn("const", 0.0, False)
        check_eqn("const1", 0.0, False)
        check_eqn("const2", 0.0, False)

        check_eqn("a", 1.0, True)
        check_eqn("aa", 2.0, True)
        check_eqn("2*a", 2.0, True)
        check_eqn("2*aa", 4.0, True)
        check_eqn("2*a-1", 1.0, True)
        check_eqn("2*aa-1", 3.0, True)

        check_eqn("0xff", 255.0, True)
        check_eqn("0x97 + 0xff", 406.0, True)

    def test_get_used_var(self):
        p = Parser()
        refs = [Variable(float(i + 1)) for i in range(5)]
        for name, ref in zip(["a", "b", "c", "d", "e"], refs):
            p.DefineVar(name, ref)

        p.SetExpr("a+b+c+d")
        used = p.GetUsedVar()
        assert len(used) == 4
        assert len(p.GetVar()) == 5
        for name, ref in zip(["a", "b", "c", "d"], refs[:4]):
            assert used[name] is ref

        p.SetExpr("undef1+undef2+undef3")
        used_undef = p.GetUsedVar()
        assert len(used_undef) == 3
        assert len(p.GetVar()) == 5
        for k, v in used_undef.items():
            assert v is None

        p.SetExpr("a+b")
        used2 = p.GetUsedVar()
        assert len(used2) == 2


class TestMuParserMultiArg:
    def test_multi_arg(self):
        check_throw(
            "6, +, +, +, +, +, +, +, +, +, +, +, +, +, +, 1, +, +, +, +, +, +, +, +, +, +, +, +, +, +, +, +, +, +, +, +, +, +, +, +, +, +, +, +, +, +, +, +",
            EErrorCodes.ecUNEXPECTED_ARG_SEP,
        )
        check_throw('sin(0?4:(""))', EErrorCodes.ecUNEXPECTED_STR)
        check_throw('avg(0?4:(""),1)', EErrorCodes.ecUNEXPECTED_STR)

        check_eqn("1,2,3", 3.0, True)
        check_eqn("a,b,c", 3.0, True)
        check_eqn("a=10,b=20,c=a*b", 200.0, True)
        check_eqn("1,\n2,\n3", 3.0, True)
        check_eqn("a,\nb,\nc", 3.0, True)
        check_eqn("a=10,\nb=20,\nc=a*b", 200.0, True)
        check_eqn("1,\r\n2,\r\n3", 3.0, True)
        check_eqn("a,\r\nb,\r\nc", 3.0, True)
        check_eqn("a=10,\r\nb=20,\r\nc=a*b", 200.0, True)

        check_eqn("f1of1(1)", 1.0, True)
        check_eqn("f1of2(1, 2)", 1.0, True)
        check_eqn("f2of2(1, 2)", 2.0, True)
        check_eqn("f1of3(1, 2, 3)", 1.0, True)
        check_eqn("f2of3(1, 2, 3)", 2.0, True)
        check_eqn("f3of3(1, 2, 3)", 3.0, True)
        check_eqn("f1of4(1, 2, 3, 4)", 1.0, True)
        check_eqn("f2of4(1, 2, 3, 4)", 2.0, True)
        check_eqn("f3of4(1, 2, 3, 4)", 3.0, True)
        check_eqn("f4of4(1, 2, 3, 4)", 4.0, True)
        check_eqn("f1of5(1, 2, 3, 4, 5)", 1.0, True)
        check_eqn("f2of5(1, 2, 3, 4, 5)", 2.0, True)
        check_eqn("f3of5(1, 2, 3, 4, 5)", 3.0, True)
        check_eqn("f4of5(1, 2, 3, 4, 5)", 4.0, True)
        check_eqn("f5of5(1, 2, 3, 4, 5)", 5.0, True)

        check_eqn("1+ping()", 11.0, True)
        check_eqn("ping()+1", 11.0, True)
        check_eqn("2*ping()", 20.0, True)
        check_eqn("ping()*2", 20.0, True)
        check_eqn("ping(1,2)", 0.0, False)
        check_eqn("1+ping(1,2)", 0.0, False)
        check_eqn("f1of1(1,2)", 0.0, False)
        check_eqn("f1of1()", 0.0, False)
        check_eqn("f1of2(1, 2, 3)", 0.0, False)
        check_eqn("f1of2(1)", 0.0, False)
        check_eqn("f1of3(1, 2, 3, 4)", 0.0, False)
        check_eqn("f1of3(1)", 0.0, False)
        check_eqn("f1of4(1, 2, 3, 4, 5)", 0.0, False)
        check_eqn("f1of4(1)", 0.0, False)
        check_eqn("(1,2,3)", 0.0, False)
        check_eqn("1,2,3", 0.0, False)
        check_eqn("(1*a,2,3)", 0.0, False)
        check_eqn("1,2*a,3", 0.0, False)

        check_eqn("min(a, 1)", 1.0, True)
        check_eqn("min(3*2, 1)", 1.0, True)
        check_eqn("min(3*2, 1)", 6.0, False)
        check_eqn("firstArg(2,3,4)", 2.0, True)
        check_eqn("lastArg(2,3,4)", 4.0, True)
        check_eqn("min(3*a+1, 1)", 1.0, True)
        check_eqn("max(3*a+1, 1)", 4.0, True)
        check_eqn("max(3*a+1, 1)*2", 8.0, True)
        check_eqn("2*max(3*a+1, 1)+2", 10.0, True)

        check_eqn("sum(a)", 1.0, True)
        check_eqn("sum(1,2,3)", 6.0, True)
        check_eqn("sum(a,b,c)", 6.0, True)
        check_eqn("sum(1,-max(1,2),3)*2", 4.0, True)
        check_eqn("2*sum(1,2,3)", 12.0, True)
        check_eqn("2*sum(1,2,3)+2", 14.0, True)
        check_eqn("2*sum(-1,2,3)+2", 10.0, True)
        check_eqn("2*sum(-1,2,-(-a))+2", 6.0, True)
        check_eqn("2*sum(-1,10,-a)+2", 18.0, True)
        check_eqn("2*sum(1,2,3)*2", 24.0, True)
        check_eqn("sum(1*3, 4, a+2)", 10.0, True)
        check_eqn("sum(1*3, 2*sum(1,2,2), a+2)", 16.0, True)
        check_eqn("sum(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2)", 24.0, True)

        check_eqn("sum()", 0.0, False)
        check_eqn("sum(,)", 0.0, False)
        check_eqn("sum(1,2,)", 0.0, False)
        check_eqn("sum(,1,2)", 0.0, False)

        check_eqn("funud0_8()", 8.0, True)
        check_eqn("funud1_16(10)", 26.0, True)
        check_eqn("funud2_24(10, 100)", 134.0, True)
        check_eqn("funud10_32(1,2,3,4,5,6,7,8,9,10)", 87.0, True)
        check_eqn("funud0_9()", 9.0, True)
        check_eqn("funud1_17(10)", 27.0, True)
        check_eqn("funud2_25(10, 100)", 135.0, True)
        check_eqn("funud10_33(1,2,3,4,5,6,7,8,9,10)", 88.0, True)
        check_eqn("sumud_100(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2)", 124.0, True)
        check_eqn("sumud_100()", 0.0, False)


class TestMuParserInfixAndPostfix:
    def test_infix_oprt(self):
        check_eqn("+1", 1.0, True)
        check_eqn("-(+1)", -1.0, True)
        check_eqn("-(+1)*2", -2.0, True)
        check_eqn("-(+2)*sqrt(4)", -4.0, True)
        check_eqn("3-+a", 2.0, True)
        check_eqn("+1*3", 3.0, True)

        check_eqn("-1", -1.0, True)
        check_eqn("-(-1)", 1.0, True)
        check_eqn("-(-1)*2", 2.0, True)
        check_eqn("-(-2)*sqrt(4)", 4.0, True)
        check_eqn("-_pi", -MathImpl.CONST_PI, True)
        check_eqn("-a", -1.0, True)
        check_eqn("-(a)", -1.0, True)
        check_eqn("-(-a)", 1.0, True)
        check_eqn("-(-a)*2", 2.0, True)
        check_eqn("-(8)", -8.0, True)
        check_eqn("-8", -8.0, True)
        check_eqn("-(2+1)", -3.0, True)
        check_eqn("-(f1of1(1+2*3)+1*2)", -9.0, True)
        check_eqn("-(-f1of1(1+2*3)+1*2)", 5.0, True)
        check_eqn("-sin(8)", -0.989358, True)
        check_eqn("3-(-a)", 4.0, True)
        check_eqn("3--a", 4.0, True)
        check_eqn("-1*3", -3.0, True)

        check_eqn("~2#", 8.0, True)
        check_eqn("~f1of1(2)#", 8.0, True)
        check_eqn("~(b)#", 8.0, True)
        check_eqn("(~b)#", 12.0, True)
        check_eqn("~(2#)", 8.0, True)
        check_eqn("~(f1of1(2)#)", 8.0, True)

        check_eqn("-2^2", -4.0, True)
        check_eqn("-(a+b)^2", -9.0, True)
        check_eqn("(-3)^2", 9.0, True)
        check_eqn("-(-2^2)", 4.0, True)
        check_eqn("3+-3^2", -6.0, True)

        check_eqn("-2'", -4.0, True)
        check_eqn("-(1+1)'", -4.0, True)
        check_eqn("2+-(1+1)'", -2.0, True)
        check_eqn("2+-2'", -2.0, True)

        check_eqn("$2^2", 4.0, True)
        check_eqn("$(a+b)^2", 9.0, True)
        check_eqn("($3)^2", 9.0, True)
        check_eqn("$($2^2)", -4.0, True)
        check_eqn("3+$3^2", 12.0, True)

        check_eqn("~ 123", 125.0, True)
        check_eqn("~~ 123", 125.0, True)

    def test_postfix_oprt(self):
        check_eqn("3{m}+5", 5.003, True)
        check_eqn("1000{m}", 1.0, True)
        check_eqn("1000 {m}", 1.0, True)
        check_eqn("(a){m}", 1e-3, True)
        check_eqn("a{m}", 1e-3, True)
        check_eqn("a {m}", 1e-3, True)
        check_eqn("-(a){m}", -1e-3, True)
        check_eqn("-2{m}", -2e-3, True)
        check_eqn("-2 {m}", -2e-3, True)
        check_eqn("f1of1(1000){m}", 1.0, True)
        check_eqn("-f1of1(1000){m}", -1.0, True)
        check_eqn("-f1of1(-1000){m}", 1.0, True)
        check_eqn("f4of4(0,0,0,1000){m}", 1.0, True)
        check_eqn("2+(a*1000){m}", 3.0, True)
        check_eqn("1 n", 1e-9, True)
        check_eqn("1n", 1e-9, True)
        check_eqn("2*3000meg+2", 2 * 3e9 + 2, True)

        check_eqn("1000{m}", 0.1, False)
        check_eqn("(a){m}", 2.0, False)

        check_throw("0x", EErrorCodes.ecUNASSIGNABLE_TOKEN)
        check_throw("3+", EErrorCodes.ecUNEXPECTED_EOF)
        check_throw("4 + {m}", EErrorCodes.ecUNASSIGNABLE_TOKEN)
        check_throw("{m}4", EErrorCodes.ecUNASSIGNABLE_TOKEN)
        check_throw("sin({m})", EErrorCodes.ecUNASSIGNABLE_TOKEN)
        check_throw("{m} {m}", EErrorCodes.ecUNASSIGNABLE_TOKEN)
        check_throw("{m}(8)", EErrorCodes.ecUNASSIGNABLE_TOKEN)
        check_throw("4,{m}", EErrorCodes.ecUNASSIGNABLE_TOKEN)
        check_throw("-{m}", EErrorCodes.ecUNASSIGNABLE_TOKEN)
        check_throw("2(-{m})", EErrorCodes.ecUNEXPECTED_PARENS)
        check_throw("2({m})", EErrorCodes.ecUNEXPECTED_PARENS)
        check_throw("multi*1.0", EErrorCodes.ecUNASSIGNABLE_TOKEN)


class TestMuParserExpressionsAndOperators:
    def test_expressions(self):
        b = 2.0
        check_eqn("f0()", 42.0, True)
        check_eqn("b^2", 4.0, True)
        check_eqn("b^1", 2.0, True)
        check_eqn("b^0", 1.0, True)
        check_eqn("b^-1", 0.5, True)

        check_eqn("2*b*5", 20.0, True)
        check_eqn("2*b*5 + 4*b", 28.0, True)
        check_eqn("2*a/3", 2.0 / 3.0, True)

        check_eqn("3+b", b + 3, True)
        check_eqn("b+3", b + 3, True)
        check_eqn("b*3+2", b * 3 + 2, True)
        check_eqn("3*b+2", b * 3 + 2, True)
        check_eqn("2+b*3", b * 3 + 2, True)
        check_eqn("2+3*b", b * 3 + 2, True)
        check_eqn("b+3*b", b + 3 * b, True)
        check_eqn("3*b+b", b + 3 * b, True)
        check_eqn("2+b*3+b", 2 + b * 3 + b, True)
        check_eqn("b+2+b*3", b + 2 + b * 3, True)
        check_eqn("(2*b+1)*4", (2 * b + 1) * 4, True)
        check_eqn("4*(2*b+1)", (2 * b + 1) * 4, True)

        check_eqn("1+2-3*4/5^6", 2.99923, True)
        check_eqn("1^2/3*4-5+6", 2.33333333, True)
        check_eqn("1+2*3", 7.0, True)
        check_eqn("(1+2)*3", 9.0, True)
        check_eqn("(1+2)*(-3)", -9.0, True)
        check_eqn("2/4", 0.5, True)

        check_eqn("exp(ln(7))", 7.0, True)
        check_eqn("e^ln(7)", 7.0, True)
        check_eqn("e^(ln(7))", 7.0, True)
        check_eqn("(e^(ln(7)))", 7.0, True)
        check_eqn("1-(e^(ln(7)))", -6.0, True)
        check_eqn("2*(e^(ln(7)))", 14.0, True)
        check_eqn("10^log(5)", math.pow(10.0, math.log(5.0)), True)
        check_eqn("10^log10(5)", 5.0, True)
        check_eqn("2^log2(4)", 4.0, True)
        check_eqn("-(sin(0)+1)", -1.0, True)
        check_eqn("-(2^1.1)", -2.14354692, True)

        check_eqn("(cos(2.41)/b)", -0.372056, True)
        check_eqn("(1*(2*(3*(4*(5*(6*(a+b)))))))", 2160.0, True)
        check_eqn("(1*(2*(3*(4*(5*(6*(7*(a+b))))))))", 15120.0, True)
        check_eqn(
            "(a/((((b+(((e*(((((pi*((((3.45*((pi+a)+pi))+b)+b)*a))+0.68)+e)+a)/a))+a)+b))+b)*a)-pi))",
            0.00377999,
            True,
        )
        check_eqn(
            "(((-9))-e/(((((((pi-(((-7)+(-3)/4/e))))/(((-5))-2)-((pi+(-0))*(sqrt((e+e))*(-8))*(((-pi)+(-pi)-(-9)*(6*5))"
            "/(-e)-e))/2)/((((sqrt(2/(-e)+6)-(4-2))+((5/(-2))/(1*(-pi)+3))/8)*pi*((pi/((-2)/(-6)*1*(-1))*(-6)+(-e)))))/"
            "((e+(-2)+(-e)*((((-3)*9+(-e)))+(-9)))))))-((((e-7+(((5/pi-(3/1+pi)))))/e)/(-5))/(sqrt((((((1+(-7))))+((((-"
            "e)*(-e)))-8))*(-5)/((-e)))*(-6)-((((((-2)-(-9)-(-e)-1)/3))))/(sqrt((8+(e-((-6))+(9*(-9))))*(((3+2-8))*(7+6"
            "+(-5))+((0/(-e)*(-pi))+7)))+(((((-e)/e/e)+((-6)*5)*e+(3+(-5)/pi))))+pi))/sqrt((((9))+((((pi))-8+2))+pi))/e"
            "*4)*((-5)/(((-pi))*(sqrt(e)))))-(((((((-e)*(e)-pi))/4+(pi)*(-9)))))))+(-pi)",
            -12.23016549,
            True,
        )
        check_eqn(
            "(atan(sin((((((((((((((((pi/cos((a/((((0.53-b)-pi)*e)/b))))+2.51)+a)-0.54)/0.98)+b)*b)+e)/a)+b)+a)+b)+pi)/e"
            ")+a)))*2.77)",
            -2.16995656,
            True,
        )
        check_eqn("1+2-3*4/5^6*(2*(1-5+(3*7^9)*(4+6*7-3)))+12", -7995810.09926, True)

    def test_bin_oprt(self):
        check_eqn("a++b", 3.0, True)
        check_eqn("a ++ b", 3.0, True)
        check_eqn("1++2", 3.0, True)
        check_eqn("1 ++ 2", 3.0, True)
        check_eqn("a add b", 3.0, True)
        check_eqn("1 add 2", 3.0, True)
        check_eqn("a<b", 1.0, True)
        check_eqn("b>a", 1.0, True)
        check_eqn("a>a", 0.0, True)
        check_eqn("a<a", 0.0, True)
        check_eqn("a<=a", 1.0, True)
        check_eqn("a<=b", 1.0, True)
        check_eqn("b<=a", 0.0, False if False else True)
        check_eqn("a>=a", 1.0, True)
        check_eqn("b>=a", 1.0, True)
        check_eqn("a>=b", 0.0, True)

        check_eqn("1 && 1", 1.0, True)
        check_eqn("1 && 0", 0.0, True)
        check_eqn("(a<b) && (b>a)", 1.0, True)
        check_eqn("(a<b) && (a>b)", 0.0, True)
        check_eqn("12 & 255", 12.0, True)
        check_eqn("12 & 0", 0.0, True)
        check_eqn("12&255", 12.0, True)
        check_eqn("12&0", 0.0, True)

        check_eqn("0 && 0 || 1", 1.0, True)
        check_eqn("0 && 1 || 0", 0.0, True)
        check_eqn("1 && 0 || 0", 0.0, True)
        check_eqn("1 && 1 || 0", 1.0, True)
        check_eqn("1 && 0 + 1", 1.0, True)
        check_eqn("1 && 1 - 1", 0.0, True)

        check_eqn("a = b", 2.0, True)
        check_eqn("a = sin(b)", 0.909297, True)
        check_eqn("a = 1+sin(b)", 1.909297, True)
        check_eqn("(a=b)*2", 4.0, True)
        check_eqn("2*(a=b)", 4.0, True)
        check_eqn("2*(a=b+1)", 6.0, True)
        check_eqn("(a=b+1)*2", 6.0, True)
        check_eqn("a=c, a*10", 30.0, True)

        check_eqn("2^2^3", 256.0, True)
        check_eqn("1/2/3", 1.0 / 6.0, True)

        check_eqn("(1/0)>1", 1.0, True)
        check_eqn("(-1/0)<-1", 1.0, True)
        check_eqn("(0/0)==(0/0)", 0.0, True)
        check_eqn("min(4/0,2)", 2.0, True)

        check_eqn("3+4*2/(1-5)^2^3", 3.0001220703125, True)

    def test_parser_int(self):
        check_eqn_int("1 | 2", 3, True)
        check_eqn_int("1 || 2", 1, True)
        check_eqn_int("123 & 456", 72, True)
        check_eqn_int("(123 & 456) % 10", 2, True)
        check_eqn_int("1 && 0", 0, True)
        check_eqn_int("123 && 456", 1, True)
        check_eqn_int("1 << 3", 8, True)
        check_eqn_int("8 >> 3", 1, True)
        check_eqn_int("#11", 3, True)
        check_eqn_int("#101 + 1", 6, True)
        check_eqn_int("#1111111111111111111111111111111", 2147483647, True)
        check_eqn_int("9 / 4", 2, True)
        check_eqn_int("9 % 4", 1, True)
        check_eqn_int("if(5%2,1,0)", 1, True)
        check_eqn_int("if(4%2,1,0)", 0, True)
        check_eqn_int("-10+1", -9, True)
        check_eqn_int("1+2*3", 7, True)
        check_eqn_int("const1 != const2", 1, True)
        check_eqn_int("const1 != const2", 0, False)
        check_eqn_int("const1 == const2", 0, True)
        check_eqn_int("const1 == 1", 1, True)
        check_eqn_int("10*(const1 == 1)", 10, True)
        check_eqn_int("2*(const1 | const2)", 6, True)
        check_eqn_int("2*(const1 | const2)", 7, False)
        check_eqn_int("const1 < const2", 1, True)
        check_eqn_int("const2 > const1", 1, True)
        check_eqn_int("const1 <= 1", 1, True)
        check_eqn_int("const2 >= 2", 1, True)
        check_eqn_int("2*(const1 + const2)", 6, True)
        check_eqn_int("2*(const1 - const2)", -2, True)
        check_eqn_int("a != b", 1, True)
        check_eqn_int("a == b", 0, True)
        check_eqn_int("a == 1", 1, True)
        check_eqn_int("10*(a == 1)", 10, True)
        check_eqn_int("2*(a | b)", 6, True)
        check_eqn_int("a < b", 1, True)
        check_eqn_int("b > a", 1, True)
        check_eqn_int("a <= 1", 1, True)
        check_eqn_int("b >= 2", 1, True)
        check_eqn_int("2*(a + b)", 6, True)
        check_eqn_int("2*(a - b)", -2, True)
        check_eqn_int("a + (a << b)", 5, True)
        check_eqn_int("-2^2", -4, True)
        check_eqn_int("3--a", 4, True)
        check_eqn_int("3+-3^2", -6, True)

        check_eqn_int("0xff", 255, True)
        check_eqn_int("10+0xff", 265, True)
        check_eqn_int("0xff+10", 265, True)
        check_eqn_int("10*0xff", 2550, True)
        check_eqn_int("0xff*10", 2550, True)
        check_eqn_int("10+0xff+1", 266, True)
        check_eqn_int("1+0xff+10", 266, True)

        check_eqn_int("a + b * c", 7, True)
        check_eqn_int("a * b + c", 5, True)
        check_eqn_int("a<b && b>10", 0, True)
        check_eqn_int("a<b && b<10", 1, True)
        check_eqn_int("a + b << c", 17, True)
        check_eqn_int("a << b + c", 7, True)
        check_eqn_int("c * b < a", 0, True)
        check_eqn_int("c * b == 6 * a", 1, True)
        check_eqn_int("2^2^3", 256, True)

        check_throw_int("8 >> 100", EErrorCodes.ecDOMAIN_ERROR)
        check_throw_int("8 << -1", EErrorCodes.ecDOMAIN_ERROR)


class TestMuParserIfThenElse:
    def test_if_then_else_errors(self):
        check_throw('0^3^avg(0>3?4:(""),0^3?4:(""))', EErrorCodes.ecUNEXPECTED_STR)
        check_throw('avg(0?(""):4,1)', EErrorCodes.ecUNEXPECTED_STR)
        check_throw('avg(0>3?4:(""),0^3?4:(""))', EErrorCodes.ecUNEXPECTED_STR)
        check_throw('0?4:("")', EErrorCodes.ecUNEXPECTED_STR)
        check_throw('(0)?4:("")', EErrorCodes.ecUNEXPECTED_STR)
        check_throw('(0>3)?4:("")', EErrorCodes.ecUNEXPECTED_STR)
        check_throw('0>3?4:("")', EErrorCodes.ecUNEXPECTED_STR)
        check_throw("3!=min(0?2>2,2>5,1:6)", EErrorCodes.ecUNEXPECTED_ARG_SEP)

        check_throw(":3", EErrorCodes.ecUNEXPECTED_CONDITIONAL)
        check_throw("? 1 : 2", EErrorCodes.ecUNEXPECTED_CONDITIONAL)
        check_throw("(a<b) ? (b<c) ? 1 : 2", EErrorCodes.ecMISSING_ELSE_CLAUSE)
        check_throw("(a<b) ? 1", EErrorCodes.ecMISSING_ELSE_CLAUSE)
        check_throw("(a<b) ? a", EErrorCodes.ecMISSING_ELSE_CLAUSE)
        check_throw("(a<b) ? a+b", EErrorCodes.ecMISSING_ELSE_CLAUSE)
        check_throw("a : b", EErrorCodes.ecMISPLACED_COLON)
        check_throw("1 : 2", EErrorCodes.ecMISPLACED_COLON)
        check_throw("(1) ? 1 : 2 : 3", EErrorCodes.ecMISPLACED_COLON)
        check_throw("(true) ? 1 : 2 : 3", EErrorCodes.ecUNASSIGNABLE_TOKEN)
        check_throw("1?2:0?(7:1)", EErrorCodes.ecMISPLACED_COLON)

        check_throw("sum(0?1,0,0:3)", EErrorCodes.ecUNEXPECTED_ARG_SEP)
        check_throw("sum(0?(1,0,0):3)", EErrorCodes.ecUNEXPECTED_ARG)
        check_throw("sum(2>3?2,4,2:4)", EErrorCodes.ecUNEXPECTED_ARG_SEP)
        check_throw("sum(2>3?2,4,sin(2):4)", EErrorCodes.ecUNEXPECTED_ARG_SEP)
        check_throw("sum(2>3?sin(2),4,2:4)", EErrorCodes.ecUNEXPECTED_ARG_SEP)
        check_throw("sum(2>3?sin(a),4,2:4)", EErrorCodes.ecUNEXPECTED_ARG_SEP)

    def test_if_then_else_evaluation(self):
        check_eqn("1 ? 128 : 255", 128.0, True)
        check_eqn("1<2 ? 128 : 255", 128.0, True)
        check_eqn("a<b ? 128 : 255", 128.0, True)
        check_eqn("(a<b) ? 128 : 255", 128.0, True)
        check_eqn("(1) ? 10 : 11", 10.0, True)
        check_eqn("(0) ? 10 : 11", 11.0, True)
        check_eqn("(1) ? a+b : c+d", 3.0, True)
        check_eqn("(0) ? a+b : c+d", 1.0, True)
        check_eqn("(1) ? 0 : 1", 0.0, True)
        check_eqn("(0) ? 0 : 1", 1.0, True)
        check_eqn("(a<b) ? 10 : 11", 10.0, True)
        check_eqn("(a>b) ? 10 : 11", 11.0, True)
        check_eqn("(a<b) ? c : d", 3.0, True)
        check_eqn("(a>b) ? c : d", -2.0, True)

        check_eqn("(a>b) ? 1 : 0", 0.0, True)
        check_eqn("((a>b) ? 1 : 0) ? 1 : 2", 2.0, True)
        check_eqn("((a>b) ? 1 : 0) ? 1 : sum((a>b) ? 1 : 2)", 2.0, True)
        check_eqn("((a>b) ? 0 : 1) ? 1 : sum((a>b) ? 1 : 2)", 1.0, True)

        check_eqn("sum((a>b) ? 1 : 2)", 2.0, True)
        check_eqn("sum((1) ? 1 : 2)", 1.0, True)
        check_eqn("sum((a>b) ? 1 : 2, 100)", 102.0, True)
        check_eqn("sum((1) ? 1 : 2, 100)", 101.0, True)
        check_eqn("sum(3, (a>b) ? 3 : 10)", 13.0, True)
        check_eqn("sum(3, (a<b) ? 3 : 10)", 6.0, True)
        check_eqn("10*sum(3, (a>b) ? 3 : 10)", 130.0, True)
        check_eqn("10*sum(3, (a<b) ? 3 : 10)", 60.0, True)
        check_eqn("sum(3, (a>b) ? 3 : 10)*10", 130.0, True)
        check_eqn("sum(3, (a<b) ? 3 : 10)*10", 60.0, True)
        check_eqn("(a<b) ? sum(3, (a<b) ? 3 : 10)*10 : 99", 60.0, True)
        check_eqn("(a>b) ? sum(3, (a<b) ? 3 : 10)*10 : 99", 99.0, True)
        check_eqn("(a<b) ? sum(3, (a<b) ? 3 : 10,10,20)*10 : 99", 360.0, True)
        check_eqn("(a>b) ? sum(3, (a<b) ? 3 : 10,10,20)*10 : 99", 99.0, True)
        check_eqn("(a>b) ? sum(3, (a<b) ? 3 : 10,10,20)*10 : sum(3, (a<b) ? 3 : 10)*10", 60.0, True)

        check_eqn("(a<b)&&(a<b) ? 128 : 255", 128.0, True)
        check_eqn("(a>b)&&(a<b) ? 128 : 255", 255.0, True)
        check_eqn("(1<2)&&(1<2) ? 128 : 255", 128.0, True)
        check_eqn("(1>2)&&(1<2) ? 128 : 255", 255.0, True)
        check_eqn("((1<2)&&(1<2)) ? 128 : 255", 128.0, True)
        check_eqn("((1>2)&&(1<2)) ? 128 : 255", 255.0, True)
        check_eqn("((a<b)&&(a<b)) ? 128 : 255", 128.0, True)
        check_eqn("((a>b)&&(a<b)) ? 128 : 255", 255.0, True)

        check_eqn("1>0 ? 1>2 ? 128 : 255 : 1>0 ? 32 : 64", 255.0, True)
        check_eqn("1>0 ? 1>2 ? 128 : 255 :(1>0 ? 32 : 64)", 255.0, True)
        check_eqn("1>0 ? 1>0 ? 128 : 255 : 1>2 ? 32 : 64", 128.0, True)
        check_eqn("1>0 ? 1>0 ? 128 : 255 :(1>2 ? 32 : 64)", 128.0, True)
        check_eqn("1>2 ? 1>2 ? 128 : 255 : 1>0 ? 32 : 64", 32.0, True)
        check_eqn("1>2 ? 1>0 ? 128 : 255 : 1>2 ? 32 : 64", 64.0, True)
        check_eqn("1>0 ? 50 :  1>0 ? 128 : 255", 50.0, True)
        check_eqn("1>0 ? 50 : (1>0 ? 128 : 255)", 50.0, True)
        check_eqn("1>0 ? 1>0 ? 128 : 255 : 50", 128.0, True)
        check_eqn("1>2 ? 1>2 ? 128 : 255 : 1>0 ? 32 : 1>2 ? 64 : 16", 32.0, True)
        check_eqn("1>2 ? 1>2 ? 128 : 255 : 1>0 ? 32 :(1>2 ? 64 : 16)", 32.0, True)
        check_eqn("1>0 ? 1>2 ? 128 : 255 :  1>0 ? 32 :1>2 ? 64 : 16", 255.0, True)
        check_eqn("1>0 ? 1>2 ? 128 : 255 : (1>0 ? 32 :1>2 ? 64 : 16)", 255.0, True)
        check_eqn("1 ? 0 ? 128 : 255 : 1 ? 32 : 64", 255.0, True)

        # Assignment in ternary branches (lazy evaluation)
        check_eqn("a= 0 ? 128 : 255, a", 255.0, True)
        check_eqn("a=((a>b)&&(a<b)) ? 128 : 255, a", 255.0, True)
        check_eqn("c=(a<b)&&(a<b) ? 128 : 255, c", 128.0, True)
        check_eqn("0 ? a=a+1 : 666, a", 1.0, True)
        check_eqn("1?a=10:a=20, a", 10.0, True)
        check_eqn("0?a=10:a=20, a", 20.0, True)
        check_eqn("0?a=sum(3,4):10, a", 1.0, True)

        check_eqn("a=1?b=1?3:4:5, a", 3.0, True)
        check_eqn("a=1?b=1?3:4:5, b", 3.0, True)
        check_eqn("a=0?b=1?3:4:5, a", 5.0, True)
        check_eqn("a=0?b=1?3:4:5, b", 2.0, True)

        check_eqn("a=1?5:b=1?3:4, a", 5.0, True)
        check_eqn("a=1?5:b=1?3:4, b", 2.0, True)
        check_eqn("a=0?5:b=1?3:4, a", 3.0, True)
        check_eqn("a=0?5:b=1?3:4, b", 3.0, True)


class TestMuParserStringsAndExceptions:
    def test_str_arg(self):
        check_throw('6 - 6 ? 4 : "", ? 4 : "", ? 4 : ""', EErrorCodes.ecUNEXPECTED_STR)
        check_throw('avg(0?4:(""),1)', EErrorCodes.ecUNEXPECTED_STR)
        check_throw('1 ? 4 : ""', EErrorCodes.ecUNEXPECTED_STR)
        check_throw('1 ? "" : 4', EErrorCodes.ecUNEXPECTED_STR)
        check_throw('1 ? "" : ""', EErrorCodes.ecUNEXPECTED_STR)
        check_throw('0 ? 4 : ""', EErrorCodes.ecUNEXPECTED_STR)
        check_throw('0 ? 4 : ("")', EErrorCodes.ecUNEXPECTED_STR)
        check_throw(
            '"","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","",8',
            EErrorCodes.ecSTR_RESULT,
        )
        check_throw_expr = ['(""), 7', '(("")), 7', '"","",9']
        for e in check_throw_expr:
            check_throw(e, EErrorCodes.ecSTR_RESULT)

        check_eqn('valueof("")', 123.0, True)
        check_eqn('valueof("aaa")+valueof("bbb")  ', 246.0, True)
        check_eqn('2*(valueof("aaa")-23)+valueof("bbb")', 323.0, True)
        check_eqn('a*(atof("10")-b)', 8.0, True)
        check_eqn('a-(atof("10")*b)', -19.0, True)

        check_eqn('strfun1("100")', 100.0, True)
        check_eqn('strfun2("100",1)', 101.0, True)
        check_eqn('strfun3("99",1,2)', 102.0, True)
        check_eqn('strfun4("99",1,2,3)', 105.0, True)
        check_eqn('strfun5("99",1,2,3,4)', 109.0, True)
        check_eqn('strfun6("99",1,2,3,4,5)', 114.0, True)

        check_eqn('strlen("1")+strlen("11")', 3.0, True)
        check_eqn('strlen("1")+strlen("22")++strlen("333")', 6.0, True)
        check_eqn("atof(str1)+atof(str2)", 3.33, True)
        check_eqn('strfunud3_10("99",1,2)', 112.0, True)

        check_eqn('multstrfun("hello",1,2,3)', 6.0, True)
        check_eqn('multstrfun("hello",1)', 1.0, True)
        check_eqn('multstrfun("hello",a,b)+1', 4.0, True)
        check_eqn('multstrfun("hello",sum(1,2),3)', 6.0, True)
        check_eqn('sum(multstrfun("a",1,2),multstrfun("b",3))', 6.0, True)
        check_eqn('multstrfun("a",multstrfun("b",1,2),3)', 6.0, True)

    def test_exceptions(self):
        check_throw("3+", EErrorCodes.ecUNEXPECTED_EOF)
        check_throw("3+)", EErrorCodes.ecUNEXPECTED_PARENS)
        check_throw("()", EErrorCodes.ecUNEXPECTED_PARENS)
        check_throw("3+()", EErrorCodes.ecUNEXPECTED_PARENS)
        check_throw("sin(3,4)", EErrorCodes.ecTOO_MANY_PARAMS)
        check_throw("sin()", EErrorCodes.ecTOO_FEW_PARAMS)
        check_throw("(1+2", EErrorCodes.ecMISSING_PARENS)
        check_throw("sin(3)3", EErrorCodes.ecUNEXPECTED_VAL)
        check_throw("sin(3)xyz", EErrorCodes.ecUNASSIGNABLE_TOKEN)
        check_throw("sin(3)cos(3)", EErrorCodes.ecUNEXPECTED_FUN)
        check_throw("a+b+c=10", EErrorCodes.ecUNEXPECTED_OPERATOR)
        check_throw("a=b=3", EErrorCodes.ecUNEXPECTED_OPERATOR)

        check_throw("3+ping(2)", EErrorCodes.ecTOO_MANY_PARAMS)
        check_throw("3+ping(a+2)", EErrorCodes.ecTOO_MANY_PARAMS)
        check_throw("3+ping(sin(a)+2)", EErrorCodes.ecTOO_MANY_PARAMS)
        check_throw("3+ping(1+sin(a))", EErrorCodes.ecTOO_MANY_PARAMS)

        check_throw("valueof()", EErrorCodes.ecUNEXPECTED_PARENS)
        check_throw('1+valueof("abc"', EErrorCodes.ecMISSING_PARENS)
        check_throw('valueof("abc"', EErrorCodes.ecMISSING_PARENS)
        check_throw('valueof("abc', EErrorCodes.ecUNTERMINATED_STRING)
        check_throw('valueof("abc",3)', EErrorCodes.ecTOO_MANY_PARAMS)
        check_throw("valueof(3)", EErrorCodes.ecSTRING_EXPECTED)
        check_throw('sin("abc")', EErrorCodes.ecVAL_EXPECTED)
        check_throw('"hello world"', EErrorCodes.ecSTR_RESULT)
        check_throw('("hello world")', EErrorCodes.ecSTR_RESULT)
        check_throw('"abcd"+100', EErrorCodes.ecSTR_RESULT)
        check_throw('"a"+"b"', EErrorCodes.ecSTR_RESULT)
        check_throw('strfun1("100",3)', EErrorCodes.ecTOO_MANY_PARAMS)
        check_throw('strfun2("100",3,5)', EErrorCodes.ecTOO_MANY_PARAMS)
        check_throw('strfun3("100",3,5,6)', EErrorCodes.ecTOO_MANY_PARAMS)
        check_throw('strfun2("100")', EErrorCodes.ecTOO_FEW_PARAMS)
        check_throw('strfun3("100",6)', EErrorCodes.ecTOO_FEW_PARAMS)
        check_throw("strfun2(1,1)", EErrorCodes.ecSTRING_EXPECTED)
        check_throw("strfun2(a,1)", EErrorCodes.ecSTRING_EXPECTED)
        check_throw("strfun2(1,1,1)", EErrorCodes.ecTOO_MANY_PARAMS)
        check_throw('multstrfun("hello")', EErrorCodes.ecTOO_FEW_PARAMS)
        check_throw("multstrfun(1,2)", EErrorCodes.ecSTRING_EXPECTED)
        check_throw('multstrfun(1,"hello")', EErrorCodes.ecSTRING_EXPECTED)
        check_throw('multstrfun("a","b")', EErrorCodes.ecVAL_EXPECTED)
        check_throw("strfun2(a,1,1)", EErrorCodes.ecTOO_MANY_PARAMS)
        check_throw("strfun3(1,2,3)", EErrorCodes.ecSTRING_EXPECTED)
        check_throw('strfun3(1, "100",3)', EErrorCodes.ecSTRING_EXPECTED)
        check_throw('strfun3("1", "100",3)', EErrorCodes.ecVAL_EXPECTED)
        check_throw('strfun3("1", 3, "100")', EErrorCodes.ecVAL_EXPECTED)
        check_throw('strfun3("1", "100", "100", "100")', EErrorCodes.ecTOO_MANY_PARAMS)

        check_throw("3=4", EErrorCodes.ecUNEXPECTED_OPERATOR)
        check_throw("sin(8)=4", EErrorCodes.ecUNEXPECTED_OPERATOR)
        check_throw('"test"=a', EErrorCodes.ecSTR_RESULT)
        check_throw("(8)=5", EErrorCodes.ecUNEXPECTED_OPERATOR)
        check_throw("(a)=5", EErrorCodes.ecUNEXPECTED_OPERATOR)
        check_throw('a="tttt"', EErrorCodes.ecOPRT_TYPE_CONFLICT)


class TestMuParserMiscFeatures:
    def test_interface_and_remove_var(self):
        p = Parser()
        p.DefineVar("a", 1.0)
        p.DefineVar("b", 2.0)
        p.DefineVar("c", 3.0)
        p.SetExpr("a+b+c")
        assert p.Eval() == pytest.approx(6.0)

        p.RemoveVar("c")
        with pytest.raises(ParserError):
            p.Eval()

    def test_optimizer_issue_93(self):
        p = Parser()
        p.DefineFun("unoptimizable", f1of1, False)
        p.SetExpr("unoptimizable(1)")
        assert p.Eval() == pytest.approx(1.0)
        bc = p.GetByteCode()
        toks = bc.GetBase()
        assert toks[1].Cmd == ECmdCode.cmFUNC

        p.ClearFun()
        p.DefineFun("unoptimizable", f1of1, True)
        p.SetExpr("unoptimizable(1)")
        assert p.Eval() == pytest.approx(1.0)
        bc = p.GetByteCode()
        toks = bc.GetBase()
        assert toks[0].Cmd == ECmdCode.cmVAL

    def test_localization(self):
        p = Parser()
        p.SetDecSep(",")
        p.SetArgSep(";")
        p.SetThousandsSep(".")
        p.DefineVar("a", 1.0)
        p.DefineVar("b", 2.0)
        p.SetExpr("1,2")
        assert p.Eval() == pytest.approx(1.2)

        p.SetExpr("max(1.000,5; 2,5)")
        assert p.Eval() == pytest.approx(1000.5)

        p.ResetLocale()
        p.SetExpr("1.2")
        assert p.Eval() == pytest.approx(1.2)

    def test_issue_165_and_168(self):
        p = Parser()
        p.DefineFun("strlen", str_len)
        p.SetExpr('strlen("1")+strlen("22")+strlen("333")')
        assert p.Eval() == pytest.approx(6.0)
        assert len(p._string_buf) == 3

        p.DefineStrConst("s1", "hello")
        p.DefineStrConst("s2", "world")
        p.SetExpr("strlen(s1)+strlen(s2)")
        assert p.Eval() == pytest.approx(10.0)

        p.ClearConst()
        p.DefineStrConst("s1", "hi")
        p.DefineStrConst("s2", "!")
        p.SetExpr("strlen(s1)+strlen(s2)")
        assert p.Eval() == pytest.approx(3.0)
        assert len(p._string_var_buf) == 2

    def test_bulk_mode(self):
        def run_bulk(expr: str) -> list[float]:
            p = Parser()
            p.DefineConst("const1", 1.0)
            p.DefineConst("const2", 2.0)
            p.DefineVar("a", [1.0, 2.0, 3.0, 4.0])
            p.DefineVar("b", [2.0, 2.0, 2.0, 2.0])
            p.DefineVar("c", [3.0, 3.0, 3.0, 3.0])
            results = [0.0, 0.0, 0.0, 0.0]
            p.SetExpr(expr)
            p.Eval(results, 4)
            return results

        assert run_bulk("a") == pytest.approx([1.0, 2.0, 3.0, 4.0])
        assert run_bulk("b=a") == pytest.approx([1.0, 2.0, 3.0, 4.0])
        assert run_bulk("b=a, b*10") == pytest.approx([10.0, 20.0, 30.0, 40.0])
        assert run_bulk("b=a, b*10, a") == pytest.approx([1.0, 2.0, 3.0, 4.0])
        assert run_bulk("a+b") == pytest.approx([3.0, 4.0, 5.0, 6.0])
        assert run_bulk("c*(a+b)") == pytest.approx([9.0, 12.0, 15.0, 18.0])

    def test_has_fun_and_diff_and_cot(self):
        p = Parser()
        assert p.HasFun("sin")
        assert p.HasFun("cot")
        assert p.HasFun("coth")
        assert p.HasFun("rnd")
        assert not p.HasFun("nonexistent_fun")

        # Test cot and coth
        p.SetExpr("cot(0.5)")
        assert p.Eval() == pytest.approx(1.0 / math.tan(0.5))
        p.SetExpr("coth(0.5)")
        assert p.Eval() == pytest.approx(1.0 / math.tanh(0.5))

        # Test numerical differentiation Diff()
        v = p.DefineVar("x", 0.0)
        p.SetExpr("x^3 + 2*x^2 - 5*x + 1")
        # Derivative is 3*x^2 + 4*x - 5; at x=2 -> 12 + 8 - 5 = 15
        assert p.Diff(v, 2.0) == pytest.approx(15.0, rel=1e-6)
        assert p.diff("x", 0.0) == pytest.approx(-5.0, rel=1e-6)

    def test_var_factory_and_bytecode_export(self):
        p = Parser()
        created = {}

        def factory(name: str, user_data):
            ref = Variable(user_data, name)
            created[name] = ref
            return ref

        p.SetVarFactory(factory, 10.0)
        p.SetExpr("new_var_1 + new_var_2 * 2")
        assert p.Eval() == pytest.approx(30.0)
        assert "new_var_1" in created and "new_var_2" in created

        # Test bytecode export and import
        bc = p.GetByteCode()
        p2 = Parser()
        p2.DefineVar("new_var_1", created["new_var_1"])
        p2.DefineVar("new_var_2", created["new_var_2"])
        p2.SetByteCode(bc)
        assert p2.Eval() == pytest.approx(30.0)

    def test_polynomial_and_division_optimizations(self):
        p = Parser()
        p.DefineVar("x", 3.0)
        p.SetExpr("x^3 + x^4 + x*x + 4*x/2 - (x + x)")
        # 27 + 81 + 9 + 6 - 6 = 117
        assert p.Eval() == pytest.approx(117.0)

        # Test rnd()
        p.SetExpr("rnd()")
        r = p.Eval()
        assert 0.0 <= r < 1.0

    def test_escaped_quotes_and_limits_and_errors(self):
        p = Parser()
        p.DefineFun("strlen", str_len)
        p.SetExpr(r'strlen("\"abc\"")')
        assert p.Eval() == pytest.approx(5.0)

        # Control character error (ASCII 15)
        with pytest.raises(ParserError) as exc_info:
            p.SetExpr("1 + \x0f 2")
            p.Eval()
        assert exc_info.value.GetCode() == EErrorCodes.ecINVALID_CHARACTERS_FOUND

        # Expression too long
        with pytest.raises(ParserError) as exc_info:
            p.SetExpr("1+" * 10005)
        assert exc_info.value.GetCode() == EErrorCodes.ecEXPRESSION_TOO_LONG

        # Locale conflict (arg_sep == dec_sep)
        p_loc = Parser()
        p_loc.SetDecSep(",")
        with pytest.raises(ParserError) as exc_info:
            p_loc.SetExpr("1,2")
        assert exc_info.value.GetCode() == EErrorCodes.ecLOCALE

        # ParserInt division by zero, overflow, and unary !
        check_eqn_int("!0", 1, True)
        check_eqn_int("!5", 0, True)
        check_eqn_int("abs(-7) + sign(-7) + sign(0) + sign(7)", 7, True)
        check_eqn_int("sum(1, 2, 3) + min(4, 2) + max(4, 2)", 12, True)
        check_throw_int("10 / 0", EErrorCodes.ecDIV_BY_ZERO)
        check_throw_int("10 % 0", EErrorCodes.ecDIV_BY_ZERO)
        check_throw_int("2147483647 + 1", EErrorCodes.ecDOMAIN_ERROR)
        check_throw_int("-2147483647 - 2", EErrorCodes.ecDOMAIN_ERROR)
        check_throw_int("2147483647 * 2", EErrorCodes.ecDOMAIN_ERROR)

