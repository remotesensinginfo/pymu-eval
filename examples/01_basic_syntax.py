"""Example 1: Basic muParser syntax, custom operators, functions, and integer mode."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pymu_eval import (
    EOprtAssociativity,
    EOprtPrecedence,
    Parser,
    ParserError,
    ParserInt,
    eval_expr,
)


def main() -> None:
    print("=== 1. One-shot Scalar Expressions ===")
    expr = "2 * _pi * r + sin(_pi / 4)^2 + cos(_pi / 4)^2"
    res = eval_expr(expr, r=5.0)
    print(f"Expression : {expr} (with r=5.0)")
    print(f"Result     : {res:.6f}\n")

    print("=== 2. Reusable Parser with Variables & Constants ===")
    p = Parser()
    x_ref = p.define_var("x", 3.0)
    y_ref = p.define_var("y", 4.0)
    p.define_const("scale", 2.5)
    p.set_expr("scale * sqrt(x^2 + y^2)")
    print(f"Initial (x=3, y=4): {p.eval()}")
    x_ref.value = 6.0
    y_ref.value = 8.0
    print(f"Updated (x=6, y=8): {p.eval()}\n")

    print("=== 3. Ternary Conditional (if-then-else) & Multi-Expressions ===")
    p_cond = Parser()
    a = p_cond.define_var("a", -5.0)
    b = p_cond.define_var("b", 0.0)
    p_cond.set_expr("b = (a < 0 ? -a : a * 2), b + 10")
    results = p_cond.eval_multi()
    print(f"Multi-expression results: {results}, updated b = {b.value}\n")

    print("=== 4. Custom Functions, Postfix, Infix & Binary Operators ===")
    p_custom = Parser()
    p_custom.define_fun("hypot3", lambda u, v, w: (u**2 + v**2 + w**2) ** 0.5)
    p_custom.define_postfix_oprt("k", lambda v: v * 1000.0)
    p_custom.define_infix_oprt("~", lambda v: -v - 1.0)
    p_custom.define_oprt(
        "#",
        lambda u, v: (u + v) / 2.0,
        prec=EOprtPrecedence.prADD_SUB,
        associativity=EOprtAssociativity.oaLEFT,
    )
    p_custom.set_expr("hypot3(3, 4, 12) + 2k + (~4) + (10 # 20)")
    print(f"Custom operators & functions result: {p_custom.eval()}\n")

    print("=== 5. Numerical Differentiation (5-point 4th-order) ===")
    p_diff = Parser("sin(x) * exp(-0.2 * x)")
    x_d = p_diff.define_var("x", 1.0)
    deriv = p_diff.diff(x_d)
    print(f"d/dx [sin(x)*exp(-0.2*x)] at x=1.0: {deriv:.8f}\n")

    print("=== 6. Integer Parser (Hex, Binary, Bitwise) ===")
    p_int = ParserInt("(0xff & #10101010) | (1 << 4)")
    print(f"ParserInt '(0xff & #10101010) | (1 << 4)' = {p_int.eval()}\n")

    print("=== 7. Structured Error Handling ===")
    try:
        eval_expr("sin(x + * 2)", x=1.0)
    except ParserError as err:
        print(f"Caught ParserError: code={err.code.name}, token={err.token!r}, pos={err.pos}")
        print(f"Message: {err.msg}")


if __name__ == "__main__":
    main()
