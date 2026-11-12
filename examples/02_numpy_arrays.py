"""Example 2: Vectorized mathematical expressions with 1D, 2D, and 3D NumPy arrays."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from pymu_eval import Parser, eval_expr


def main() -> None:
    print("=== 1. 1D Signal Processing & Damped Harmonic Oscillator ===")
    t = np.linspace(0.0, 10.0, 6)
    omega = 2.0 * np.pi * 0.5  # 0.5 Hz
    zeta = 0.15                # damping ratio

    y = eval_expr(
        "exp(-zeta * omega * t) * cos(omega * sqrt(1 - zeta^2) * t)",
        t=t,
        omega=omega,
        zeta=zeta,
    )
    print("t :", np.round(t, 2))
    print("y :", np.round(y, 4), "\n")

    print("=== 2. Piecewise Functions via Vectorized Ternary (? :) ===")
    x = np.linspace(-3.0, 3.0, 7)
    # Soft-clipping / Huber-like piecewise loss function:
    # |x| <= 1.5 ? 0.5 * x^2 : 1.5 * (abs(x) - 0.75)
    huber = eval_expr("abs(x) <= 1.5 ? 0.5 * x^2 : 1.5 * (abs(x) - 0.75)", x=x)
    print("x     :", x)
    print("huber :", np.round(huber, 4), "\n")

    print("=== 3. 2D Spatial Grid Broadcasting & Variadic Aggregations ===")
    gx = np.linspace(-2.0, 2.0, 5)[np.newaxis, :]  # (1, 5)
    gy = np.linspace(-2.0, 2.0, 4)[:, np.newaxis]  # (4, 1)

    p = Parser()
    p.define_var("x", gx)
    p.define_var("y", gy)
    # Compute radially symmetric Mexican Hat (Ricker) wavelet in 2D
    p.set_expr("(1 - (x^2 + y^2)) * exp(-(x^2 + y^2) / 2)")
    wavelet_2d = p.eval()
    print(f"2D Wavelet shape: {wavelet_2d.shape}")
    print(np.round(wavelet_2d, 3), "\n")

    print("=== 4. Element-Wise Numerical Derivative over a NumPy Array ===")
    p_deriv = Parser("x^3 - 3*x")
    p_deriv.define_var("x", x)
    dy_dx = p_deriv.diff("x")
    expected = 3.0 * x**2 - 3.0
    print("x           :", x)
    print("p.diff('x') :", np.round(dy_dx, 6))
    print("3*x^2 - 3   :", np.round(expected, 6))


if __name__ == "__main__":
    main()
