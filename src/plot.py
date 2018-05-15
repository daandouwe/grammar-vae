#!/usr/bin/env python

from math import sin, exp

import matplotlib.pyplot as plt
from sympy import lambdify
from sympy.abc import x
from sympy.parsing.sympy_parser import parse_expr

functions = []

with open('../data/equation2_15_dataset.txt') as f:
    for _ in range(10):
        eq = f.readline().strip()
        # x = 2
        # ans = eval(eq)
        print(eq.__repr__())

        # Using sympy:
        expr = parse_expr(eq)
        fun = lambdify(x, expr(x))
        ans = f(5)
        print(ans)
