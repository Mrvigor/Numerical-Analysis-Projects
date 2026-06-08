from Methods import *
import numpy as np

# Problem A
def f(t,y): # f function
    return y - t**2 + 1

def ff(t,y): # y''
    return y - t**2 - 2*t + 1
def g(t): # exact solution
    return (t + 1)**2 - 0.5*np.exp(t)

def dfdy(t,y):
    return 1

def runfunc(h,N,t0,y0):
# Run all methods
    results = {
        "Euler": euler(f, t0, y0, h, N),
        "Taylor2": taylor2(f, ff, t0, y0, h, N),
        "Midpoint": midpoint(f, t0, y0, h, N),
        "Heun": heun(f, t0, y0, h, N),
        "RK4": rk4(f, t0, y0, h, N),
        "Adams-Bashforth": adams_b(f, t0, y0, h, N),
        "Adams-Moulton": adams_m(f,dfdy, t0, y0, h, N),
        "Predictor-corrector": pre_cor(f, t0, y0, h, N)
    }

    # Print error tables
    for method_name, result in results.items():
        print(f"\n{'='*60}")
        print(method_name)
        print(f"{'='*60}")
        print(error_table(g, result))

    # Comparison plots
    plot_methods_comparison(g, results)

# For step size h= 0.2
# runfunc(0.2, 10, 0, 0.5)

# For step size h = 0.1
# runfunc(0.1, 20, 0, 0.5)

# For step size h = 0.05
# runfunc(0.05, 40, 0, 0.5)

# Problem B
def f(t,y): # f function
    return 2* y

def ff(t,y): # y''
    return 4*y

def g(t): # exact solution
    return np.exp(2* t)

def dfdy(t,y):
    return 2

# For step size h= 0.2
# runfunc(0.2, 10, 0, 1)

# For step size h = 0.1
# runfunc(0.1, 20, 0, 1)

# For step size h = 0.05
# runfunc(0.05, 40, 0, 1)

# Problem C
def f(t, y):
    return -15*y

def g(t):
    return np.exp(-15*t)

def ff(t, y):
    return 225*y

def dfdy(t,y):
    return -15

# For step size h= 0.2
runfunc(0.2, 10, 0, 1)

# For step size h = 0.1
runfunc(0.1, 20, 0, 1)

# For step size h = 0.05
runfunc(0.05, 40, 0, 1)