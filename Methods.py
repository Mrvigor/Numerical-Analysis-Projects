import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#  Euler's Method
def euler(f, t0, y0, h, N):
    t = np.zeros(N+1)
    y = np.zeros(N+1)

    t[0] = t0
    y[0] = y0

    for n in range(N):
        t[n+1] = t[n] + h
        y[n+1] = y[n] + h * f(t[n], y[n])

    return t, y

# Taylor's Method (Second order)
def taylor2(f, g, t0, y0, h, N):

    t = np.zeros(N+1)
    y = np.zeros(N+1)

    t[0] = t0
    y[0] = y0

    for n in range(N):
        t[n+1] = t[n] + h

        y[n+1] = (
            y[n]
            + h*f(t[n], y[n])
            + 0.5*h**2*g(t[n], y[n])
        )

    return t, y

# Midpoint method
def midpoint(f, t0, y0, h, N):

    t = np.zeros(N+1)
    y = np.zeros(N+1)

    t[0] = t0
    y[0] = y0

    for n in range(N):

        k1 = f(t[n], y[n])

        k2 = f(
            t[n] + h/2,
            y[n] + h*k1/2
        )

        y[n+1] = y[n] + h*k2
        t[n+1] = t[n] + h

    return t, y

# Heun's Method
def heun(f, t0, y0, h, N):

    t = np.zeros(N+1)
    y = np.zeros(N+1)

    t[0] = t0
    y[0] = y0

    for n in range(N):

        k1 = f(t[n], y[n])

        k2 = f(
            t[n] + h,
            y[n] + h*k1
        )

        y[n+1] = y[n] + h*(k1+k2)/2
        t[n+1] = t[n] + h

    return t, y

# RK4
def rk4(f, t0, y0, h, N):

    t = np.zeros(N+1)
    y = np.zeros(N+1)

    t[0] = t0
    y[0] = y0

    for n in range(N):

        k1 = f(t[n], y[n])

        k2 = f(
            t[n] + h/2,
            y[n] + h*k1/2
        )

        k3 = f(
            t[n] + h/2,
            y[n] + h*k2/2
        )

        k4 = f(
            t[n] + h,
            y[n] + h*k3
        )

        y[n+1] = y[n] + h*(k1 + 2*k2 + 2*k3 + k4)/6
        t[n+1] = t[n] + h

    return t, y

# Admas-B
def adams_b(f, t0, y0, h, N):

    t = np.zeros(N+1)
    y = np.zeros(N+1)

    t[0] = t0
    y[0] = y0

    # RK4 startup
    k1 = f(t0,y0)
    k2 = f(t0+h/2, y0+h*k1/2)
    k3 = f(t0+h/2, y0+h*k2/2)
    k4 = f(t0+h, y0+h*k3)

    y[1] = y0 + h*(k1+2*k2+2*k3+k4)/6
    t[1] = t0 + h

    for n in range(1,N):

        fn = f(t[n], y[n])
        fn1 = f(t[n-1], y[n-1])

        y[n+1] = (
            y[n]
            + h*(3*fn - fn1)/2
        )

        t[n+1] = t[n] + h

    return t, y

# Adams-M
def adams_m(f, dfdy, t0, y0, h, N):

    import numpy as np

    t = np.zeros(N+1)
    y = np.zeros(N+1)

    t[0] = t0
    y[0] = y0

    # Use RK4 for y1
    k1 = f(t0,y0)
    k2 = f(t0+h/2, y0+h*k1/2)
    k3 = f(t0+h/2, y0+h*k2/2)
    k4 = f(t0+h, y0+h*k3)

    y[1] = y0 + h*(k1+2*k2+2*k3+k4)/6
    t[1] = t0 + h

    for n in range(1,N):

        t[n+1] = t[n] + h

        fn  = f(t[n],y[n])
        fn1 = f(t[n-1],y[n-1])

        z = y[n]      # initial guess

        for _ in range(10):

            G = (
                z
                - y[n]
                - h*(5*f(t[n+1],z)+8*fn-fn1)/12
            )

            dG = (
                1
                - 5*h*dfdy(t[n+1],z)/12
            )

            z = z - G/dG

        y[n+1] = z

    return t,y


# Predictor-Corrector
def pre_cor(f, t0, y0, h, N):

    t = np.zeros(N+1)
    y = np.zeros(N+1)

    t[0] = t0
    y[0] = y0

    # RK4 startup
    k1 = f(t0,y0)
    k2 = f(t0+h/2, y0+h*k1/2)
    k3 = f(t0+h/2, y0+h*k2/2)
    k4 = f(t0+h, y0+h*k3)

    y[1] = y0 + h*(k1+2*k2+2*k3+k4)/6
    t[1] = t0 + h

    for n in range(1,N):

        fn = f(t[n], y[n])
        fn1 = f(t[n-1], y[n-1])

        # predictor
        yp = y[n] + h*(3*fn - fn1)/2

        fp = f(t[n]+h, yp)

        # corrector
        y[n+1] = (
            y[n]
            + h*(5*fp + 8*fn - fn1)/12
        )

        t[n+1] = t[n] + h

    return t, y

# error table function
def error_table(exact_solution, method_output):

    t, y_approx = method_output
    y_exact = exact_solution(t)
    abs_error = np.abs(y_exact - y_approx)

    table = pd.DataFrame({
        "t": t,
        "Exact Solution": y_exact,
        "Approximation": y_approx,
        "Absolute Error": abs_error
    })

    return table

def plot_solution_and_error(exact_solution, method_output, method_name="Numerical Method"):

    t, y_approx = method_output
    y_exact = exact_solution(t)
    abs_error = np.abs(y_exact - y_approx)

    # Plot exact solution and numerical approximation
    plt.figure(figsize=(8, 5))
    plt.plot(t, y_exact, label="Exact Solution", linewidth=2)
    plt.plot(t, y_approx, marker="o", linestyle="--", label=method_name)
    plt.xlabel("t")
    plt.ylabel("y")
    plt.title(f"Exact Solution vs {method_name}")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot absolute error
    plt.figure(figsize=(8, 5))
    plt.plot(t, abs_error, marker="o", label="Absolute Error")
    plt.xlabel("t")
    plt.ylabel("Absolute Error")
    plt.title(f"Absolute Error of {method_name}")
    plt.legend()
    plt.grid(True)
    plt.show()


# comparison plot among methods

def plot_methods_comparison(exact_solution, method_results, error_scale="linear"):

    # Use the time grid from the first method
    first_method_output = next(iter(method_results.values()))
    t, _ = first_method_output
    y_exact = exact_solution(t)

    # Plot exact solution and all approximations
    plt.figure(figsize=(9, 5))
    plt.plot(t, y_exact, label="Exact Solution", linewidth=2)

    for method_name, method_output in method_results.items():
        t_method, y_approx = method_output
        plt.plot(t_method, y_approx, marker="o", linestyle="--", label=method_name)

    plt.xlabel("t")
    plt.ylabel("y")
    plt.title("Comparison of Numerical Methods")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot absolute error for all methods
    plt.figure(figsize=(9, 5))

    for method_name, method_output in method_results.items():
        t_method, y_approx = method_output
        y_exact_method = exact_solution(t_method)
        abs_error = np.abs(y_exact_method - y_approx)
        plt.plot(t_method, abs_error, marker="o", label=method_name)

    if error_scale == "log":
        plt.yscale("log")

    plt.xlabel("t")
    plt.ylabel("Absolute Error")
    plt.title("Absolute Error Comparison")
    plt.legend()
    plt.grid(True)
    plt.show()




