import numpy as np
import pandas as pd
# Helper: Thomas algorithm for tridiagonal systems

def thomas_algorithm(lower, main, upper, rhs):
    lower = np.array(lower, dtype=float)
    main = np.array(main, dtype=float)
    upper = np.array(upper, dtype=float)
    rhs = np.array(rhs, dtype=float)

    n = len(rhs)

    for i in range(1, n):
        factor = lower[i - 1] / main[i - 1]
        main[i] = main[i] - factor * upper[i - 1]
        rhs[i] = rhs[i] - factor * rhs[i - 1]

    x = np.zeros(n)
    x[-1] = rhs[-1] / main[-1]

    for i in range(n - 2, -1, -1):
        x[i] = (rhs[i] - upper[i] * x[i + 1]) / main[i]

    return x


# Five-point finite-difference method for elliptic PDEs
def five_point_elliptic(
    N,
    boundary_func,
    source_func=None,
    method="gauss_seidel",
    omega=1.5,
    tol=1e-10,
    max_iter=100000,
):
    h = 1.0 / N
    x_grid = np.linspace(0.0, 1.0, N + 1)
    y_grid = np.linspace(0.0, 1.0, N + 1)

    U = np.zeros((N + 1, N + 1))

    # Set boundary values
    for i in range(N + 1):
        U[i, 0] = boundary_func(x_grid[i], 0.0)
        U[i, N] = boundary_func(x_grid[i], 1.0)

    for j in range(N + 1):
        U[0, j] = boundary_func(0.0, y_grid[j])
        U[N, j] = boundary_func(1.0, y_grid[j])

    if source_func is None:
        def source_func(x, y):
            return 0.0

    if method not in {"gauss_seidel", "sor"}:
        raise ValueError("method must be 'gauss_seidel' or 'sor'.")

    for iteration in range(1, max_iter + 1):
        max_diff = 0.0

        for i in range(1, N):
            for j in range(1, N):
                old_value = U[i, j]

                gs_value = 0.25 * (
                    U[i + 1, j]
                    + U[i - 1, j]
                    + U[i, j + 1]
                    + U[i, j - 1]
                    - h**2 * source_func(x_grid[i], y_grid[j])
                )

                if method == "gauss_seidel":
                    U[i, j] = gs_value
                else:
                    U[i, j] = (1.0 - omega) * old_value + omega * gs_value

                max_diff = max(max_diff, abs(U[i, j] - old_value))

        if max_diff < tol:
            return x_grid, y_grid, U, iteration

    return x_grid, y_grid, U, max_iter


# Forward-Difference method for heat equation
def forward_difference_heat(
    h,
    k,
    T,
    initial_func,
    left_boundary_func=None,
    right_boundary_func=None,
    x_start=0.0,
    x_end=1.0,
):
    if left_boundary_func is None:
        left_boundary_func = lambda t: 0.0
    if right_boundary_func is None:
        right_boundary_func = lambda t: 0.0

    M = int(round((x_end - x_start) / h))
    Nt = int(round(T / k))

    x_grid = np.linspace(x_start, x_end, M + 1)
    t_grid = np.linspace(0.0, T, Nt + 1)

    U = np.zeros((Nt + 1, M + 1))
    U[0, :] = initial_func(x_grid)

    lam = k / h**2

    for n in range(Nt):
        U[n, 0] = left_boundary_func(t_grid[n])
        U[n, M] = right_boundary_func(t_grid[n])

        for i in range(1, M):
            U[n + 1, i] = (
                lam * U[n, i - 1]
                + (1.0 - 2.0 * lam) * U[n, i]
                + lam * U[n, i + 1]
            )

        U[n + 1, 0] = left_boundary_func(t_grid[n + 1])
        U[n + 1, M] = right_boundary_func(t_grid[n + 1])

    return x_grid, t_grid, U, lam


# Backward-Difference method for heat equation
def backward_difference_heat(
    h,
    k,
    T,
    initial_func,
    left_boundary_func=None,
    right_boundary_func=None,
    x_start=0.0,
    x_end=1.0,
):
    if left_boundary_func is None:
        left_boundary_func = lambda t: 0.0
    if right_boundary_func is None:
        right_boundary_func = lambda t: 0.0

    M = int(round((x_end - x_start) / h))
    Nt = int(round(T / k))

    x_grid = np.linspace(x_start, x_end, M + 1)
    t_grid = np.linspace(0.0, T, Nt + 1)

    U = np.zeros((Nt + 1, M + 1))
    U[0, :] = initial_func(x_grid)

    lam = k / h**2
    interior_size = M - 1

    lower = -lam * np.ones(interior_size - 1)
    main = (1.0 + 2.0 * lam) * np.ones(interior_size)
    upper = -lam * np.ones(interior_size - 1)

    for n in range(Nt):
        rhs = U[n, 1:M].copy()

        rhs[0] += lam * left_boundary_func(t_grid[n + 1])
        rhs[-1] += lam * right_boundary_func(t_grid[n + 1])

        U[n + 1, 1:M] = thomas_algorithm(lower, main, upper, rhs)
        U[n + 1, 0] = left_boundary_func(t_grid[n + 1])
        U[n + 1, M] = right_boundary_func(t_grid[n + 1])

    return x_grid, t_grid, U, lam


#  Crank-Nicolson method for heat equation


def crank_nicolson_heat(
    h,
    k,
    T,
    initial_func,
    left_boundary_func=None,
    right_boundary_func=None,
    x_start=0.0,
    x_end=1.0,
):
    if left_boundary_func is None:
        left_boundary_func = lambda t: 0.0
    if right_boundary_func is None:
        right_boundary_func = lambda t: 0.0

    M = int(round((x_end - x_start) / h))
    Nt = int(round(T / k))

    x_grid = np.linspace(x_start, x_end, M + 1)
    t_grid = np.linspace(0.0, T, Nt + 1)

    U = np.zeros((Nt + 1, M + 1))
    U[0, :] = initial_func(x_grid)

    lam = k / h**2
    interior_size = M - 1

    lower = -lam * np.ones(interior_size - 1)
    main = 2.0 * (1.0 + lam) * np.ones(interior_size)
    upper = -lam * np.ones(interior_size - 1)

    for n in range(Nt):
        rhs = np.zeros(interior_size)

        for i in range(1, M):
            rhs[i - 1] = (
                lam * U[n, i - 1]
                + 2.0 * (1.0 - lam) * U[n, i]
                + lam * U[n, i + 1]
            )

        rhs[0] += lam * left_boundary_func(t_grid[n + 1])
        rhs[-1] += lam * right_boundary_func(t_grid[n + 1])

        U[n + 1, 1:M] = thomas_algorithm(lower, main, upper, rhs)
        U[n + 1, 0] = left_boundary_func(t_grid[n + 1])
        U[n + 1, M] = right_boundary_func(t_grid[n + 1])

    return x_grid, t_grid, U, lam


# Utility functions for error and plotting

def compute_error_table_1d(x_grid, t_value, numerical_values, exact_func):
    exact_values = exact_func(x_grid, t_value)
    errors = np.abs(exact_values - numerical_values)

    return pd.DataFrame({
        "x": x_grid,
        "t": t_value,
        "Exact": exact_values,
        "Approximation": numerical_values,
        "Absolute Error": errors,
    })



def max_error_1d(x_grid, t_grid, U, exact_func):
    """
    Compute maximum absolute error over all mesh points for a 1D time-dependent PDE.
    """
    max_error = 0.0

    for n, t_value in enumerate(t_grid):
        exact_values = exact_func(x_grid, t_value)
        error = np.max(np.abs(exact_values - U[n, :]))
        max_error = max(max_error, error)

    return max_error



def compute_error_table_2d(x_grid, y_grid, U, exact_func, include_boundary=True):
    rows = []
    nx = len(x_grid)
    ny = len(y_grid)

    i_start = 0 if include_boundary else 1
    i_end = nx if include_boundary else nx - 1
    j_start = 0 if include_boundary else 1
    j_end = ny if include_boundary else ny - 1

    for i in range(i_start, i_end):
        for j in range(j_start, j_end):
            x = x_grid[i]
            y = y_grid[j]
            exact_value = exact_func(x, y)
            approximation = U[i, j]
            abs_error = abs(exact_value - approximation)

            rows.append({
                "x": x,
                "y": y,
                "Exact": exact_value,
                "Approximation": approximation,
                "Absolute Error": abs_error,
            })

    return pd.DataFrame(rows)



def max_error_2d(x_grid, y_grid, U, exact_func, include_boundary=True):
    max_error = 0.0
    nx = len(x_grid)
    ny = len(y_grid)

    i_start = 0 if include_boundary else 1
    i_end = nx if include_boundary else nx - 1
    j_start = 0 if include_boundary else 1
    j_end = ny if include_boundary else ny - 1

    for i in range(i_start, i_end):
        for j in range(j_start, j_end):
            exact_value = exact_func(x_grid[i], y_grid[j])
            max_error = max(max_error, abs(exact_value - U[i, j]))

    return max_error




def plot_2d_error(x_grid, y_grid, U, exact_func, title="Absolute Error"):
    import matplotlib.pyplot as plt

    X, Y = np.meshgrid(x_grid, y_grid, indexing="ij")
    exact_values = exact_func(X, Y)
    error = np.abs(exact_values - U)

    plt.figure(figsize=(7, 6))
    contour = plt.contourf(X, Y, error, levels=20, cmap="viridis")
    plt.colorbar(contour, label="Absolute Error")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.show()



def mesh_refinement_table_1d(method_func, h_values, k_func, T, initial_func, exact_func,
                             left_boundary_func=None, right_boundary_func=None,
                             x_start=0.0, x_end=1.0):
    rows = []

    for h in h_values:
        k = k_func(h)
        x_grid, t_grid, U, lam = method_func(
            h=h,
            k=k,
            T=T,
            initial_func=initial_func,
            left_boundary_func=left_boundary_func,
            right_boundary_func=right_boundary_func,
            x_start=x_start,
            x_end=x_end,
        )

        max_err = max_error_1d(x_grid, t_grid, U, exact_func)

        rows.append({
            "h": h,
            "k": k,
            "lambda": lam,
            "Max Absolute Error": max_err,
        })

    return pd.DataFrame(rows)



def mesh_refinement_table_2d(N_values, boundary_func, exact_func, source_func=None,
                             method="sor", omega=1.5, tol=1e-10, max_iter=100000):
    rows = []

    for N in N_values:
        x_grid, y_grid, U, iterations = five_point_elliptic(
            N=N,
            boundary_func=boundary_func,
            source_func=source_func,
            method=method,
            omega=omega,
            tol=tol,
            max_iter=max_iter,
        )

        max_err = max_error_2d(
            x_grid=x_grid,
            y_grid=y_grid,
            U=U,
            exact_func=exact_func,
            include_boundary=False,
        )

        rows.append({
            "N": N,
            "h": 1.0 / N,
            "Max Absolute Error": max_err,
            "Iterations": iterations,
        })

    return pd.DataFrame(rows)



def plot_mesh_refinement(error_table, title="Effect of Mesh Refinement"):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 5))
    plt.loglog(error_table["h"], error_table["Max Absolute Error"], marker="o")
    plt.xlabel("h")
    plt.ylabel("Maximum Absolute Error")
    plt.title(title)
    plt.grid(True, which="both")
    plt.show()



def plot_1d_error_final_time(x_grid, t_grid, method_results, exact_func):
    import matplotlib.pyplot as plt

    final_time = t_grid[-1]

    plt.figure(figsize=(8, 5))

    for method_name, U in method_results.items():
        exact_values = exact_func(x_grid, final_time)
        error = np.abs(exact_values - U[-1, :])
        plt.plot(x_grid, error, marker="o", linestyle="--", label=method_name)

    plt.xlabel("x")
    plt.ylabel("Absolute Error")
    plt.title(f"Absolute Error at t = {final_time:.4f}")
    plt.legend()
    plt.grid(True)
    plt.show()



def plot_1d_final_time(x_grid, t_grid, method_results, exact_func):
    import matplotlib.pyplot as plt

    final_time = t_grid[-1]
    exact_values = exact_func(x_grid, final_time)

    plt.figure(figsize=(8, 5))
    plt.plot(x_grid, exact_values, label="Exact", linewidth=2)

    for method_name, U in method_results.items():
        plt.plot(x_grid, U[-1, :], marker="o", linestyle="--", label=method_name)

    plt.xlabel("x")
    plt.ylabel("u(x,t)")
    plt.title(f"Comparison at t = {final_time:.4f}")
    plt.legend()
    plt.grid(True)
    plt.show()