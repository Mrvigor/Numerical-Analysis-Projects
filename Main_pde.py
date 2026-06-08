from Methods_pde import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Problem A: Laplace Equation
# ============================================================
#
# u_xx + u_yy = 0,  0 < x < 1, 0 < y < 1
#
# Boundary conditions:
# u(0,y) = 0
# u(x,0) = 0
# u(x,1) = 100x
# u(1,y) = 100y
#
# Exact solution:
# u(x,y) = 100xy
# ============================================================


def boundary_A(x, y):
    """
    Boundary condition for Problem A.
    """
    if np.isclose(x, 0.0):
        return 0.0
    if np.isclose(y, 0.0):
        return 0.0
    if np.isclose(y, 1.0):
        return 100.0 * x
    if np.isclose(x, 1.0):
        return 100.0 * y

    raise ValueError("The point is not on the boundary.")


# Since Problem A is Laplace's equation, the source term is zero.
def source_A(x, y):
    return 0.0


# Exact solution for Problem A.
def exact_A(x, y):
    return 100.0 * x * y


# Mesh sizes required for mesh refinement comparison.
N_values = [4, 8, 16]


# ============================================================
# Run five-point method with Gauss-Seidel and SOR
# ============================================================

for N in N_values:
    print("\n" + "=" * 70)
    print(f"Problem A: N = {N}, h = {1/N:.4f}")
    print("=" * 70)

    # Gauss-Seidel method
    x_gs, y_gs, U_gs, iter_gs = five_point_elliptic(
        N=N,
        boundary_func=boundary_A,
        source_func=source_A,
        method="gauss_seidel",
        tol=1e-10,
        max_iter=100000,
    )

    error_table_gs = compute_error_table_2d(
        x_grid=x_gs,
        y_grid=y_gs,
        U=U_gs,
        exact_func=exact_A,
        include_boundary=False,
    )

    print("\nGauss-Seidel iteration count:", iter_gs)
    print("Gauss-Seidel error table, interior points:")
    print(error_table_gs)
    print("Gauss-Seidel maximum absolute error:", max_error_2d(x_gs, y_gs, U_gs, exact_A, include_boundary=False))

    # SOR method
    x_sor, y_sor, U_sor, iter_sor = five_point_elliptic(
        N=N,
        boundary_func=boundary_A,
        source_func=source_A,
        method="sor",
        omega=1.5,
        tol=1e-10,
        max_iter=100000,
    )

    error_table_sor = compute_error_table_2d(
        x_grid=x_sor,
        y_grid=y_sor,
        U=U_sor,
        exact_func=exact_A,
        include_boundary=False,
    )

    print("\nSOR iteration count:", iter_sor)
    print("SOR error table, interior points:")
    print(error_table_sor)
    print("SOR maximum absolute error:", max_error_2d(x_sor, y_sor, U_sor, exact_A, include_boundary=False))


# ============================================================
# Mesh refinement tables
# ============================================================

print("\n" + "=" * 70)
print("Mesh Refinement Table: Gauss-Seidel")
print("=" * 70)

gs_refinement = mesh_refinement_table_2d(
    N_values=N_values,
    boundary_func=boundary_A,
    exact_func=exact_A,
    source_func=source_A,
    method="gauss_seidel",
    tol=1e-10,
    max_iter=100000,
)
print(gs_refinement)

print("\n" + "=" * 70)
print("Mesh Refinement Table: SOR")
print("=" * 70)

sor_refinement = mesh_refinement_table_2d(
    N_values=N_values,
    boundary_func=boundary_A,
    exact_func=exact_A,
    source_func=source_A,
    method="sor",
    omega=1.5,
    tol=1e-10,
    max_iter=100000,
)
print(sor_refinement)


# ============================================================
# Visualizations for finest grid
# ============================================================

N = 16

# Gauss-Seidel solution on the finest grid
x_gs, y_gs, U_gs, iter_gs = five_point_elliptic(
    N=N,
    boundary_func=boundary_A,
    source_func=source_A,
    method="gauss_seidel",
    tol=1e-10,
    max_iter=100000,
)

# SOR solution on the finest grid
x, y, U, iterations = five_point_elliptic(
    N=N,
    boundary_func=boundary_A,
    source_func=source_A,
    method="sor",
    omega=1.5,
    tol=1e-10,
    max_iter=100000,
)


plot_2d_error(
    x_grid=x,
    y_grid=y,
    U=U,
    exact_func=exact_A,
    title="Problem A: Absolute Error"
)

plot_mesh_refinement(
    error_table=sor_refinement,
    title="Problem A: Effect of Mesh Refinement, SOR"
)



# ============================================================
# Method comparison plots under different mesh sizes h
# ============================================================

h_values = []
gs_max_errors = []
sor_max_errors = []
gs_iterations_list = []
sor_iterations_list = []

for N in N_values:
    h = 1.0 / N
    h_values.append(h)

    # Gauss-Seidel solution for this mesh size
    x_gs, y_gs, U_gs, iter_gs = five_point_elliptic(
        N=N,
        boundary_func=boundary_A,
        source_func=source_A,
        method="gauss_seidel",
        tol=1e-10,
        max_iter=100000,
    )

    # SOR solution for this mesh size
    x_sor, y_sor, U_sor, iter_sor = five_point_elliptic(
        N=N,
        boundary_func=boundary_A,
        source_func=source_A,
        method="sor",
        omega=1.5,
        tol=1e-10,
        max_iter=100000,
    )

    # Store maximum errors and iteration counts
    gs_max_error = max_error_2d(
        x_grid=x_gs,
        y_grid=y_gs,
        U=U_gs,
        exact_func=exact_A,
        include_boundary=False,
    )

    sor_max_error = max_error_2d(
        x_grid=x_sor,
        y_grid=y_sor,
        U=U_sor,
        exact_func=exact_A,
        include_boundary=False,
    )

    gs_max_errors.append(gs_max_error)
    sor_max_errors.append(sor_max_error)
    gs_iterations_list.append(iter_gs)
    sor_iterations_list.append(iter_sor)

    # ------------------------------------------------------------
    # Approximation comparison graph for this h
    # Compare along the horizontal slice y = 0.5
    # ------------------------------------------------------------
    y_target = 0.5
    j_mid = np.argmin(np.abs(y_sor - y_target))
    y_slice = y_sor[j_mid]

    exact_slice = exact_A(x_sor, y_slice)
    gs_slice = U_gs[:, j_mid]
    sor_slice = U_sor[:, j_mid]

    plt.figure(figsize=(8, 5))
    plt.plot(x_sor, exact_slice, label="Exact Solution", linewidth=2)
    plt.plot(x_sor, gs_slice, marker="o", linestyle="--", label="Gauss-Seidel")
    plt.plot(x_sor, sor_slice, marker="s", linestyle="--", label="SOR")
    plt.xlabel("x")
    plt.ylabel("u(x,y)")
    plt.title(f"Problem A: Approximation Comparison, h = {h:.4f}, y = {y_slice:.2f}")
    plt.legend()
    plt.grid(True)
    plt.show()

    # ------------------------------------------------------------
    # Error comparison graph for this h
    # ------------------------------------------------------------
    gs_error_slice = np.abs(exact_slice - gs_slice)
    sor_error_slice = np.abs(exact_slice - sor_slice)

    plt.figure(figsize=(8, 5))
    plt.plot(x_sor, gs_error_slice, marker="o", linestyle="--", label="Gauss-Seidel Error")
    plt.plot(x_sor, sor_error_slice, marker="s", linestyle="--", label="SOR Error")
    plt.xlabel("x")
    plt.ylabel("Absolute Error")
    plt.title(f"Problem A: Error Comparison, h = {h:.4f}, y = {y_slice:.2f}")
    plt.legend()
    plt.grid(True)
    plt.show()


# ============================================================
# Summary comparison across mesh sizes
# ============================================================

# Maximum absolute error vs h
plt.figure(figsize=(8, 5))
plt.loglog(h_values, gs_max_errors, marker="o", linestyle="--", label="Gauss-Seidel")
plt.loglog(h_values, sor_max_errors, marker="s", linestyle="--", label="SOR")
plt.xlabel("h")
plt.ylabel("Maximum Absolute Error")
plt.title("Problem A: Maximum Error vs Mesh Size")
plt.legend()
plt.grid(True, which="both")

plt.show()


# ============================================================
# Problem C: Additional Numerical Experiments
# ============================================================
# Experiment 1:
#   Study the effect of mesh refinement for a Poisson equation.
# Experiment 2:
#   Compare a direct solver and an iterative SOR solver.
# ============================================================


# ============================================================
# Problem C, Experiment 1: Mesh refinement for Poisson equation
# ============================================================
# PDE:
# u_xx + u_yy = f(x,y), 0 < x < 1, 0 < y < 1
# Exact solution:
# u(x,y) = sin(pi x) sin(pi y)
# Source term:
# f(x,y) = -2 pi^2 sin(pi x) sin(pi y)
# Boundary condition:
# u = 0 on the boundary
# ============================================================


def exact_C1(x, y):
    """Exact solution for Problem C, Experiment 1."""
    return np.sin(np.pi * x) * np.sin(np.pi * y)


def source_C1(x, y):
    """Source term for u_xx + u_yy = f(x,y)."""
    return -2.0 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)


def boundary_C1(x, y):
    """Boundary condition for Experiment 1: u=0 on all boundaries."""
    return 0.0


N_values_C1 = [4, 8, 16, 32]
C1_max_errors = []
C1_iterations = []
C1_h_values = []

print("\n" + "=" * 70)
print("Problem C, Experiment 1: Mesh Refinement for Poisson Equation")
print("=" * 70)

for N in N_values_C1:
    h = 1.0 / N
    C1_h_values.append(h)

    x_C1, y_C1, U_C1, iter_C1 = five_point_elliptic(
        N=N,
        boundary_func=boundary_C1,
        source_func=source_C1,
        method="sor",
        omega=1.5,
        tol=1e-10,
        max_iter=100000,
    )

    error_table_C1 = compute_error_table_2d(
        x_grid=x_C1,
        y_grid=y_C1,
        U=U_C1,
        exact_func=exact_C1,
        include_boundary=False,
    )

    max_error_C1 = max_error_2d(
        x_grid=x_C1,
        y_grid=y_C1,
        U=U_C1,
        exact_func=exact_C1,
        include_boundary=False,
    )

    C1_max_errors.append(max_error_C1)
    C1_iterations.append(iter_C1)

    print("\n" + "-" * 70)
    print(f"N = {N}, h = {h:.5f}")
    print(f"SOR iterations: {iter_C1}")
    print(f"Maximum absolute error: {max_error_C1}")
    print("Error table, interior points:")
    print(error_table_C1)


C1_refinement_table = pd.DataFrame({
    "N": N_values_C1,
    "h": C1_h_values,
    "Max Absolute Error": C1_max_errors,
    "SOR Iterations": C1_iterations,
})

print("\nProblem C, Experiment 1: Mesh Refinement Summary")
print(C1_refinement_table)

# Plot numerical solution and error for the finest grid.
N = 32
x_C1, y_C1, U_C1, iter_C1 = five_point_elliptic(
    N=N,
    boundary_func=boundary_C1,
    source_func=source_C1,
    method="sor",
    omega=1.5,
    tol=1e-10,
    max_iter=100000,
)

# Exact vs approximation along y = 0.5
j_mid = np.argmin(np.abs(y_C1 - 0.5))
y_slice = y_C1[j_mid]
exact_slice_C1 = exact_C1(x_C1, y_slice)
approx_slice_C1 = U_C1[:, j_mid]
error_slice_C1 = np.abs(exact_slice_C1 - approx_slice_C1)

plt.figure(figsize=(8, 5))
plt.plot(x_C1, exact_slice_C1, label="Exact Solution", linewidth=2)
plt.plot(x_C1, approx_slice_C1, marker="o", linestyle="--", label="SOR Approximation")
plt.xlabel("x")
plt.ylabel("u(x,y)")
plt.title(f"Problem C1: Exact vs Approximation, h = {1/N:.5f}, y = {y_slice:.2f}")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(x_C1, error_slice_C1, marker="o", linestyle="--")
plt.xlabel("x")
plt.ylabel("Absolute Error")
plt.title(f"Problem C1: Error Along y = {y_slice:.2f}, h = {1/N:.5f}")
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 5))
plt.loglog(C1_h_values, C1_max_errors, marker="o", linestyle="--")
plt.xlabel("h")
plt.ylabel("Maximum Absolute Error")
plt.title("Problem C1: Effect of Mesh Refinement")
plt.grid(True, which="both")
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(C1_h_values, C1_iterations, marker="o", linestyle="--")
plt.xlabel("h")
plt.ylabel("SOR Iteration Count")
plt.title("Problem C1: SOR Iterations Under Mesh Refinement")
plt.grid(True)
plt.show()


# ============================================================
# Problem C, Experiment 2: Direct solver vs iterative solver
# ============================================================
# PDE:
# u_xx + u_yy = 0, 0 < x < 1, 0 < y < 1
# Exact solution:
# u(x,y) = x + y
# Boundary conditions:
# u(0,y) = y
# u(1,y) = 1 + y
# u(x,0) = x
# u(x,1) = x + 1
# ============================================================


def exact_C2(x, y):
    """Exact solution for Problem C, Experiment 2."""
    return x + y


def source_C2(x, y):
    """Source term for Laplace equation."""
    return 0.0


def boundary_C2(x, y):
    """Boundary condition from exact solution u(x,y)=x+y."""
    return x + y


def five_point_elliptic_direct(N, boundary_func, source_func=None):
    """
    Solve u_xx + u_yy = f(x,y) on [0,1]x[0,1]
    using the five-point method and a direct linear solver.

    Returns
    -------
    x_grid, y_grid, U
    """
    if source_func is None:
        source_func = lambda x, y: 0.0

    h = 1.0 / N
    x_grid = np.linspace(0.0, 1.0, N + 1)
    y_grid = np.linspace(0.0, 1.0, N + 1)

    U = np.zeros((N + 1, N + 1))

    # Boundary values
    for i in range(N + 1):
        U[i, 0] = boundary_func(x_grid[i], 0.0)
        U[i, N] = boundary_func(x_grid[i], 1.0)

    for j in range(N + 1):
        U[0, j] = boundary_func(0.0, y_grid[j])
        U[N, j] = boundary_func(1.0, y_grid[j])

    interior_points = []
    index_map = {}
    counter = 0

    for i in range(1, N):
        for j in range(1, N):
            interior_points.append((i, j))
            index_map[(i, j)] = counter
            counter += 1

    m = len(interior_points)
    A = np.zeros((m, m))
    b = np.zeros(m)

    for row, (i, j) in enumerate(interior_points):
        A[row, row] = 4.0
        b[row] = -h**2 * source_func(x_grid[i], y_grid[j])

        neighbors = [
            (i + 1, j),
            (i - 1, j),
            (i, j + 1),
            (i, j - 1),
        ]

        for ni, nj in neighbors:
            if 1 <= ni <= N - 1 and 1 <= nj <= N - 1:
                col = index_map[(ni, nj)]
                A[row, col] = -1.0
            else:
                b[row] += boundary_func(x_grid[ni], y_grid[nj])

    solution = np.linalg.solve(A, b)

    for value, (i, j) in zip(solution, interior_points):
        U[i, j] = value

    return x_grid, y_grid, U


N_values_C2 = [4, 8, 16]
C2_h_values = []
C2_direct_errors = []
C2_sor_errors = []
C2_sor_iterations = []

print("\n" + "=" * 70)
print("Problem C, Experiment 2: Direct Solver vs SOR Solver")
print("=" * 70)

for N in N_values_C2:
    h = 1.0 / N
    C2_h_values.append(h)

    # Direct solver
    x_direct, y_direct, U_direct = five_point_elliptic_direct(
        N=N,
        boundary_func=boundary_C2,
        source_func=source_C2,
    )

    direct_error = max_error_2d(
        x_grid=x_direct,
        y_grid=y_direct,
        U=U_direct,
        exact_func=exact_C2,
        include_boundary=False,
    )

    # Iterative SOR solver
    x_sor, y_sor, U_sor, iter_sor = five_point_elliptic(
        N=N,
        boundary_func=boundary_C2,
        source_func=source_C2,
        method="sor",
        omega=1.5,
        tol=1e-10,
        max_iter=100000,
    )

    sor_error = max_error_2d(
        x_grid=x_sor,
        y_grid=y_sor,
        U=U_sor,
        exact_func=exact_C2,
        include_boundary=False,
    )

    C2_direct_errors.append(direct_error)
    C2_sor_errors.append(sor_error)
    C2_sor_iterations.append(iter_sor)

    error_table_direct = compute_error_table_2d(
        x_grid=x_direct,
        y_grid=y_direct,
        U=U_direct,
        exact_func=exact_C2,
        include_boundary=False,
    )

    error_table_sor = compute_error_table_2d(
        x_grid=x_sor,
        y_grid=y_sor,
        U=U_sor,
        exact_func=exact_C2,
        include_boundary=False,
    )

    print("\n" + "-" * 70)
    print(f"N = {N}, h = {h:.5f}")
    print(f"Direct solver maximum error: {direct_error}")
    print(f"SOR maximum error: {sor_error}")
    print(f"SOR iterations: {iter_sor}")
    print("\nDirect solver error table:")
    print(error_table_direct)
    print("\nSOR solver error table:")
    print(error_table_sor)


C2_comparison_table = pd.DataFrame({
    "N": N_values_C2,
    "h": C2_h_values,
    "Direct Max Error": C2_direct_errors,
    "SOR Max Error": C2_sor_errors,
    "SOR Iterations": C2_sor_iterations,
})

print("\nProblem C, Experiment 2: Direct vs SOR Summary")
print(C2_comparison_table)

# Plot error comparison across mesh sizes
plt.figure(figsize=(8, 5))
plt.loglog(C2_h_values, C2_direct_errors, marker="o", linestyle="--", label="Direct Solver")
plt.loglog(C2_h_values, C2_sor_errors, marker="s", linestyle="--", label="SOR Solver")
plt.xlabel("h")
plt.ylabel("Maximum Absolute Error")
plt.title("Problem C2: Direct Solver vs SOR Error")
plt.legend()
plt.grid(True, which="both")
plt.show()

# Plot SOR iteration count across mesh sizes
plt.figure(figsize=(8, 5))
plt.plot(C2_h_values, C2_sor_iterations, marker="s", linestyle="--")
plt.xlabel("h")
plt.ylabel("SOR Iteration Count")
plt.title("Problem C2: SOR Iteration Count Under Mesh Refinement")
plt.grid(True)
plt.show()

# Plot direct and SOR approximation compared with exact solution along y=0.5
N = 16
x_direct, y_direct, U_direct = five_point_elliptic_direct(
    N=N,
    boundary_func=boundary_C2,
    source_func=source_C2,
)

x_sor, y_sor, U_sor, iter_sor = five_point_elliptic(
    N=N,
    boundary_func=boundary_C2,
    source_func=source_C2,
    method="sor",
    omega=1.5,
    tol=1e-10,
    max_iter=100000,
)

j_mid = np.argmin(np.abs(y_direct - 0.5))
y_slice = y_direct[j_mid]
exact_slice_C2 = exact_C2(x_direct, y_slice)
direct_slice_C2 = U_direct[:, j_mid]
sor_slice_C2 = U_sor[:, j_mid]

plt.figure(figsize=(8, 5))
plt.plot(x_direct, exact_slice_C2, label="Exact Solution", linewidth=2)
plt.plot(x_direct, direct_slice_C2, marker="o", linestyle="--", label="Direct Solver")
plt.plot(x_sor, sor_slice_C2, marker="s", linestyle="--", label="SOR Solver")
plt.xlabel("x")
plt.ylabel("u(x,y)")
plt.title(f"Problem C2: Direct vs SOR Approximation, h = {1/N:.5f}, y = {y_slice:.2f}")
plt.legend()
plt.grid(True)
plt.show()


# ============================================================
# Problem B: Heat Equation
# ============================================================
#
# u_t = u_xx, 0 < x < 1, t > 0
# u(0,t) = u(1,t) = 0
# u(x,0) = sin(pi x)
# Exact solution: u(x,t) = exp(-pi^2 t) sin(pi x)
#
# Methods compared:
# 1. Forward-Difference Method
# 2. Backward-Difference Method, linear system solved by SOR
# 3. Crank-Nicolson Method, linear system solved by SOR
#
# Required stability cases for Forward-Difference:
# Stable case: lambda = k/h^2 <= 1/2
# Unstable case: lambda = k/h^2 > 1/2
# ============================================================


def initial_B(x):
    """Initial condition for Problem B."""
    return np.sin(np.pi * x)


def exact_B(x, t):
    """Exact solution for Problem B."""
    return np.exp(-np.pi**2 * t) * np.sin(np.pi * x)


def left_B(t):
    """Left boundary condition u(0,t)=0."""
    return 0.0


def right_B(t):
    """Right boundary condition u(1,t)=0."""
    return 0.0


def sor_tridiagonal_solver(lower, main, upper, rhs, omega=1.2, tol=1e-10, max_iter=100000):
    """
    Solve a tridiagonal linear system using SOR iteration.
    """
    lower = np.array(lower, dtype=float)
    main = np.array(main, dtype=float)
    upper = np.array(upper, dtype=float)
    rhs = np.array(rhs, dtype=float)

    n = len(rhs)
    x = np.zeros(n)

    for iteration in range(1, max_iter + 1):
        x_old = x.copy()

        for i in range(n):
            lower_sum = lower[i - 1] * x[i - 1] if i > 0 else 0.0
            upper_sum = upper[i] * x_old[i + 1] if i < n - 1 else 0.0

            gs_value = (rhs[i] - lower_sum - upper_sum) / main[i]
            x[i] = (1.0 - omega) * x_old[i] + omega * gs_value

        if np.max(np.abs(x - x_old)) < tol:
            return x, iteration

    return x, max_iter


def backward_difference_heat_sor(
    h,
    k,
    T,
    initial_func,
    left_boundary_func=None,
    right_boundary_func=None,
    x_start=0.0,
    x_end=1.0,
    omega=1.2,
    tol=1e-10,
    max_iter=100000,
):
    """
    Backward-Difference method for the heat equation.
    The implicit linear system is solved by SOR at each time step.
    """
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

    iteration_counts = []

    for n in range(Nt):
        rhs = U[n, 1:M].copy()
        rhs[0] += lam * left_boundary_func(t_grid[n + 1])
        rhs[-1] += lam * right_boundary_func(t_grid[n + 1])

        solution, iters = sor_tridiagonal_solver(
            lower=lower,
            main=main,
            upper=upper,
            rhs=rhs,
            omega=omega,
            tol=tol,
            max_iter=max_iter,
        )

        U[n + 1, 1:M] = solution
        U[n + 1, 0] = left_boundary_func(t_grid[n + 1])
        U[n + 1, M] = right_boundary_func(t_grid[n + 1])
        iteration_counts.append(iters)

    return x_grid, t_grid, U, lam, iteration_counts


def crank_nicolson_heat_sor(
    h,
    k,
    T,
    initial_func,
    left_boundary_func=None,
    right_boundary_func=None,
    x_start=0.0,
    x_end=1.0,
    omega=1.2,
    tol=1e-10,
    max_iter=100000,
):
    """
    Crank-Nicolson method for the heat equation.
    The implicit linear system is solved by SOR at each time step.
    """
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

    iteration_counts = []

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

        solution, iters = sor_tridiagonal_solver(
            lower=lower,
            main=main,
            upper=upper,
            rhs=rhs,
            omega=omega,
            tol=tol,
            max_iter=max_iter,
        )

        U[n + 1, 1:M] = solution
        U[n + 1, 0] = left_boundary_func(t_grid[n + 1])
        U[n + 1, M] = right_boundary_func(t_grid[n + 1])
        iteration_counts.append(iters)

    return x_grid, t_grid, U, lam, iteration_counts


def heat_error_table(x_grid, t_value, numerical_values, exact_func):
    """
    Create a table containing exact solution, approximation, and absolute error.
    """
    exact_values = exact_func(x_grid, t_value)
    abs_error = np.abs(exact_values - numerical_values)

    return pd.DataFrame({
        "x": x_grid,
        "t": t_value,
        "Exact Solution": exact_values,
        "Approximation": numerical_values,
        "Absolute Error": abs_error,
    })


def max_error_over_time(x_grid, t_grid, U, exact_func):
    """
    Compute maximum absolute error at each time level.
    """
    errors = []

    for n, t_value in enumerate(t_grid):
        exact_values = exact_func(x_grid, t_value)
        errors.append(np.max(np.abs(exact_values - U[n, :])))

    return np.array(errors)


def plot_problem_b_solution_comparison(x_grid, t_grid, method_results, exact_func, title):
    """
    Plot exact solution and approximation solutions at the final time.
    """
    final_time = t_grid[-1]
    exact_values = exact_func(x_grid, final_time)

    plt.figure(figsize=(8, 5))
    plt.plot(x_grid, exact_values, label="Exact Solution", linewidth=2)

    for method_name, U in method_results.items():
        plt.plot(x_grid, U[-1, :], marker="o", linestyle="--", label=method_name)

    plt.xlabel("x")
    plt.ylabel("u(x,t)")
    plt.title(title + f" at t = {final_time:.3f}")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_problem_b_error_behavior(x_grid, t_grid, method_results, exact_func, title):
    """
    Plot maximum absolute error over time for each method.
    """
    plt.figure(figsize=(8, 5))

    for method_name, U in method_results.items():
        errors = max_error_over_time(x_grid, t_grid, U, exact_func)
        plt.plot(t_grid, errors, marker="o", linestyle="--", label=method_name)

    plt.xlabel("t")
    plt.ylabel("Maximum Absolute Error")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


# ============================================================
# Problem B: stable and unstable cases
# ============================================================

T_B = 0.3
h_B_values = [0.25, 0.125, 0.0625]
lambda_stable = 0.4      # lambda <= 1/2, stable for Forward-Difference
lambda_unstable = 0.6    # lambda > 1/2, unstable for Forward-Difference
omega_heat = 1.2

cases = {
    "Stable Case": lambda_stable,
    "Unstable Forward-Difference Case": lambda_unstable,
}

for h_B in h_B_values:
    for case_name, lambda_B in cases.items():
        k_B = lambda_B * h_B**2
        print("\n" + "=" * 70)
        print(f"Problem B: {case_name}")
        print(f"h = {h_B}, k = {k_B}, lambda = {lambda_B:.4f}")
        print("=" * 70)

        x_f, t_f, U_forward, lambda_forward = forward_difference_heat(
            h=h_B,
            k=k_B,
            T=T_B,
            initial_func=initial_B,
            left_boundary_func=left_B,
            right_boundary_func=right_B,
        )

        x_b, t_b, U_backward, lambda_backward, backward_iters = backward_difference_heat_sor(
            h=h_B,
            k=k_B,
            T=T_B,
            initial_func=initial_B,
            left_boundary_func=left_B,
            right_boundary_func=right_B,
            omega=omega_heat,
            tol=1e-10,
            max_iter=100000,
        )

        x_c, t_c, U_cn, lambda_cn, cn_iters = crank_nicolson_heat_sor(
            h=h_B,
            k=k_B,
            T=T_B,
            initial_func=initial_B,
            left_boundary_func=left_B,
            right_boundary_func=right_B,
            omega=omega_heat,
            tol=1e-10,
            max_iter=100000,
        )

        method_results = {
            "Forward Difference": U_forward,
            "Backward Difference (SOR)": U_backward,
            "Crank-Nicolson (SOR)": U_cn,
        }

        print("\nForward-Difference final-time error table:")
        print(heat_error_table(x_f, t_f[-1], U_forward[-1, :], exact_B))

        print("\nBackward-Difference final-time error table:")
        print(heat_error_table(x_b, t_b[-1], U_backward[-1, :], exact_B))
        print("Backward-Difference SOR average iterations:", np.mean(backward_iters))
        print("Backward-Difference SOR max iterations:", np.max(backward_iters))

        print("\nCrank-Nicolson final-time error table:")
        print(heat_error_table(x_c, t_c[-1], U_cn[-1, :], exact_B))
        print("Crank-Nicolson SOR average iterations:", np.mean(cn_iters))
        print("Crank-Nicolson SOR max iterations:", np.max(cn_iters))

        plot_problem_b_solution_comparison(
            x_grid=x_f,
            t_grid=t_f,
            method_results=method_results,
            exact_func=exact_B,
            title=f"Problem B: Exact vs Numerical Solutions, {case_name}",
        )

        plot_problem_b_error_behavior(
            x_grid=x_f,
            t_grid=t_f,
            method_results=method_results,
            exact_func=exact_B,
            title=f"Problem B: Error Behavior Over Time, {case_name}",
        )


# ============================================================
# Problem B: effect of mesh refinement
# ============================================================
# Keep lambda = 0.4 for all mesh-refinement cases so that the
# Forward-Difference method remains stable.

h_refinement = [0.25, 0.125, 0.0625]
lambda_refinement = 0.4
T_refinement = 0.1

fd_ref_errors = []
bd_ref_errors = []
cn_ref_errors = []
bd_avg_iters = []
cn_avg_iters = []
actual_h_values = []
actual_k_values = []

for h in h_refinement:
    k = lambda_refinement * h**2
    actual_h_values.append(h)
    actual_k_values.append(k)

    x_f, t_f, U_forward, lam_f = forward_difference_heat(
        h=h,
        k=k,
        T=T_refinement,
        initial_func=initial_B,
        left_boundary_func=left_B,
        right_boundary_func=right_B,
    )

    x_b, t_b, U_backward, lam_b, backward_iters = backward_difference_heat_sor(
        h=h,
        k=k,
        T=T_refinement,
        initial_func=initial_B,
        left_boundary_func=left_B,
        right_boundary_func=right_B,
        omega=omega_heat,
        tol=1e-10,
        max_iter=100000,
    )

    x_c, t_c, U_cn, lam_c, cn_iters = crank_nicolson_heat_sor(
        h=h,
        k=k,
        T=T_refinement,
        initial_func=initial_B,
        left_boundary_func=left_B,
        right_boundary_func=right_B,
        omega=omega_heat,
        tol=1e-10,
        max_iter=100000,
    )

    fd_ref_errors.append(np.max(np.abs(exact_B(x_f, t_f[-1]) - U_forward[-1, :])))
    bd_ref_errors.append(np.max(np.abs(exact_B(x_b, t_b[-1]) - U_backward[-1, :])))
    cn_ref_errors.append(np.max(np.abs(exact_B(x_c, t_c[-1]) - U_cn[-1, :])))
    bd_avg_iters.append(np.mean(backward_iters))
    cn_avg_iters.append(np.mean(cn_iters))

refinement_table_B = pd.DataFrame({
    "h": actual_h_values,
    "k": actual_k_values,
    "lambda": [lambda_refinement] * len(actual_h_values),
    "Forward Max Error": fd_ref_errors,
    "Backward Max Error": bd_ref_errors,
    "Crank-Nicolson Max Error": cn_ref_errors,
    "Backward Avg SOR Iterations": bd_avg_iters,
    "Crank-Nicolson Avg SOR Iterations": cn_avg_iters,
})

print("\n" + "=" * 70)
print("Problem B: Mesh Refinement Table")
print("=" * 70)
print(refinement_table_B)

plt.figure(figsize=(8, 5))
plt.loglog(actual_h_values, fd_ref_errors, marker="o", linestyle="--", label="Forward Difference")
plt.loglog(actual_h_values, bd_ref_errors, marker="s", linestyle="--", label="Backward Difference (SOR)")
plt.loglog(actual_h_values, cn_ref_errors, marker="^", linestyle="--", label="Crank-Nicolson (SOR)")
plt.xlabel("h")
plt.ylabel("Maximum Absolute Error at Final Time")
plt.title("Problem B: Effect of Mesh Refinement")
plt.legend()
plt.grid(True, which="both")
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(actual_h_values, bd_avg_iters, marker="s", linestyle="--", label="Backward Difference (SOR)")
plt.plot(actual_h_values, cn_avg_iters, marker="^", linestyle="--", label="Crank-Nicolson (SOR)")
plt.xlabel("h")
plt.ylabel("Average SOR Iterations per Time Step")
plt.title("Problem B: SOR Iteration Cost Under Mesh Refinement")
plt.legend()
plt.grid(True)
plt.show()


# Iteration count vs h
plt.figure(figsize=(8, 5))
plt.plot(h_values, gs_iterations_list, marker="o", linestyle="--", label="Gauss-Seidel")
plt.plot(h_values, sor_iterations_list, marker="s", linestyle="--", label="SOR")
plt.xlabel("h")
plt.ylabel("Iteration Count")
plt.title("Problem A: Iteration Count vs Mesh Size")
plt.legend()
plt.grid(True)
plt.show()


# Bar plot: maximum errors for each h and method
x_pos = np.arange(len(N_values))
bar_width = 0.35

plt.figure(figsize=(8, 5))
plt.bar(x_pos - bar_width / 2, gs_max_errors, width=bar_width, label="Gauss-Seidel")
plt.bar(x_pos + bar_width / 2, sor_max_errors, width=bar_width, label="SOR")
plt.xticks(x_pos, [f"h=1/{N}" for N in N_values])
plt.ylabel("Maximum Absolute Error")
plt.title("Problem A: Maximum Error Comparison Under Different h")
plt.legend()
plt.grid(axis="y")
plt.show()

