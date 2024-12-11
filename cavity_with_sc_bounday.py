# Solve Maxwell eigenvalue problem for a cavity with superconducting boundary conditions.

import os
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import dolfinx

assert dolfinx.has_petsc

if PETSc.IntType == np.int64 and MPI.COMM_WORLD.size > 1:
    print(
        "This solver fails with PETSc and 64-bit integers becaude of memory errors in MUMPS."
    )
    # Note: when PETSc.IntType == np.int32, superlu_dist is used
    # rather than MUMPS and does not trigger memory failures.
    exit(0)

real_type = PETSc.RealType
scalar_type = PETSc.ScalarType

import ufl
from basix.ufl import element
from dolfinx import fem, io, plot
from dolfinx.fem.petsc import assemble_matrix
from dolfinx.mesh import (
    CellType,
    create_rectangle,
    exterior_facet_indices,
    locate_entities,
)

from slepc4py import SLEPc

try:
    import pyvista

    have_pyvista = True
except ModuleNotFoundError:
    print("pyvista and pyvistaqt are required to visualise the solution")
    have_pyvista = False

# Parameters

lx = 1.0
ly = 1.0
sc = 0.5
lambda_l = 0.03
cell_size = 0.01
fem_degree = 2
gui = True

# Mesh

nx = int((lx + 2 * sc) / cell_size)
ny = int((ly + 2 * sc) / cell_size)
print(f"Number of cells: {nx} x {ny}")

msh = create_rectangle(
    MPI.COMM_WORLD,
    np.array([[-sc, -sc], [lx + sc, ly + sc]]),
    np.array([nx, ny]),
    CellType.quadrilateral,
)
msh.topology.create_connectivity(msh.topology.dim - 1, msh.topology.dim)

print(f"msh.topology.dim: {msh.topology.dim}")

# 1/lambda^2


def at_sc_boundary(pos):
    return pos[0] <= 0 or pos[0] >= lx or pos[1] <= 0 or pos[1] >= ly


def SC(x):
    return np.array([at_sc_boundary(pos) for pos in x.T], dtype=bool)


def cavity(pos_list):
    return np.logical_not(SC(pos_list))


D = fem.functionspace(msh, ("DG", 0))
inv_lambda_sqr = fem.Function(D)

cells_scb = locate_entities(msh, msh.topology.dim, SC)
cells_cavity = locate_entities(msh, msh.topology.dim, cavity)

ils = 1 / lambda_l**2
inv_lambda_sqr.x.array[cells_scb] = np.full_like(cells_scb, ils, dtype=scalar_type)
inv_lambda_sqr.x.array[cells_cavity] = np.full_like(cells_cavity, 0, dtype=scalar_type)

# FEM

N1curl = element("N1curl", msh.basix_cell(), fem_degree, dtype=real_type)
V = fem.functionspace(msh, N1curl)

e = ufl.TrialFunction(V)
w = ufl.TestFunction(V)

a_weak = (
    ufl.inner(ufl.curl(e), ufl.curl(w)) + inv_lambda_sqr * ufl.inner(e, w)
) * ufl.dx

b_weak = (ufl.inner(e, w)) * ufl.dx

a = fem.form(a_weak)
b = fem.form(b_weak)

# Direchlet boundary conditions

bc_facets = exterior_facet_indices(msh.topology)
bc_dofs = fem.locate_dofs_topological(V, msh.topology.dim - 1, bc_facets)
u_bc = fem.Function(V)
with u_bc.x.petsc_vec.localForm() as loc:
    loc.set(0)
bc = fem.dirichletbc(u_bc, bc_dofs)

# Assemble matrices for eigenvalue problem
# Ax = lambda B x

A = assemble_matrix(a, bcs=[bc])
A.assemble()
B = assemble_matrix(b, bcs=[bc], diagonal=0.0)
B.assemble()

# Create SLEPc Eigenvalue solver
eps_target = 15.0
num_eigs = 13

eps = SLEPc.EPS().create(msh.comm)
eps.setOperators(A, B)
eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)
eps.setWhichEigenpairs(eps.Which.TARGET_MAGNITUDE)
eps.setTarget(eps_target)

# Set shift-and-invert transformation
st = eps.getST()
st.setType(SLEPc.ST.Type.SINVERT)
st.setShift(eps_target)


def monitor(eps, its, nconv, eig, err):
    print(f"Iteration: {its}, Error: {err[nconv]}")


eps.setMonitor(monitor)

eps.setDimensions(num_eigs, PETSc.DECIDE, PETSc.DECIDE)
eps.setFromOptions()
eps.setTolerances(tol=1e-10)
eps.solve()
eps.view()
eps.errorView()

its = eps.getIterationNumber()
eps_type = eps.getType()
n_ev, n_cv, mpd = eps.getDimensions()
tol, max_it = eps.getTolerances()
n_conv = eps.getConverged()

print("##################################")
print(f"Number of iterations: {its}")
print(f"Solution method: {eps_type}")
print(f"Number of requested eigenvalues: {n_ev}")
print(f"Stopping condition: tol={tol}, maxit={max_it}")
print(f"Number of converged eigenpairs: {n_conv}")
print("##################################\n")

vals = [(i, eps.getEigenvalue(i)) for i in range(eps.getConverged())]
vals.sort(key=lambda x: x[1].real)

eig_list = []
os.makedirs(f"results/lambda_{lambda_l}", exist_ok=True)
field_B = fem.Function(V)
for i, eig in vals:
    # Save eigenvector in field_B
    eps.getEigenpair(i, field_B.x.petsc_vec)

    error = eps.computeError(i, SLEPc.EPS.ErrorType.RELATIVE)
    if error >= tol:  # and np.isclose(eig.imag, 0, atol=tol):
        continue

    eig_list.append(eig)

    # TODO: Comparing with analytical modes when λ_L is zero.

    eig_str = f"#{i} Eigenvalue: {eig.real}\t L√E/π: {np.sqrt(eig.real)/np.pi}"
    print(eig_str)
    with open(f"results/lambda_{lambda_l}/eigs.txt", "a") as f:
        f.write(eig_str + "\n")

    field_B.x.scatter_forward()

    gdim = msh.geometry.dim
    V_dg = fem.functionspace(msh, ("DG", fem_degree, (gdim,)))
    B_dg = fem.Function(V_dg)

    B_dg.interpolate(field_B)

    # Save solutions
    with io.VTXWriter(msh.comm, f"results/lambda_{lambda_l}/sols/B_{i}.bp", B_dg) as f:
        f.write(0.0)

    # Visualize solutions with Pyvista
    if have_pyvista:
        V_cells, V_types, V_x = plot.vtk_mesh(V_dg)
        V_grid = pyvista.UnstructuredGrid(V_cells, V_types, V_x)

        B_vectors = B_dg.x.array.reshape(V_x.shape[0], msh.topology.dim).real
        B_values = np.linalg.norm(B_vectors, axis=1)

        V_grid.point_data["b"] = B_values
        # B_max = np.max(B_values)
        V_grid.set_active_scalars("b")
        warped = V_grid.warp_by_scalar("b", factor=0.3)

        plotter = pyvista.Plotter(off_screen=not gui)
        plotter.add_mesh(warped, show_edges=False, cmap="twilight")
        # plotter.add_mesh(V_grid.copy(), show_edges=False)
        plotter.add_text(
            f"L√E/π: {np.sqrt(eig.real)/np.pi}",
            position="upper_left",
        )

        # plotter.view_xy()
        # plotter.link_views()

        plotter.save_graphic(
            f"results/lambda_{lambda_l}/eig_{np.sqrt(eig.real)/np.pi:.3f}_i{i}.svg"
        )

        if not pyvista.OFF_SCREEN:
            plotter.show()

eps.destroy()
