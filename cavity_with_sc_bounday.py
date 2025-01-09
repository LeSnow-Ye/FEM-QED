# Solve Maxwell eigenvalue problem for a cavity with superconducting boundary conditions.

import os
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import dolfinx
import argparse

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
from dolfinx.io import gmshio
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


def solve(
    mesh,
    inv_lambda_sqr: fem.Function,
    fem_degree=2,
    gui=False,
    eps_target=15.0,
    num_eigs=13,
    tol=1e-10,
    output_path="results",
):
    print(f"Assembling matrices...")

    N1curl = element("N1curl", mesh.basix_cell(), fem_degree, dtype=real_type)
    V = fem.functionspace(mesh, N1curl)

    b_func = ufl.TrialFunction(V)
    w_func = ufl.TestFunction(V)

    a_weak = (
        ufl.inner(ufl.curl(b_func), ufl.curl(w_func))
        + inv_lambda_sqr * ufl.inner(b_func, w_func)
    ) * ufl.dx

    b_weak = (ufl.inner(b_func, w_func)) * ufl.dx

    a = fem.form(a_weak)
    b = fem.form(b_weak)

    # Direchlet boundary conditions

    bc_facets = exterior_facet_indices(mesh.topology)
    bc_dofs = fem.locate_dofs_topological(V, mesh.topology.dim - 1, bc_facets)
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

    # TODO: Check parameters for SLEPc of Palace:
    # Configuring SLEPc eigenvalue solver:
    # Scaling γ = 3.036e+04, δ = 5.892e-05
    # Configuring divergence-free projection
    # Using random starting vector
    # Shift-and-invert σ = 5.000e+00 GHz (3.144e-01)

    print(f"Solving eigenvalues...")
    eps = SLEPc.EPS().create(mesh.comm)
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
    eps.setTolerances(tol=tol)
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
    field_B = fem.Function(V)

    if os.path.exists(f"{output_path}/lambda_{lambda_l}"):
        print(
            f"Warning:Folder {output_path}/lambda_{lambda_l} already exists. Removing it..."
        )
        for root, dirs, files in os.walk(
            f"{output_path}/lambda_{lambda_l}", topdown=False
        ):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
    else:
        os.makedirs(f"{output_path}/lambda_{lambda_l}", exist_ok=True)

    for i, eig in vals:
        # Save eigenvector in field_B
        eps.getEigenpair(i, field_B.x.petsc_vec)

        error = eps.computeError(i, SLEPc.EPS.ErrorType.RELATIVE)
        if error >= tol:  # and np.isclose(eig.imag, 0, atol=tol):
            print(
                f"Warning: The error {error} of eigenvalue {eig.real} is greater than tolerance {tol}. Neglecting it."
            )

            continue

        eig_list.append(eig)

        # TODO: Comparing with analytical modes when λ_L is zero.

        eig_str = f"#{i} Eigenvalue: {eig.real}\t L√E/π: {np.sqrt(eig.real)/np.pi}"
        print(eig_str)
        with open(f"{output_path}/lambda_{lambda_l}/eigs.txt", "a") as f:
            f.write(eig_str + "\n")

        field_B.x.scatter_forward()

        gdim = mesh.geometry.dim
        V_dg = fem.functionspace(mesh, ("DG", fem_degree, (gdim,)))
        B_dg = fem.Function(V_dg)

        B_dg.interpolate(field_B)

        # Save solutions
        with io.VTXWriter(
            mesh.comm, f"{output_path}/lambda_{lambda_l}/sols/B_{i}.bp", B_dg
        ) as f:
            f.write(0.0)

        # Visualize solutions with Pyvista
        if have_pyvista:
            V_cells, V_types, V_x = plot.vtk_mesh(V_dg)
            V_grid = pyvista.UnstructuredGrid(V_cells, V_types, V_x)

            B_vectors = B_dg.x.array.reshape(V_x.shape[0], mesh.topology.dim).real
            B_values = np.linalg.norm(B_vectors, axis=1)

            # V_grid.point_data["b"] = B_vectors[:, 0]
            V_grid.point_data["b"] = B_values
            # B_max = np.max(B_values)
            V_grid.set_active_scalars("b")
            warped = V_grid.warp_by_scalar("b", factor=0.3)

            plotter = pyvista.Plotter(off_screen=not gui)
            plotter.add_mesh(warped, show_edges=False, cmap="twilight")
            # plotter.add_mesh(V_grid.copy(), show_edges=False, cmap="twilight")
            plotter.add_text(
                f"L√E/π: {np.sqrt(eig.real)/np.pi}",
                position="upper_left",
            )

            # plotter.view_xy()
            # plotter.link_views()

            plotter.save_graphic(
                f"{output_path}/lambda_{lambda_l}/eig_{np.sqrt(eig.real)/np.pi:.3f}_i{i}.svg"
            )

            if not pyvista.OFF_SCREEN:
                plotter.show()

    eps.destroy()


def load_gmsh_mesh(mesh_path):
    # Using Gmsh.
    Cavity = 1
    SC_boundary = 2

    mesh, cell_markers, facet_markers = gmshio.read_from_msh(
        mesh_path, MPI.COMM_WORLD, gdim=2
    )
    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)

    D = fem.functionspace(mesh, ("DG", 0))
    num_dofs = D.dofmap.index_map.size_local + D.dofmap.index_map.num_ghosts
    print(f"Number of DOFs: {num_dofs}")

    inv_lambda_sqr = fem.Function(D)
    ils = 1 / lambda_l**2

    material_tags = np.unique(cell_markers.values)
    print(f"Material tags: {material_tags}")

    for tag in material_tags:
        cells = cell_markers.find(tag)
        # Set values for inv lambda^2
        if tag == SC_boundary:
            inv_lambda_sqr.x.array[cells] = np.full_like(cells, ils, dtype=scalar_type)
        elif tag == Cavity:
            inv_lambda_sqr.x.array[cells] = np.full_like(cells, 0, dtype=scalar_type)

    return mesh, inv_lambda_sqr


def generate_mesh():
    # Using FEniCS buid-in mesh generator.
    lx = 1.0
    ly = 1.0
    sc = 0.5
    cell_size = 0.01

    # Mesh

    nx = int((lx + 2 * sc) / cell_size)
    ny = int((ly + 2 * sc) / cell_size)
    print(f"Number of cells: {nx} x {ny}")

    mesh = create_rectangle(
        MPI.COMM_WORLD,
        np.array([[-sc, -sc], [lx + sc, ly + sc]]),
        np.array([nx, ny]),
        CellType.quadrilateral,
    )
    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)

    print(f"msh.topology.dim: {mesh.topology.dim}")

    # 1/lambda^2

    def at_sc_boundary(pos):
        return pos[0] <= 0 or pos[0] >= lx or pos[1] <= 0 or pos[1] >= ly

    def sc(x):
        return np.array([at_sc_boundary(pos) for pos in x.T], dtype=bool)

    def cavity(pos_list):
        return np.logical_not(sc(pos_list))

    D = fem.functionspace(mesh, ("DG", 0))
    num_dofs = D.dofmap.index_map.size_local + D.dofmap.index_map.num_ghosts
    print(f"Number of DOFs: {num_dofs}")

    inv_lambda_sqr = fem.Function(D)

    cells_scb = locate_entities(mesh, mesh.topology.dim, sc)
    cells_cavity = locate_entities(mesh, mesh.topology.dim, cavity)

    ils = 1 / lambda_l**2
    inv_lambda_sqr.x.array[cells_scb] = np.full_like(cells_scb, ils, dtype=scalar_type)
    inv_lambda_sqr.x.array[cells_cavity] = np.full_like(
        cells_cavity, 0, dtype=scalar_type
    )

    return mesh, inv_lambda_sqr


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Solve Maxwell eigenvalue problem for a cavity with superconducting boundary conditions."
    )
    parser.add_argument(
        "-l", "--lambda_l", type=float, default=0.01, help="Lambda parameter"
    )
    parser.add_argument(
        "-msh",
        "--mesh_path",
        type=str,
        help="GMSH format mesh file path. If not set, use built-in mesh generator.",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        default="results",
        help="Output path for results",
    )
    parser.add_argument(
        "-n",
        "--num_eigs",
        type=int,
        default=13,
        help="Number of eigenvalues to compute",
    )
    parser.add_argument("--gui", action="store_true", help="Enable GUI")
    parser.add_argument(
        "--eps_target", type=float, default=15.0, help="Target eigenvalue for SLEPc"
    )
    parser.add_argument(
        "--tol", type=float, default=1e-10, help="Tolerance for eigenvalue solver"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    lambda_l = args.lambda_l
    gui = args.gui
    eps_target = args.eps_target
    num_eigs = args.num_eigs
    tol = args.tol
    output_path = args.output_path
    mesh_path = args.mesh_path

    if mesh_path:
        mesh, inv_lambda_sqr = load_gmsh_mesh(mesh_path)
    else:
        mesh, inv_lambda_sqr = generate_mesh()

    solve(
        mesh,
        inv_lambda_sqr,
        fem_degree=3,
        gui=gui,
        eps_target=eps_target,
        num_eigs=num_eigs,
        tol=tol,
        output_path=output_path,
    )
