import gmsh
import os


def generate_square_mesh(
    lx, ly, size_min, size_max, sc_boundary=0.0, filename="square.msh", gui=False
):
    gmsh.initialize()
    gmsh.clear()
    gmsh.model.add("square")

    rect = gmsh.model.occ.addRectangle(-lx / 2.0, -ly / 2.0, 0.0, lx, ly)

    sc_dimtags = []
    if sc_boundary > 0.0:
        sc_raw = gmsh.model.occ.addRectangle(
            -lx / 2.0 - sc_boundary,
            -ly / 2.0 - sc_boundary,
            0.0,
            lx + 2 * sc_boundary,
            ly + 2 * sc_boundary,
        )
        sc_dimtags, _ = gmsh.model.occ.cut([(2, sc_raw)], [(2, rect)], -1, True, False)

    gmsh.model.occ.synchronize()

    # Physical groups
    gmsh.model.addPhysicalGroup(2, [rect], -1, "Cavity")
    gmsh.model.addPhysicalGroup(
        2, [x[1] for x in sc_dimtags if x[0] == 2], -1, "SC_boundary"
    )

    # Set mesh size
    points = [
        x[1]
        for x in gmsh.model.getBoundary([(2, rect)], False, True, True)
        if x[0] == 0
    ]
    curves = [
        x[1]
        for x in gmsh.model.getBoundary([(2, rect)], False, False, False)
        if x[0] == 1
    ]

    print("Points:", points)
    print("Curves:", curves)

    gmsh.option.setNumber("Mesh.MeshSizeMin", size_min)
    gmsh.option.setNumber("Mesh.MeshSizeMax", size_max)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)

    gmsh.model.mesh.field.add("Distance", 1)
    gmsh.model.mesh.field.setNumbers(1, "PointsList", points)
    gmsh.model.mesh.field.setNumbers(1, "CurvesList", curves)
    # gmsh.model.mesh.field.setNumbers(1, "SurfacesList", gmsh.model.getEntities(2))
    gmsh.model.mesh.field.setNumber(1, "Sampling", max(lx, ly) / size_min)

    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "InField", 1)
    gmsh.model.mesh.field.setNumber(2, "SizeMin", size_min)
    gmsh.model.mesh.field.setNumber(2, "SizeMax", size_max)
    gmsh.model.mesh.field.setNumber(2, "DistMin", 3 * size_min)
    gmsh.model.mesh.field.setNumber(2, "DistMax", 0.2 * min(lx, ly))

    gmsh.model.mesh.field.add("Min", 101)
    gmsh.model.mesh.field.setNumbers(101, "FieldsList", [2])
    gmsh.model.mesh.field.setAsBackgroundMesh(101)

    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.model.mesh.generate(2)

    # Save mesh
    if not os.path.exists("mesh"):
        os.makedirs("mesh")

    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.option.setNumber("Mesh.Binary", 0)
    gmsh.write(os.path.join("mesh", filename))

    # Optionally launch GUI
    if gui:
        gmsh.fltk.run()

    gmsh.finalize()


if __name__ == "__main__":
    # generate_square_mesh(1.0, 1.0, 10, 10)
    generate_square_mesh(1.0, 1.0, 0.005, 0.05, 0.5, "square_sc_boundary.msh", True)
