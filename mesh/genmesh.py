import sys
import gmsh

gmsh.initialize(sys.argv)

MSH_VER = 2.0


def proc_M5():
    """
    Generate a mesh from .STEP geometry
    """
    gmsh.clear()
    gmsh.model.add('main')

    gmsh.option.set_string('Geometry.OCCTargetUnit', 'CM')

    # change merge to .STEP file name with extension
    gmsh.merge(f'M5_BC.STEP')

    """
    Set physical surfaces, curves and points required for simulation
    0 = Physical Point
    1 = Physical Curve
    2 = Physical Surface
    3 = Physical Volume

    Numbers in square brackets are geometry ids, can be found by loading .STEP file in gmsh
    """
    gmsh.model.add_physical_group(2, [2], name='body')
    gmsh.model.add_physical_group(2, [1], name='cover')

    gmsh.model.add_physical_group(1, [11, 10, 9, 8, 12], name='pressure')
    gmsh.model.add_physical_group(1, [13, 7, 1], name='fixed')

    gmsh.model.add_physical_group(0, [10], name='separation-inf')
    gmsh.model.add_physical_group(0, [9], name='separation-sup')

    # Set msh to save as version 2.0, which is required to not save unnecessary points
    gmsh.option.set_number('Mesh.MshFileVersion', MSH_VER)

    # generate and save mesh (2D)
    gmsh.model.mesh.generate(2)
    gmsh.write(f'M5_BC.msh')

# Create and insert other generated meshes in main section, can also iterate through different parameters that can be passed to proc_M5()
if __name__ == '__main__':
    proc_M5()
        