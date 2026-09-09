import sys
import gmsh

gmsh.initialize(sys.argv)

MSH_VER = 2.0
SIZE_FACTOR = 0.2


def proc_BC():
    """
    Generate a mesh for BC_Half.STEP geometry
    """
    gmsh.clear()
    gmsh.model.add('main')

    gmsh.option.set_string('Geometry.OCCTargetUnit', 'MM')
    gmsh.merge(f'BC_Half.STEP')

    gmsh.model.add_physical_group(2, [1], name='body')
    gmsh.model.add_physical_group(2, [2], name='cover')
    
    gmsh.model.add_physical_group(1, [8, 9, 10, 11, 12], name='pressure')
    gmsh.model.add_physical_group(1, [13, 7, 1], name='fixed')

    gmsh.model.add_physical_group(0, [10], name='separation-inf')
    gmsh.model.add_physical_group(0, [9], name='separation-sup')

    gmsh.option.set_number('Mesh.MshFileVersion', MSH_VER)
    gmsh.option.set_number('Mesh.MeshSizeFactor', SIZE_FACTOR)

    gmsh.model.mesh.generate(2)
    gmsh.write(f'BC_Half.msh')

def proc_BCS():
    """
    Generate a mesh for scar tissue geometries
    """
    gmsh.clear()
    gmsh.model.add('main')

    gmsh.option.set_string('Geometry.OCCTargetUnit', 'MM')
    gmsh.merge(f'BCS_0.16_0.08.STEP')

    gmsh.model.add_physical_group(2, [1], name='body')
    gmsh.model.add_physical_group(2, [2, 3], name='cover')
    gmsh.model.add_physical_group(2, [4, 5], name='scar')
    
    gmsh.model.add_physical_group(1, [9, 13, 12, 15, 16, 17, 19], name='pressure')
    gmsh.model.add_physical_group(1, [11, 6, 18], name='fixed')

    gmsh.model.add_physical_group(0, [9], name='separation-inf')
    gmsh.model.add_physical_group(0, [12], name='separation-sup')

    gmsh.option.set_number('Mesh.MshFileVersion', MSH_VER)
    gmsh.option.set_number('Mesh.MeshSizeFactor', SIZE_FACTOR)

    gmsh.model.mesh.generate(2)
    gmsh.write(f'BCS_0.16_0.08.msh')

if __name__ == '__main__':
    proc_BC()
    proc_BCS()
