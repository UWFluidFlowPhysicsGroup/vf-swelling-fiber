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

    gmsh.option.set_string('Geometry.OCCTargetUnit', 'CM')
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

def proc_BCS_04():
    """
    Generate a mesh for scar tissue geometries
    """
    gmsh.clear()
    gmsh.model.add('main')

    gmsh.option.set_string('Geometry.OCCTargetUnit', 'CM')
    gmsh.merge(f'BCS_0.04.STEP')

    gmsh.model.add_physical_group(2, [3], name='body')
    gmsh.model.add_physical_group(2, [1], name='cover')
    gmsh.model.add_physical_group(2, [2], name='scar')
    
    gmsh.model.add_physical_group(1, [5, 4, 3, 15, 1, 14, 13], name='pressure')
    gmsh.model.add_physical_group(1, [6, 16, 12], name='fixed')

    gmsh.model.add_physical_group(0, [14], name='separation-inf')
    gmsh.model.add_physical_group(0, [1], name='separation-sup')

    gmsh.option.set_number('Mesh.MshFileVersion', MSH_VER)
    gmsh.option.set_number('Mesh.MeshSizeFactor', SIZE_FACTOR)

    gmsh.model.mesh.generate(2)
    gmsh.write(f'BCS_0.04.msh')

def proc_BCS_08():
    """
    Generate a mesh for scar tissue geometries
    """
    gmsh.clear()
    gmsh.model.add('main')

    gmsh.option.set_string('Geometry.OCCTargetUnit', 'CM')
    gmsh.merge(f'BCS_0.08.STEP')

    gmsh.model.add_physical_group(2, [3], name='body')
    gmsh.model.add_physical_group(2, [1], name='cover')
    gmsh.model.add_physical_group(2, [2], name='scar')
    
    gmsh.model.add_physical_group(1, [5, 4, 3, 15, 1, 14, 13], name='pressure')
    gmsh.model.add_physical_group(1, [6, 16, 12], name='fixed')

    gmsh.model.add_physical_group(0, [14], name='separation-inf')
    gmsh.model.add_physical_group(0, [1], name='separation-sup')

    gmsh.option.set_number('Mesh.MshFileVersion', MSH_VER)
    gmsh.option.set_number('Mesh.MeshSizeFactor', SIZE_FACTOR)

    gmsh.model.mesh.generate(2)
    gmsh.write(f'BCS_0.08.msh')

def proc_BCS_12():
    """
    Generate a mesh for scar tissue geometries
    """
    gmsh.clear()
    gmsh.model.add('main')

    gmsh.option.set_string('Geometry.OCCTargetUnit', 'CM')
    gmsh.merge(f'BCS_0.12.STEP')

    gmsh.model.add_physical_group(2, [4], name='body')
    gmsh.model.add_physical_group(2, [1, 2], name='cover')
    gmsh.model.add_physical_group(2, [3], name='scar')
    
    gmsh.model.add_physical_group(1, [13, 12, 11, 17, 1, 8, 7], name='pressure')
    gmsh.model.add_physical_group(1, [14, 18, 6], name='fixed')

    gmsh.model.add_physical_group(0, [8], name='separation-inf')
    gmsh.model.add_physical_group(0, [1], name='separation-sup')

    gmsh.option.set_number('Mesh.MshFileVersion', MSH_VER)
    gmsh.option.set_number('Mesh.MeshSizeFactor', SIZE_FACTOR)

    gmsh.model.mesh.generate(2)
    gmsh.write(f'BCS_0.12.msh')

def proc_BCS_16():
    """
    Generate a mesh for scar tissue geometries
    """
    gmsh.clear()
    gmsh.model.add('main')

    gmsh.option.set_string('Geometry.OCCTargetUnit', 'CM')
    gmsh.merge(f'BCS_0.16.STEP')

    gmsh.model.add_physical_group(2, [1], name='body')
    gmsh.model.add_physical_group(2, [2, 3], name='cover')
    gmsh.model.add_physical_group(2, [4, 5], name='scar')
    
    gmsh.model.add_physical_group(1, [17, 16, 15, 19, 9, 13, 12], name='pressure')
    gmsh.model.add_physical_group(1, [11, 6, 18], name='fixed')

    gmsh.model.add_physical_group(0, [9], name='separation-inf')
    gmsh.model.add_physical_group(0, [12], name='separation-sup')

    gmsh.option.set_number('Mesh.MshFileVersion', MSH_VER)
    gmsh.option.set_number('Mesh.MeshSizeFactor', SIZE_FACTOR)

    gmsh.model.mesh.generate(2)
    gmsh.write(f'BCS_0.16.msh')

if __name__ == '__main__':
    # proc_BC()
    proc_BCS_04()
    proc_BCS_08()
    proc_BCS_12()
    proc_BCS_16()
