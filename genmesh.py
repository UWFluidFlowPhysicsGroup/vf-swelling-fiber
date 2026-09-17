import sys
import gmsh

gmsh.initialize(sys.argv)

MSH_VER = 2.0
SIZE_FACTOR = 0.2
OUTPUT_DIR = 'mesh/'

def proc_BC():
    """
    Generate a mesh for BC_Half.STEP geometry
    """
    gmsh.clear()
    gmsh.model.add('main')

    gmsh.option.set_string('Geometry.OCCTargetUnit', 'CM')
    gmsh.merge(f'BC_Half.STEP')

    gmsh.model.add_physical_group(2, [2], name='body')
    gmsh.model.add_physical_group(2, [1], name='cover')
    
    gmsh.model.add_physical_group(1, [8, 9, 10, 11, 12], name='pressure')
    gmsh.model.add_physical_group(1, [13, 7, 1], name='fixed')

    gmsh.model.add_physical_group(0, [10], name='separation-inf')
    gmsh.model.add_physical_group(0, [9], name='separation-sup')

    gmsh.option.set_number('Mesh.MshFileVersion', MSH_VER)
    gmsh.option.set_number('Mesh.MeshSizeFactor', SIZE_FACTOR)

    gmsh.model.mesh.generate(2)
    gmsh.write(OUTPUT_DIR + f'BC_Half.msh')

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

    gmsh.model.add_physical_group(0, [1], name='separation-inf')
    gmsh.model.add_physical_group(0, [14], name='separation-sup')

    gmsh.option.set_number('Mesh.MshFileVersion', MSH_VER)
    gmsh.option.set_number('Mesh.MeshSizeFactor', SIZE_FACTOR)

    gmsh.model.mesh.generate(2)
    gmsh.write(OUTPUT_DIR + f'BCS--SR0.04--DZ0.00--NZ1--clscale2.50e-01')

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

    gmsh.model.add_physical_group(0, [1], name='separation-inf')
    gmsh.model.add_physical_group(0, [14], name='separation-sup')

    gmsh.option.set_number('Mesh.MshFileVersion', MSH_VER)
    gmsh.option.set_number('Mesh.MeshSizeFactor', SIZE_FACTOR)

    gmsh.model.mesh.generate(2)
    gmsh.write(OUTPUT_DIR + f'BCS--SR0.08--DZ0.00--NZ1--clscale2.50e-01')

def proc_BCS_12():
    """
    Generate a mesh for scar tissue geometries
    """
    gmsh.clear()
    gmsh.model.add('main')

    gmsh.option.set_string('Geometry.OCCTargetUnit', 'CM')
    gmsh.merge(f'BCS_0.12.STEP')

    gmsh.model.add_physical_group(2, [1], name='body')
    gmsh.model.add_physical_group(2, [2, 3], name='cover')
    gmsh.model.add_physical_group(2, [4, 5], name='scar')
    
    gmsh.model.add_physical_group(1, [17, 18, 14, 20, 10, 11, 12], name='pressure')
    gmsh.model.add_physical_group(1, [13, 6, 16], name='fixed')


    gmsh.model.add_physical_group(0, [13], name='separation-inf')
    gmsh.model.add_physical_group(0, [16], name='separation-sup')

    gmsh.option.set_number('Mesh.MshFileVersion', MSH_VER)
    gmsh.option.set_number('Mesh.MeshSizeFactor', SIZE_FACTOR)

    gmsh.model.mesh.generate(2)
    gmsh.write(OUTPUT_DIR + f'BCS--SR0.12--DZ0.00--NZ1--clscale2.50e-01')

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
    gmsh.write(OUTPUT_DIR + f'BCS--SR0.16--DZ0.00--NZ1--clscale2.50e-01')

if __name__ == '__main__':
    proc_BC()
    proc_BCS_04()
    proc_BCS_08()
    proc_BCS_12()
    proc_BCS_16()
