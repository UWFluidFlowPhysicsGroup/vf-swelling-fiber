import sys
import gmsh

gmsh.initialize(sys.argv)

MSH_VER = 2.0
SIZE_FACTOR = 0.125
OUTPUT_DIR = 'mesh/'

# meshes need to be hardcoded since different scar geometries lead to different numbering of physical groups
# Ex. the surfaces that define the body/cover/scar change numbers as the scar radius crosses the BC boundary
# TODO create mesh objects that store the arrays of physical group values, which could condense code while still being readable
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

# Generate medial scar with radius of 0.04 cm
def proc_BCS_M_04():
    """
    Generate a mesh for scar tissue geometries
    """
    gmsh.clear()
    gmsh.model.add('main')

    gmsh.option.set_string('Geometry.OCCTargetUnit', 'CM')
    gmsh.merge(f'BCS_M_0.04.STEP')

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
    gmsh.write(OUTPUT_DIR + f'BCS_M--SD0.04--DZ0.00--NZ1--clscale2.50e-01.msh')

# Generate medial scar with radius of 0.08 cm
def proc_BCS_M_08():
    """
    Generate a mesh for scar tissue geometries
    """
    gmsh.clear()
    gmsh.model.add('main')

    gmsh.option.set_string('Geometry.OCCTargetUnit', 'CM')
    gmsh.merge(f'BCS_M_0.08.STEP')

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
    gmsh.write(OUTPUT_DIR + f'BCS_M--SD0.08--DZ0.00--NZ1--clscale2.50e-01.msh')

# Generate medial scar with radius of 0.12 cm
def proc_BCS_M_12():
    """
    Generate a mesh for scar tissue geometries
    """
    gmsh.clear()
    gmsh.model.add('main')

    gmsh.option.set_string('Geometry.OCCTargetUnit', 'CM')
    gmsh.merge(f'BCS_M_0.12.STEP')

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
    gmsh.write(OUTPUT_DIR + f'BCS_M--SD0.12--DZ0.00--NZ1--clscale2.50e-01.msh')

# Generate medial scar with radius of 0.16 cm
def proc_BCS_M_16():
    """
    Generate a mesh for scar tissue geometries
    """
    gmsh.clear()
    gmsh.model.add('main')

    gmsh.option.set_string('Geometry.OCCTargetUnit', 'CM')
    gmsh.merge(f'BCS_M_0.16.STEP')

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
    gmsh.write(OUTPUT_DIR + f'BCS_M--SD0.16--DZ0.00--NZ1--clscale2.50e-01.msh')

# Generate superior scar with radius of 0.04 cm
def proc_BCS_S_04():
    """
    Generate a mesh for scar tissue geometries
    """
    gmsh.clear()
    gmsh.model.add('main')

    gmsh.option.set_string('Geometry.OCCTargetUnit', 'CM')
    gmsh.merge(f'BCS_S_0.04.STEP')

    gmsh.model.add_physical_group(2, [3], name='body')
    gmsh.model.add_physical_group(2, [1], name='cover')
    gmsh.model.add_physical_group(2, [2], name='scar')
    
    gmsh.model.add_physical_group(1, [5, 4, 3, 15, 14, 1, 13], name='pressure')
    gmsh.model.add_physical_group(1, [6, 16, 12], name='fixed')

    gmsh.model.add_physical_group(0, [1], name='separation-inf')
    gmsh.model.add_physical_group(0, [14], name='separation-sup')

    gmsh.option.set_number('Mesh.MshFileVersion', MSH_VER)
    gmsh.option.set_number('Mesh.MeshSizeFactor', SIZE_FACTOR)

    gmsh.model.mesh.generate(2)
    gmsh.write(OUTPUT_DIR + f'BCS_S--SD0.04--DZ0.00--NZ1--clscale2.50e-01.msh')

# Generate superior scar with radius of 0.08 cm
def proc_BCS_S_08():
    """
    Generate a mesh for scar tissue geometries
    """
    gmsh.clear()
    gmsh.model.add('main')

    gmsh.option.set_string('Geometry.OCCTargetUnit', 'CM')
    gmsh.merge(f'BCS_S_0.08.STEP')

    gmsh.model.add_physical_group(2, [3], name='body')
    gmsh.model.add_physical_group(2, [1], name='cover')
    gmsh.model.add_physical_group(2, [2], name='scar')
    
    gmsh.model.add_physical_group(1, [5, 4, 3, 15, 14, 1, 13], name='pressure')
    gmsh.model.add_physical_group(1, [6, 16, 12], name='fixed')

    gmsh.model.add_physical_group(0, [1], name='separation-inf')
    gmsh.model.add_physical_group(0, [14], name='separation-sup')

    gmsh.option.set_number('Mesh.MshFileVersion', MSH_VER)
    gmsh.option.set_number('Mesh.MeshSizeFactor', SIZE_FACTOR)

    gmsh.model.mesh.generate(2)
    gmsh.write(OUTPUT_DIR + f'BCS_S--SD0.08--DZ0.00--NZ1--clscale2.50e-01.msh')

# Generate superior scar with radius of 0.12 cm
def proc_BCS_S_12():
    """
    Generate a mesh for scar tissue geometries
    """
    gmsh.clear()
    gmsh.model.add('main')

    gmsh.option.set_string('Geometry.OCCTargetUnit', 'CM')
    gmsh.merge(f'BCS_S_0.12.STEP')

    gmsh.model.add_physical_group(2, [1], name='body')
    gmsh.model.add_physical_group(2, [2, 3], name='cover')
    gmsh.model.add_physical_group(2, [4, 5], name='scar')
    
    gmsh.model.add_physical_group(1, [15, 14, 13, 18, 17, 8, 11], name='pressure')
    gmsh.model.add_physical_group(1, [16, 6, 10], name='fixed')

    gmsh.model.add_physical_group(0, [15], name='separation-inf')
    gmsh.model.add_physical_group(0, [8], name='separation-sup')

    gmsh.option.set_number('Mesh.MshFileVersion', MSH_VER)
    gmsh.option.set_number('Mesh.MeshSizeFactor', SIZE_FACTOR)

    gmsh.model.mesh.generate(2)
    gmsh.write(OUTPUT_DIR + f'BCS_S--SD0.12--DZ0.00--NZ1--clscale2.50e-01.msh')

# Generate superior scar with radius of 0.16 cm
def proc_BCS_S_16():
    """
    Generate a mesh for scar tissue geometries
    """
    gmsh.clear()
    gmsh.model.add('main')

    gmsh.option.set_string('Geometry.OCCTargetUnit', 'CM')
    gmsh.merge(f'BCS_S_0.16.STEP')

    gmsh.model.add_physical_group(2, [1], name='body')
    gmsh.model.add_physical_group(2, [2, 3], name='cover')
    gmsh.model.add_physical_group(2, [4, 5], name='scar')
    
    gmsh.model.add_physical_group(1, [15, 14, 13, 18, 17, 8, 11], name='pressure')
    gmsh.model.add_physical_group(1, [16, 6, 10], name='fixed')

    gmsh.model.add_physical_group(0, [15], name='separation-inf')
    gmsh.model.add_physical_group(0, [8], name='separation-sup')

    gmsh.option.set_number('Mesh.MshFileVersion', MSH_VER)
    gmsh.option.set_number('Mesh.MeshSizeFactor', SIZE_FACTOR)

    gmsh.model.mesh.generate(2)
    gmsh.write(OUTPUT_DIR + f'BCS_S--SD0.16--DZ0.00--NZ1--clscale2.50e-01.msh')

# Generate inferior scar with radius of 0.04 cm
def proc_BCS_I_04():
    """
    Generate a mesh for scar tissue geometries
    """
    gmsh.clear()
    gmsh.model.add('main')

    gmsh.option.set_string('Geometry.OCCTargetUnit', 'CM')
    gmsh.merge(f'BCS_I_0.04.STEP')

    gmsh.model.add_physical_group(2, [3], name='body')
    gmsh.model.add_physical_group(2, [1], name='cover')
    gmsh.model.add_physical_group(2, [2], name='scar')
    
    gmsh.model.add_physical_group(1, [4, 3, 15, 14, 1, 13, 12], name='pressure')
    gmsh.model.add_physical_group(1, [5, 16, 11], name='fixed')

    gmsh.model.add_physical_group(0, [1], name='separation-inf')
    gmsh.model.add_physical_group(0, [13], name='separation-sup')

    gmsh.option.set_number('Mesh.MshFileVersion', MSH_VER)
    gmsh.option.set_number('Mesh.MeshSizeFactor', SIZE_FACTOR)

    gmsh.model.mesh.generate(2)
    gmsh.write(OUTPUT_DIR + f'BCS_I--SD0.04--DZ0.00--NZ1--clscale2.50e-01.msh')

# Generate inferior scar with radius of 0.08 cm
def proc_BCS_I_08():
    """
    Generate a mesh for scar tissue geometries
    """
    gmsh.clear()
    gmsh.model.add('main')

    gmsh.option.set_string('Geometry.OCCTargetUnit', 'CM')
    gmsh.merge(f'BCS_I_0.08.STEP')

    gmsh.model.add_physical_group(2, [3], name='body')
    gmsh.model.add_physical_group(2, [1], name='cover')
    gmsh.model.add_physical_group(2, [2], name='scar')
    
    gmsh.model.add_physical_group(1, [4, 3, 15, 14, 1, 13, 12], name='pressure')
    gmsh.model.add_physical_group(1, [5, 16, 11], name='fixed')

    gmsh.model.add_physical_group(0, [1], name='separation-inf')
    gmsh.model.add_physical_group(0, [13], name='separation-sup')

    gmsh.option.set_number('Mesh.MshFileVersion', MSH_VER)
    gmsh.option.set_number('Mesh.MeshSizeFactor', SIZE_FACTOR)

    gmsh.model.mesh.generate(2)
    gmsh.write(OUTPUT_DIR + f'BCS_I--SD0.08--DZ0.00--NZ1--clscale2.50e-01.msh')

# Generate inferior scar with radius of 0.12 cm
def proc_BCS_I_12():
    """
    Generate a mesh for scar tissue geometries
    """
    gmsh.clear()
    gmsh.model.add('main')

    gmsh.option.set_string('Geometry.OCCTargetUnit', 'CM')
    gmsh.merge(f'BCS_I_0.12.STEP')

    gmsh.model.add_physical_group(2, [1], name='body')
    gmsh.model.add_physical_group(2, [2, 3], name='cover')
    gmsh.model.add_physical_group(2, [4, 5], name='scar')
    
    gmsh.model.add_physical_group(1, [10, 9, 20, 19, 12, 16, 15], name='pressure')
    gmsh.model.add_physical_group(1, [11, 5, 14], name='fixed')

    gmsh.model.add_physical_group(0, [11], name='separation-inf')
    gmsh.model.add_physical_group(0, [14], name='separation-sup')

    gmsh.option.set_number('Mesh.MshFileVersion', MSH_VER)
    gmsh.option.set_number('Mesh.MeshSizeFactor', SIZE_FACTOR)

    gmsh.model.mesh.generate(2)
    gmsh.write(OUTPUT_DIR + f'BCS_I--SD0.12--DZ0.00--NZ1--clscale2.50e-01.msh')

# Generate inferior scar with radius of 0.16 cm
def proc_BCS_I_16():
    """
    Generate a mesh for scar tissue geometries
    """
    gmsh.clear()
    gmsh.model.add('main')

    gmsh.option.set_string('Geometry.OCCTargetUnit', 'CM')
    gmsh.merge(f'BCS_I_0.16.STEP')

    gmsh.model.add_physical_group(2, [1], name='body')
    gmsh.model.add_physical_group(2, [2, 3], name='cover')
    gmsh.model.add_physical_group(2, [4, 5], name='scar')
    
    gmsh.model.add_physical_group(1, [10, 9, 20, 19, 12, 16, 15], name='pressure')
    gmsh.model.add_physical_group(1, [11, 5, 14], name='fixed')

    gmsh.model.add_physical_group(0, [11], name='separation-inf')
    gmsh.model.add_physical_group(0, [14], name='separation-sup')

    gmsh.option.set_number('Mesh.MshFileVersion', MSH_VER)
    gmsh.option.set_number('Mesh.MeshSizeFactor', SIZE_FACTOR)

    gmsh.model.mesh.generate(2)
    gmsh.write(OUTPUT_DIR + f'BCS_I--SD0.16--DZ0.00--NZ1--clscale2.50e-01.msh')

if __name__ == '__main__':
    proc_BC()
    proc_BCS_M_04()
    proc_BCS_M_08()
    proc_BCS_M_12()
    proc_BCS_M_16()
    proc_BCS_S_04()
    proc_BCS_S_08()
    proc_BCS_S_12()
    proc_BCS_S_16()    
    proc_BCS_I_04()
    proc_BCS_I_08()
    proc_BCS_I_12()
    proc_BCS_I_16()