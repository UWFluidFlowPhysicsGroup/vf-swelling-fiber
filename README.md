# vf-swelling-fiber
Installing Conda
Conda is used for version control between environments (Ex. Having Fenics 1.0.0 vs 1.1.0 for running different programs)
Fenics needs an environment to run

Installing Conda through WSL Terminal (Linux installation)
•	sudo apt install -y curl git wget build-essential software-properties-common
•	cd ~ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh bash Miniconda3-latest-Linux-x86_64.sh
Install with default location
Will ask if Conda initializes on startup, selecting yes since no other environments are being used, and can reverse with ‘conda init --reverse $SHELL’

After installing Conda, will see (base) before user input line (green text), can verify installation using ‘conda --version’
Have to accept terms of service for Conda before using 
•	conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
•	conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

Creating and managing environments
•	conda create -n fenics-env python=3.10 -y
Creates new environment fenics-env using python version 3.10
Similar to creating a new git branch, any variables installed in the new environment using conda will be added when the environment is loaded, and removed when changing to a different environment (See below: Installing Fenics section for environment specific examples)
•	conda activate fenics-env
This will change the leading (base) to (fenics-env) or whatever the new environment name is
Note: OpenIFEM MPI simulations will run n simulations instead of 1 simulation with n cores if using fenics-env environment. Change back to base environment using conda activate base when running OpenIFEM simulations.


Installing Fenics
Fenics needs to be installed through Conda for each environment using 
•	conda install -c conda-forge fenics
This also installs petsc4py and slepc4py and other dependencies required for fenics
fenics is installed only in the fenics-env environment. All dependencies that are installed from running this command are also only present in the current environment.
	

Installng other dependencies for vf-swelling
•	conda install -c conda-forge pyvista meshio gmsh matplotlib scipy numpy jupyter tqdm h5py jax jaxlib pytest lxml
Can also install other dependencies using pip install, but installs packages as global instead of per environment. Using conda lets newer versions still be run globally instead.
Conda installs the newest version of each dependency package that is compatible with fenics or other previously installed programs
	

Fenics test code: 
Paste the following as one command:
python - <<'EOF'
from dolfin import *
mesh = UnitSquareMesh(8, 8)
V = FunctionSpace(mesh, "P", 1)
u = TrialFunction(V)
v = TestFunction(V)
a = dot(grad(u), grad(v)) * dx
L = Constant(0) * v * dx
u_sol = Function(V)
solve(a == L, u_sol)
print("✅ FEniCS working fine. Degrees of freedom:", V.dim())
EOF
This should output 81 DoFs, and means Fenics should be running properly

Installing vf repos:
All folders from repo will be in same main folder
•	mkdir swelling
•	cd swelling
•	git clone [url]
Required URLs: 
•	 https://github.com/UWFluidFlowPhysicsGroup/block-array
•	https://github.com/UWFluidFlowPhysicsGroup/vf-signals 
•	https://github.com/UWFluidFlowPhysicsGroup/vf-exputils 
•	https://github.com/UWFluidFlowPhysicsGroup/nonlinear-equation   
•	https://github.com/UWFluidFlowPhysicsGroup/vf-fem    then git checkout vf-fem_swelling_fiber 
•	https://github.com/UWFluidFlowPhysicsGroup/vf-swelling-fiber 
Everything will be installed in the /swelling/ folder
•	swelling/block-array     
•	swelling/nonlinear-equation
•	swelling/vf-exputils
•	swelling/vf-fem
•	swelling/vf-swelling-fiber
	
•	Installing block-array (https://github.com/jon-deng/block-array.git   or  https://github.com/UWFluidFlowPhysicsGroup/block-array)
	pip install -e .
	
Installing nonlinear-equation (https://github.com/jon-deng/nonlinear-equation.git or https://github.com/UWFluidFlowPhysicsGroup/nonlinear-equation)
	pip install -e .

•	Installing vf-exputils (https://github.com/jon-deng/vf-exputils.git or https://github.com/UWFluidFlowPhysicsGroup/vf-exputils)
Note: url has been changed, need to go into jon’s repositories list and find in there
	pip install -e .

•	Installing vf-signals (https://github.com/jon-deng/vf-signals.git or https://github.com/UWFluidFlowPhysicsGroup/vf-signals)
Note: vf-fem not working properly because of tags, will be fixed soon when Nahid’s code is pushed
	pip install -e .	

Installing vf-onset-sensitivity (https://github.com/jon-deng/vf-onset-sensitivity.git or https://github.com/UWFluidFlowPhysicsGroup/vf-onset-sensitivity )
Note: vf-onset-sensitivity placed in same directory as home folder (cd .. from swelling)
pip install -e .

Installing vf-fem (https://github.com/jon-deng/vf-fem.git)
pip install -e .
	Note: vf-fem not working properly because of tags, will be fixed soon when Nahid’s pull request is accepted

	Change tag to vf_swelling_fiber by git checkout vf_swelling_fiber
		This changes your files to a specific commit named vf_swelling_fiber

•	Installing vf-swelling (https://github.com/jon-deng/vf-swelling.git or https://github.com/UWFluidFlowPhysicsGroup/vf-swelling-fiber )
	Note: vf-fem not working properly because of tags, will be fixed soon when Nahid’s code is pushed	

Create out folder to store outputs in vf-swelling if not already created
•	cd vf-swelling-fiber
•	mkdir out

	Open VSCode and select swelling folder as the working folder, the file tree should look like this:
 
Open Ubuntu (WSL) terminal in VSCode
	Opening terminal defaults to powershell, select Ubuntu (WSL) from side menu
 

In VS Code WSL Terminal:
Change environment to fenics-env
•	conda activate fenics-env	
Move into vf-swelling folder and set variables for other dependency paths
•	cd vf-swelling-fiber
•	export PYTHONPATH=../vf-fem:../vf-exputils:../nonlinear-equation:../block-array

Running test code for 2D case:
•	python mainsigmoid.py --study-name main_2D
Note: --postprocess was added as a build option at the end of the above command for older versions of the code. Using --postprocess will produce an error if the command is not needed.
<img width="468" height="641" alt="image" src="https://github.com/user-attachments/assets/bd971b07-1983-400a-baa0-8f2d7bc0eedd" />
