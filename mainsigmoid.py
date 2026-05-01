"""
Run a sequence of vocal fold simulations with swelling
"""

from typing import List, Tuple, Mapping, Optional, Callable
from numpy.typing import NDArray

from os import path
import argparse as ap
import multiprocessing as mp    # parallel execution
import itertools as it
import functools
from typing import List, Mapping

import numpy as np
import dolfin as dfn 
import h5py

# --- Project-specific imports ---
from femvf import forward, static, statefile as sf, meshutils
from femvf.models.transient import solid, fluid, base as trabase, coupled
from femvf.models.dynamical import base as dynbase
from femvf.postprocess.base import TimeSeries, TimeSeriesStats
from femvf.postprocess import solid as slsig
from femvf.load import load_transient_fsi_model

from blockarray import blockvec as bv

from exputils import postprocutils, exputils

dfn.set_log_level(50)

## Defaults for 'nominal' parameter values
MESH_BASE_NAME = 'M5_BC'    # base mesh name
CLSCALE = 0.25              # mesh scaling
POISSONS_RATIO = 0.4        # Poisson’s ratio
PSUB = 400 * 10             # subglottal pressure (Pa)

VCOVERS = np.array([1.0, 1.15 , 1.3])   # cover swelling multipliers
MCOVERS = np.array([-0.8])              # cover swelling modifiers

ECOV = 2.5e4   # elastic modulus (cover)
EBOD = 5e4     # elastic modulus (body)

PARAM_SPEC = {
    'MeshName': str,
    'GA': float,
    'DZ': float,
    'NZ': int,
    'clscale': float,
    'Ecov': float,
    'Ebod': float,
    'gammaFcov': float,
    'gammaFbod': float,
    'vcov': float,
    'mcov': float,
    'psub': float,
    'dt': float,
    'tf': float,
    'ModifyEffect': str,
    'SwellingDistribution': str
}
PARAM_FORMAT_STRS = {'vcov': '.4e'}

import scipy.special  # Add this import at the top if not present

def sigmoid(x, x0=0.0, k=1.0):
    """Sigmoid function for Smooth logistic transition function.. """""
    return 1 / (1 + np.exp(-k * (x - x0)))

ExpParam = exputils.make_parameters(PARAM_SPEC, PARAM_FORMAT_STRS)

Model = coupled.BaseTransientFSIModel
# Builds the unique filename string based on parameter values.
def setup_mesh_name(param: ExpParam) -> str:
    """
    Return the name of the mesh
    """
    base_name = param['MeshName']
    ga = param['GA']
    clscale = param['clscale']
    dz = param['DZ']
    nz = param['NZ']
    return f'{base_name}--GA{ga:.2f}--DZ{dz:.2f}--NZ{nz:d}--clscale{clscale:.2e}'
#Loads the transient FSI (fluid–structure interaction) model considering fiber direction from mesh.
def setup_model(param: ExpParam) -> Model:
    """
    Return the model
    """
    mesh_path = f"mesh/{setup_mesh_name(param)}.msh"

    if param['DZ'] == 0.0:
        zs = None
    elif param['DZ'] > 0.0:
        zs = np.linspace(0, param['DZ'], param['NZ']+1)
    else:
        raise ValueError("Parameter 'DZ' must be >= 0")

    model = load_transient_fsi_model(
        mesh_path, None,
        #SolidType=solid.SwellingKelvinVoigtWEpitheliumNoShape,   #no Fiber Effect
        SolidType=solid.SwellingKelvinVoigtWEpitheliumNoShapeFiber,   #with Fiber Effect
        FluidType=fluid.BernoulliAreaRatioSep,
        zs=zs
    )
    return model

def setup_state_control_props(
        params: ExpParam, model: Model 
    ) -> Tuple[bv.BlockVector, bv.BlockVector, bv.BlockVector]:
    """
    Return a (state, controls, prop) tuple defining a transient run
    """
    ## Set 'basic' model properties
    # These properties don't include the glottal gap since you may/may not
    # want to modify the glottal gap based on the swelling level
    prop = setup_basic_props(params, model)  # material/swelling props
    model.set_prop(prop)

    ## Set the initial state 
    # The initial state is based on the post-swelling static configuration
    state0 = setup_ini_state(params, model)

    # Set the glottal gap based on the post-swelling static configuration
    ndim = model.solid.residual.mesh().topology().dim()
    if (params['ModifyEffect'] == 'const_pregap'
        or params['ModifyEffect'] == 'const_mass_pregap'
        ):
        # Using the `ndim` to space things ensures you get the y-coordinate
        # for both 2D and 3D meshes
        ymax = (model.solid.XREF + state0.sub['u'])[1::ndim].max()
    else:
        ymax = (model.solid.XREF)[1::ndim].max()
    ygap = 0.03 # 0.3 mm half-gap -> 0.6 mm glottal gap  ## initial half-gap
    ycoll_offset = 1/10*ygap

    prop['ycontact'] = ymax + ygap - ycoll_offset
    prop['ymid'] = ymax + ygap
    for n in range(len(model.fluids)):
        prop[f'fluid{n}.area_lb'] = 2*ycoll_offset

    model.set_prop(prop)

    controls = setup_controls(params, model)
    return state0, controls, prop

def setup_basic_props(param: ExpParam, model: Model) -> bv.BlockVector:
    """
    Set the properties vector
    # FEM mesh + mapping from labeled regions -> DOFs
    """
    mesh = model.solid.residual.mesh()
    forms = model.solid.residual.form
    mf = model.solid.residual.mesh_function('cell')
    mf_label_to_value = model.solid.residual.mesh_function_label_to_value('cell')
    cellregion_to_sdof = meshutils.process_meshlabel_to_dofs(
        mesh, mf, mf_label_to_value,
        forms['coeff.prop.emod'].function_space().dofmap(),
        
    )

    prop = model.prop.copy()
    # prop[:] = 0
    ## Solid constant properties  # Assign baseline solid/fluid constants
    prop['rho'] = 1.0
    prop['eta'] = 5.0
    prop['kcontact'] = 1e15
    # prop['ncontact'] = [0, 1]

    ## Fluid constant properties
    for n in range(len(model.fluids)):
        prop[f'fluid{n}.r_sep'] = 1.2
        prop[f'fluid{n}.rho_air'] = 1.2e-3

    ## Swelling specific properties
    if (
            param['ModifyEffect'] == 'const_pregap'
            or param['ModifyEffect'] == ''
        ):
        # This one is controlled in the initial state
        modify_kwargs = {}
    elif (
            param['ModifyEffect'] == 'const_mass'
            or param['ModifyEffect'] == 'const_mass_pregap'
        ):
        modify_kwargs = {'modify_density': False}
    else:
        raise ValueError(f"Unkown 'ModifyEffect' parameter {param['ModifyEffect']}")

    prop = _set_swelling_props(
        param, model, prop, cellregion_to_sdof,
        **modify_kwargs
    )

    ## Set VF layer properties  # Apply elastic moduli and fiber moduli for body/cover 
    emods = {
        'cover': param['Ecov'],
        'body': param['Ebod']
    }

    gamma_fiber = {
        'cover': param['gammaFcov'],
        'body': param['gammaFbod']
    }
    prop = _set_layer_props(model, prop, emods,gamma_fiber, cellregion_to_sdof)
    #prop = _set_layer_props(model,prop, emods, cellregion_to_sdof)

    return prop

def setup_controls(param: ExpParam, model: Model) -> bv.BlockVector:
    """
    Set the controls
    """
    control = model.control.copy()
    control[:] = 0

    for n in range(len(model.fluids)):
        control[f'fluid{n}.psub'] = param['psub']
        control[f'fluid{n}.psup'] = 0.0

    return [control]

def setup_ini_state(param: ExpParam, model: Model) -> bv.BlockVector:
    """
    Set the initial state vector
    """
    state0 = model.state0.copy()
    state0[:] = 0.0
    model.solid.control[:] = 0.0

    vcov = param['vcov']
    nload = max(int(round((vcov-1)/0.025)), 1)

    prop = setup_basic_props(param, model)
    model.set_prop(prop)

    static_state, _ = solve_static_swollen_config(
        model.solid, model.solid.control, model.solid.prop, nload
    )

    state0[['u', 'v', 'a']] = static_state
    return state0

def _set_swelling_props(
        param: ExpParam,
        model: Model,
        prop: bv.BlockVector,
        cellregion_to_sdof: Mapping[str, NDArray],
        modify_density=True,
        modify_geometry=True,
        out_dir='out'
    ) -> bv.BlockVector:
    """
    Set properties related to the level of swelling

    Parameters
    ----------
    v :
        swelling field
    """
    RHO_VF = 1.0
    RHO_SWELL = 1.0
    # KSWELL_FACTOR = 20

    # Apply swelling parameters to the cover layer only
    # dofs_cov = np.unique(
    #     np.concatenate(
    #         [cellregion_to_sdof[label] for label in ['medial', 'inferior', 'superior']]
    #     )
    # )
    dofs_cov = np.unique(
        np.concatenate(
            [cellregion_to_sdof[label] for label in ['cover']]
        )
    )
    dofs_bod = cellregion_to_sdof['body']
    # dofs_sha = np.intersect1d(dofs_cov, dofs_bod)

    # prop['k_swelling'][0] = KSWELL_FACTOR * 10e3*10
    prop['m_swelling'][dofs_cov] = param['mcov']

    prop['v_swelling'][:] = 1.0
    if modify_geometry:
        prop['v_swelling'][dofs_cov] = param['vcov']
        prop['v_swelling'][dofs_bod] = 1.0

    prop['rho'][:] = RHO_VF

    if modify_density:
        _v = np.array(prop['v_swelling'][:])
        prop['rho'][:] = RHO_VF + (_v-1)*RHO_SWELL

    ## TODO : Fix this ad-hoc thing to do the swelling based on damage
    if (
        param['SwellingDistribution'] != 'uniform'
        and param['vcov'] != 1.0
        ):
        param_unswollen = param.substitute({
            'vcov': 1.0
        })
        with h5py.File(f'out/postprocess.h5', mode='a') as f:
            # damage_key = 'field.tavg_viscous_rate'
            damage_key = param['SwellingDistribution']
            group_name = param_unswollen.to_str()
            dataset_name = f'{group_name}/{damage_key}'
            # Check if the post-processed damage measure exists;
            # if not, post-process the damage measure.
            if dataset_name not in f:
                state_fpath = path.join(out_dir, f'{group_name}.h5')
                # If the simulation hasn't been run, then run it first
                if not path.isfile(state_fpath):
                    run(param_unswollen, out_dir)

                with sf.StateFile(model, state_fpath, mode='r') as fstate:
                    postprocess = get_result_name_to_postprocess(model)[damage_key]
                    group = f.require_group(group_name)
                    group.create_dataset(damage_key, data=postprocess(fstate))
            _damage = f[dataset_name][:]

        residual = model.solid.residual
        damage = residual.form['coeff.prop.v_swelling'].copy()
        damage.vector()[:] = _damage
        v_swelling = damage.copy()

        cell_label_to_id = residual.mesh_function_label_to_value('cell')
        dx = residual.measure('dx')
        dx_cover = dx(int(cell_label_to_id['cover']))
        # Make the increase in volume (i.e. `v-1`) proportional to the damage
        # measure and scale the resulting swelling field so that the total
        # prescribed volume increase is `param['vcov']`
        original_vol = dfn.assemble(1*dx_cover)
        swollen_vol_incr = dfn.assemble(damage*dx_cover)
        v_factor = (param['vcov']*original_vol - original_vol)/swollen_vol_incr
        v_swelling.vector()[:] = (1+v_factor*damage.vector()[:])
        # print(dfn.assemble(v_swell*dx_cover)/dfn.assemble(1*dx_cover))
        prop['v_swelling'][:] = v_swelling.vector()[:]
        prop['v_swelling'][dofs_bod] = 1.0

    return prop

import numpy as np
import dolfin as dfn

# optional: nearest neighbor accel (fallback to naive if missing)
try:
    from scipy.spatial import cKDTree as _KDTree
except Exception:
    _KDTree = None

def _to_np_idx(x):
    """Robustly convert a list/iterable of indices to a 1D np.int64 array."""
    if x is None:
        return np.empty(0, dtype=int)
    try:
        arr = np.asarray(x, dtype=int)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        return arr
    except Exception:
        return np.array(list(x), dtype=int)

def _set_layer_props(
        model,
        prop,
        emods: Mapping[str, float],
        gamma_fiber: Mapping[str, float],
        cellregion_to_sdof: Mapping[str, NDArray]
    ):
    """
    Set properties for each layer using a sigmoid centered on the actual
    body–cover interface (distance-based), with robust handling of cell ids.
    """
    mesh = model.solid.residual.mesh()
    dim  = mesh.geometry().dim()
    cells = mesh.cells()
    Xv = mesh.coordinates()

    # --- function spaces for properties ---
    V_E = model.solid.residual.form['coeff.prop.emod'].function_space()
    V_G = model.solid.residual.form['coeff.prop.gamma_fiber'].function_space()
    Xd_E = V_E.tabulate_dof_coordinates().reshape(-1, dim)
    Xd_G = V_G.tabulate_dof_coordinates().reshape(-1, dim)

    # --- find interface vertices from labeled cells (ROBUST ARRAYS) ---
    mf_cell = model.solid.residual.mesh_function('cell')
    lab2val = model.solid.residual.mesh_function_label_to_value('cell')

    if 'cover' in lab2val:
        cover_cell_ids = _to_np_idx(mf_cell.where_equal(lab2val['cover']))
    else:
        # legacy split cover labels
        parts = []
        for k in ('inferior', 'medial', 'superior'):
            if k in lab2val:
                parts.append(_to_np_idx(mf_cell.where_equal(lab2val[k])))
        cover_cell_ids = np.concatenate(parts) if parts else np.empty(0, dtype=int)

    body_cell_ids = _to_np_idx(mf_cell.where_equal(lab2val['body'])) if 'body' in lab2val else np.empty(0, dtype=int)

    verts_cover = np.unique(cells[cover_cell_ids].ravel()) if cover_cell_ids.size else np.empty(0, dtype=int)
    verts_body  = np.unique(cells[body_cell_ids ].ravel()) if body_cell_ids.size  else np.empty(0, dtype=int)
    verts_interface = np.array(sorted(set(verts_cover).intersection(set(verts_body))), dtype=int)

    # --- build signed distance at DOF coords (cover +, body -) ---
    def signed_distance_for(V_space, Xd, region_to_sdof):
        # region signs from dof-index maps
        if 'cover' in region_to_sdof:
            dofs_cover = np.unique(region_to_sdof['cover'])
        else:
            dofs_cover = np.unique(np.concatenate([
                region_to_sdof.get(k, np.empty(0, dtype=int))
                for k in ('inferior','medial','superior')
            ])) if any(k in region_to_sdof for k in ('inferior','medial','superior')) else np.empty(0, dtype=int)
        dofs_body = np.unique(region_to_sdof.get('body', np.empty(0, dtype=int)))

        sign = np.zeros(Xd.shape[0], dtype=float)
        sign[dofs_cover] = +1.0
        sign[dofs_body]  = -1.0
        sign[sign == 0.0] = +1.0  # unlabeled -> treat as cover

        # distance to interface (fallback to mid-y if no interface found)
        if verts_interface.size >= 1:
            Xi = Xv[verts_interface, :]
            if _KDTree is not None and Xi.shape[0] > 0:
                dist = _KDTree(Xi).query(Xd, k=1)[0]
            else:
                # naive fallback
                diffs = Xd[:, None, :] - Xi[None, :, :]
                dist = np.sqrt(np.sum(diffs*diffs, axis=2)).min(axis=1)
        else:
            y0 = 0.5*(Xv[:,1].min() + Xv[:,1].max())
            dist = np.abs(Xd[:,1] - y0)

        return sign * dist  # <0 body side, >0 cover side

    sd_E = signed_distance_for(V_E, Xd_E, cellregion_to_sdof)
    sd_G = signed_distance_for(V_G, Xd_G, cellregion_to_sdof)

    # --- pick logistic thickness from interface spacing ---
    def interface_spacing():
        if verts_interface.size >= 2:
            Xi = Xv[verts_interface, :]
            if _KDTree is not None:
                nn = _KDTree(Xi).query(Xi, k=2)[0][:, 1]
                nn = nn[np.isfinite(nn)]
                if nn.size:
                    return float(np.median(nn))
            # very rough fallback
            return float(np.median(np.linalg.norm(np.diff(np.sort(Xi, axis=0), axis=0), axis=1))) if Xi.shape[0] > 1 else 1.0
        return 1.0

    h_int = interface_spacing()
    thickness = 3.0 * max(h_int, 1e-12)
    k = 4.0 / thickness                  # 10–90% logistic width ~ thickness

    def logistic(signed_d):
        return 1.0 / (1.0 + np.exp(-k * signed_d))

    alpha_E = logistic(sd_E)             # ~0 body, ~1 cover
    alpha_G = logistic(sd_G)

    # --- blend properties ---
    E_cover, E_body = float(emods['cover']), float(emods['body'])
    G_cover, G_body = float(gamma_fiber['cover']), float(gamma_fiber['body'])

    E_vals = E_cover * alpha_E + E_body * (1.0 - alpha_E)
    G_vals = G_cover * alpha_G + G_body * (1.0 - alpha_G)

    E_fun = dfn.Function(V_E); E_fun.vector()[:] = E_vals
    G_fun = dfn.Function(V_G); G_fun.vector()[:] = G_vals

    prop['emod'][:]        = E_fun.vector()[:]
    prop['gamma_fiber'][:] = G_fun.vector()[:]

    # --- remaining constants as in your original code ---
    prop['nu'][:] = POISSONS_RATIO
    prop['emod_membrane'][:] = 50e3 * 10
    prop['th_membrane'][:]   = 0.005
    prop['nu_membrane'][:]   = POISSONS_RATIO

    return prop



def solve_static_swollen_config(
        model: Model,
        control: bv.BlockVector,
        prop: bv.BlockVector,
        nload: int=1
    ):
    """
    Solve for the static swollen configuration

    This uses `nload` loading steps of the swelling field
    """
    if isinstance(model, trabase.BaseTransientModel):
        static_state_n = model.state0.copy()
        static_state_n[:] = 0
        model.dt = 1.0
    elif isinstance(model, dynbase.BaseDynamicalModel):
        static_state_n = model.state.copy()
        static_state_n[:] = 0

    v_final = prop['v_swelling'][:].copy()
    v_0 = v_final.copy()
    v_0[:] = 1
    dv = (v_final-v_0)/nload

    props_n = prop.copy()
    props_n['v_swelling'][:] = v_0

    info = {}
    for n in range(nload+1):
        props_n['v_swelling'][:] = v_0 + n*dv
        static_state_n, info = static.static_solid_configuration(
            model, control, props_n, state=static_state_n
        )
    return static_state_n, info


def make_exp_params(study_name: str) -> List[ExpParam]:
    DEFAULT_PARAM_2D = ExpParam({
        'MeshName': MESH_BASE_NAME, 'clscale': CLSCALE,
        'GA': 3,
        'DZ': 0.00, 'NZ': 1,
        'Ecov': ECOV, 'Ebod': EBOD,
        'gammaFcov': 5e4,
        'gammaFbod': 5e5,
        'vcov': 1.0, 'mcov': 0.0,
        'psub': 600*10,
        'dt': DT, 'tf': TF,
        'ModifyEffect': '',
        'SwellingDistribution': 'uniform'
    })

    DEFAULT_PARAM_3D = ExpParam({
        'MeshName': MESH_BASE_NAME, 'clscale': CLSCALE,
        'GA': 3,
        'DZ': 1.5, 'NZ': 15,
        'Ecov': ECOV, 'Ebod': EBOD,
        'gammaFcov': 5e4,
        'gammaFbod': 5e5,
        'vcov': 1, 'mcov': 0.0,
        'psub': 600*10,
        'dt': DT, 'tf': TF,
        'ModifyEffect': '',
        'SwellingDistribution': 'uniform'
    })
    if study_name == 'none':
        params = []
    elif study_name == 'test':
        # vcovs = [1.0, 1.1, 1.2, 1.3]
        params = [
            DEFAULT_PARAM_2D.substitute({
                'MeshName': MESH_BASE_NAME, 'clscale': CLSCALE,
                'GA': 3,
                'DZ': 1.5, 'NZ': 10,
                'Ecov': ECOV, 'Ebod': EBOD,
                'vcov': 1,
                'psub': 600*10,
                'dt': 5e-5, 'tf': 0.25
            })
            #for vcov in vcovs
        ]
    elif study_name == 'independence_2D':
        def make_param(clscale, dt):
            return DEFAULT_PARAM_2D.substitute({
                'clscale': clscale,
                'Ecov': ECOV, 'Ebod': EBOD,
                'vcov': 1.3, 'mcov': -1.6,
                'psub': PSUB,
                'dt': dt, 'tf': 0.5
            })

        clscales = 1 * 2.0**np.arange(1, -2, -1)
        dts = 5e-5 * 2.0**np.arange(0, -5, -1)
        params = [
            make_param(clscale, dt) for clscale in clscales for dt in dts
        ]
    elif study_name == 'main_2D':
        def make_param(elayers, vcov, mcov, gammaFcov, gammaFbod):
            return DEFAULT_PARAM_2D.substitute({
                'Ecov': elayers['cover'],
                'Ebod': elayers['body'],
                'vcov': vcov,
                'mcov': mcov,
                'gammaFcov': gammaFcov,
                'gammaFbod': gammaFbod
         })

        # Define allowed gamma pairs
        GAMMA_PAIRS = [
            (0.0, 0.0),     # no fiber
            (5e4, 5e5)      # fiber
        ]

        params = [
            make_param(elayers, vcov, mcov, gammaFcov, gammaFbod)
            for elayers, vcov, mcov, (gammaFcov, gammaFbod) 
            in it.product(EMODS, VCOVERS, MCOVERS, GAMMA_PAIRS)
    ]
    elif study_name == 'main_2D_coarse':
        def make_param(elayers, vcov, mcov):
            return DEFAULT_PARAM_2D.substitute({
                'Ecov': elayers['cover'], 'Ebod': elayers['body'],
                'vcov': vcov, 'mcov': mcov,
                'dt': 1e-4, 'tf': 0.5
            })

        vcovs = np.array([1.0, 1.1, 1.2, 1.3])
        mcovs = np.array([0.0, -0.8])

        params = [
            make_param(*args) for args in it.product(EMODS, VCOVERS, MCOVERS)
        ]
    elif study_name == 'main_3D_setup':
        # This case is the setup for the unswollen 3D state
        def make_param(elayers, vcov, mcov, damage):
            return DEFAULT_PARAM_3D.substitute({
                'Ecov': elayers['cover'], 'Ebod': elayers['body'],
                'vcov': vcov, 'mcov': mcov,
                'SwellingDistribution': damage
            })

        vcovs = np.array([1.0])
        mcovs = np.array([0.0, -0.8, -1.6])
        damage_measures = [
            'field.tavg_viscous_rate',
            # 'field.tavg_strain_energy'
        ]

        params = [
            make_param(*args)
            for args in it.product(EMODS, vcovs, mcovs, damage_measures)
        ]
    elif study_name == 'main_3D':
        # This case is the setup for the unswollen 3D state
        def make_param(elayers, vcov, mcov, damage):
            return DEFAULT_PARAM_3D.substitute({
                'Ecov': elayers['cover'], 'Ebod': elayers['body'],
                'vcov': vcov, 'mcov': mcov,

                'SwellingDistribution': 'uniform'
            })
        
        vcovs = np.array([1, 1.15, 1.3])
        mcovs = np.array([-0.8])
        damage_measures = [
            'field.tavg_viscous_rate',
            # 'field.tavg_strain_energy'
        ]

        params = [
            make_param(*args)
            for args in it.product(EMODS, vcovs, mcovs, damage_measures)
        ]
    elif study_name == 'block_free_y_fsi0':
        # This case is the setup for the unswollen 3D state
        def make_param(elayers, vcov, mcov, damage):
            return DEFAULT_PARAM_3D.substitute({
                'Ecov': elayers['cover'], 'Ebod': elayers['body'],
                'vcov': vcov, 'mcov': mcov,
                'SwellingDistribution': damage
            })

        vcovs = np.array([1.0, 1.05, 1.1, 1.2, 1.25])
        mcovs = np.array([0.0, -0.8])
        damage_measures = [
            'field.tavg_viscous_rate',
            # 'field.tavg_strain_energy'
        ]

        params = [
            make_param(*args)
            for args in it.product(EMODS, vcovs, mcovs, damage_measures)
        ]
    elif study_name == 'main_3D_coarse':
        # This case is the setup for the unswollen 3D state
        def make_param(elayers, vcov, mcov, damage):
            return DEFAULT_PARAM_3D.substitute({
                'MeshName': MESH_BASE_NAME, 'clscale': 0.5,
                'GA': 3, 'DZ': 1.5, 'NZ': 10,
                'Ecov': elayers['cover'], 'Ebod': elayers['body'],
                'vcov': vcov, 'mcov': mcov,
                'SwellingDistribution': damage
            })

        vcovs = np.array([1.0, 1.1, 1.2, 1.3])
        mcovs = np.array([0.0, -0.8])
        damage_measures = [
            'field.tavg_viscous_rate',
            'field.tavg_strain_energy'
        ]

        params = [
            make_param(*args)
            for args in it.product(EMODS, vcovs, mcovs, damage_measures)
        ]
    elif study_name == 'main_3D_xdmf':
        # This case is the setup for the unswollen 3D state
        def make_param(elayers, vcov, mcov, damage):
            return DEFAULT_PARAM_3D.substitute({
                'Ecov': elayers['cover'], 'Ebod': elayers['body'],
                'vcov': vcov, 'mcov': mcov,
                'dt': 1e-4, 'tf': 0.25, 'clscale': 0.5, 'NZ': 10,
                'SwellingDistribution': 'uniform'
            })

        vcovs = np.array([1.0, 1.1, 1.2, 1.3])
        mcovs = np.array([ -0.8])
        damage_measures = [
            'field.tavg_viscous_rate',
            'field.tavg_strain_energy'
        ]

        params = [
            make_param(*args)
            for args in it.product(EMODS, vcovs, mcovs, damage_measures)
        ]
    elif study_name == 'const_pregap':
        def make_param(elayers, vcov, mcov):
            return DEFAULT_PARAM_2D.substitute({
                'Ecov': elayers['cover'], 'Ebod': elayers['body'],
                'vcov': vcov, 'mcov': mcov,
                'ModifyEffect': 'const_pregap'
            })

        params = [
            make_param(*args) for args in it.product(EMODS, VCOVERS, MCOVERS)
        ]
    elif study_name == 'const_mass':
        def make_param(elayers, vcov, mcov):
            return DEFAULT_PARAM_2D.substitute({
                'Ecov': elayers['cover'], 'Ebod': elayers['body'],
                'vcov': vcov, 'mcov': mcov,
                'ModifyEffect': 'const_mass'
            })

        params = [
            make_param(*args) for args in it.product(EMODS, VCOVERS, MCOVERS)
        ]
    elif study_name == 'const_mass_pregap':
        def make_param(elayers, vcov, mcov):
            return DEFAULT_PARAM_2D.substitute({
                'Ecov': elayers['cover'], 'Ebod': elayers['body'],
                'vcov': vcov, 'mcov': mcov,
                'ModifyEffect': 'const_mass_pregap'
            })

        params = [
            make_param(*args) for args in it.product(EMODS, VCOVERS, MCOVERS)
        ]
    elif study_name == '2D_test':
        params = [
            DEFAULT_PARAM_2D.substitute({
                'MeshName': MESH_BASE_NAME,
                'clscale': 0.5,
                'GA': 3,
                'DZ': 0, 'NZ': 1,
                'Ecov': ECOV,
                'Ebod': EBOD,
                'vcov': 1,
                'psub': 600*10,
                'dt': 5e-5,
                'tf': 0.25
            })
        ]
    else:
        raise ValueError(f"Unknown `--study-name` {study_name}")

    return params


## Main functions for running/postprocessing simulations #Integrates transient model and writes results to HDF5.
def run(
        param: dict,
        out_dir: str
    ):
    """
    Run the transient simulation
    """
    # We need to convert `param` from a dict to the `ExpParam` object
    # because you can't call `run` in parallel if `param`
    # is not pickleable (`ExpParam` instances can't be pickled)
    param = ExpParam(param)
    out_path = f'{out_dir}/{param.to_str()}.h5'
    if not path.isfile(out_path):
        model = setup_model(param)
        state0, controls, prop = setup_state_control_props(param, model)
        # breakpoint()

        dt = param['dt']
        tf = param['tf']
        times = dt*np.arange(0, int(round(tf/dt))+1)

        with sf.StateFile(model, out_path, mode='a') as f:
            forward.integrate(
                model, f, state0, controls, prop, times, use_tqdm=True
            )
    else:
        print(f"Skipping {out_path} because the file already exists")

    return out_path

def postprocess(
        out_fpath: str, in_fpaths: List[str],
        overwrite_results: Optional[List[str]]=None,
        num_proc: int=1
    ):
    """
    Postprocess key signals from the simulations
    """
    # signal_to_proc = get_result_to_proc(model)
    # result_names = list(signal_to_proc.keys())

    with h5py.File(out_fpath, mode='a') as f:
        for in_fpath in in_fpaths:
            case_name = path.splitext(in_fpath.split('/')[-1])[0]
            print(case_name)
            postprocutils.postprocess_parallel(
                f.require_group(case_name),
                in_fpath, get_model, get_result_name_to_postprocess,
                num_proc=num_proc, overwrite_results=overwrite_results
            )

from femvf.vis import xdmfutils
def postprocess_xdmf(
        model, param: ExpParam, xdmf_path: str,
        overwrite: bool=False
    ):
    """
    Write an XDMF file
    """
    xdfm_data_dir, xdmf_basename = path.split(xdmf_path)
    xdmf_data_basename = f'{path.splitext(xdmf_basename)[0]}.h5'
    xdmf_data_path = path.join(xdfm_data_dir, xdmf_data_basename)
    with (
            h5py.File(f'out/{param.to_str()}.h5', mode='r') as fstate,
            h5py.File(f'out/postprocess.h5', mode='r') as fpost,
            h5py.File(xdmf_data_path, mode='w') as fxdmf
        ):
        # breakpoint()
        # Export mesh values
        _labels = ['mesh/solid', 'time']
        labels = _labels
        datasets = [
            fstate[label] for label in _labels
        ]
        formats = [None, None]

        mesh = model.solid.residual.mesh()
        function_space = dfn.VectorFunctionSpace(mesh, 'CG', 1)
        _labels = ['state/u', 'state/v', 'state/a']
        labels += _labels
        datasets += [fstate[label] for label in _labels]
        formats += len(_labels)*[function_space]

        function_space = dfn.FunctionSpace(mesh, 'CG', 1)
        _labels = ['time.field.p']
        labels += _labels
        datasets += [fpost[f'{param.to_str()}/{label}'] for label in _labels]
        formats += len(_labels)*[function_space]

        function_space = dfn.FunctionSpace(mesh, 'DG', 0)
        _labels = ['field.tavg_viscous_rate', 'field.tavg_strain_energy']
        labels += _labels
        datasets += [fpost[f'{param.to_str()}/{label}'] for label in _labels]
        formats += len(_labels)*[function_space]
        xdmfutils.export_mesh_values(
            datasets, formats, fxdmf, output_names=labels
        )

        # Annotate the mesh values with an XDMF file
        static_dataset_descrs = [
            (fxdmf['state/u'], 'vector', 'node'),
            (fxdmf['field.tavg_viscous_rate'], 'scalar', 'Cell'),
            (fxdmf['field.tavg_strain_energy'], 'scalar', 'Cell'),
        ]
        static_idxs = [
            (0, ...), (slice(None),), (slice(None),)
        ]

        temporal_dataset_descrs = [
            (fxdmf['state/u'], 'vector', 'node'),
            (fxdmf['state/v'], 'vector', 'node'),
            (fxdmf['state/a'], 'vector', 'node'),
            (fxdmf['time.field.p'], 'scalar', 'node'),
        ]
        temporal_idxs = len(temporal_dataset_descrs)*[
            (slice(None),)
        ]
        # breakpoint()
        xdmfutils.write_xdmf(
            fxdmf['mesh/solid'],
            static_dataset_descrs, static_idxs,
            fxdmf['time'],
            temporal_dataset_descrs, temporal_idxs,
            xdmf_path
        )

def get_result_name_to_postprocess(
        model: trabase.BaseTransientModel
    ) -> Mapping[str, Callable[[sf.StateFile], np.ndarray]]:
    """
    Return a mapping of result names to post-processing functions

    Returns
    -------
    Mapping[str, Callable]
        A mapping from post-processed result names to functions

        The post-processed result names have a format where each axis of the
        data is given a name separated by dots. For example, the name
        'time.vertex.u' could indicate a 2-axis array of 'u' displacements where
        the first axis represents time while the second axis represents
        different vertices.
    """
    proc_gw = slsig.MeanGlottalWidth(model)

    cell_label_to_id = model.solid.residual.mesh_function_label_to_value('cell')
    facet_label_to_id = model.solid.residual.mesh_function_label_to_value('facet')
    dx = model.solid.residual.measure('dx')
    # dx_cover = (
    #     dx(int(cell_label_to_id['medial']))
    #     + dx(int(cell_label_to_id['inferior']))
    #     + dx(int(cell_label_to_id['superior']))
    # )
    dx_cover = dx(int(cell_label_to_id['cover']))
    # dx_medial = dx(int(cell_label_to_id['medial']))
    ds_medial = model.solid.residual.measure('ds')(int(facet_label_to_id['pressure']))
    proc_visc_rate = slsig.ViscousDissipationRate(model, dx=dx_cover)

    ## Project field variables onto a DG0 space
    # fspace_dg0 = model.solid.residual.form['coeff.fsi.p1'].function_space()
    fspace_dg0 = model.solid.residual.form['coeff.prop.eta'].function_space()
    proc_hydro_field = slsig.StressHydrostaticField(model)
    proc_vm_field = slsig.StressVonMisesField(model)
    proc_visc_diss_rate_field = slsig.ViscousDissipationField(
        model, dx=dx, fspace=fspace_dg0
    )

    proc_strain_energy = slsig.StrainEnergy(
        model, dx=dx, fspace=fspace_dg0
    )

    proc_contact_area_density_field = slsig.ContactAreaDensityField(
        model, dx=ds_medial, fspace=fspace_dg0
    )
    proc_contact_pressure_field = slsig.ContactPressureField(
        model, dx=ds_medial, fspace=fspace_dg0
    )

    ## Project field variables onto a CG1 space
    proc_ymom_field = slsig.YMomentum(
        model, dx=dx, fspace=model.solid.residual.form['coeff.fsi.p1'].function_space()
    )

    ## Compute spatial statistics of field variables
    def make_carea_field_stats():
        return slsig.FieldStats(
            proc_contact_area_density_field
        )
    def make_cpressure_field_stats():
        return slsig.FieldStats(
            proc_contact_pressure_field
        )
    def make_wvisc_field_stats(dx=dx):
        return slsig.FieldStats(
            slsig.ViscousDissipationField(model, dx=dx, fspace=fspace_dg0)
        )
    def make_svm_field_stats(dx=dx):
        return slsig.FieldStats(
            slsig.StressVonMisesField(model, dx=dx, fspace=fspace_dg0)
        )

    def make_ymom_field_stats():
        return slsig.FieldStats(proc_ymom_field)

    def proc_time(f):
        return f.get_times()

    def proc_q(f):
        qs = [
            np.sum([f.get_state(ii)[f'fluid{n}.q'][0] for n in range(len(f.model.fluids))])
            for ii in range(f.size)
        ]
        return np.array(qs)
    
    def proc_glottal_flow_rate(f: sf.StateFile) -> NDArray:
        """
    Return the glottal flow rate vector (time series).

    For each time step, average each fluid's q along its axial stations
    using trapezoidal weights (0.5 at ends, 1.0 interior), then sum across
    fluids. Works whether q is scalar or a 1D array.
    """
        q_t = []
        num_fluid = len(f.model.fluids)

        for i in range(f.size):
            q_sum = 0.0
            for n in range(num_fluid):
                qn_like = f.get_state(i)[f"fluid{n}.q"]  # numpy-like
                qn = np.asarray(qn_like)                 # ensure real np.ndarray

                if qn.ndim == 0 or qn.size == 1:
                    # scalar flow or single-station array
                    q_avg = float(qn.reshape(-1)[0])
                else:
                    # trapezoidal average along axial direction
                    w = np.ones(qn.size, dtype=float)
                    w[0] = w[-1] = 0.5
                    q_avg = float((qn * w).sum() / w.sum())

                q_sum += q_avg

            q_t.append(q_sum)

        return np.asarray(q_t, dtype=float)


    result_name_to_postprocess = {
        'time.q': proc_glottal_flow_rate,
        'time.gw': TimeSeries(proc_gw),
        'time.t': proc_time,
        'time.field.p': TimeSeries(slsig.FSIPressure(model)),

        'time.savg_viscous_rate': TimeSeries(proc_visc_rate),

        'field.tavg_viscous_rate': lambda f: TimeSeriesStats(proc_visc_diss_rate_field).mean(f, range(f.size//2, f.size)),
        'field.tavg_vm': lambda f: TimeSeriesStats(proc_vm_field).mean(f, range(f.size//2, f.size)),
        'field.tavg_hydrostatic': lambda f: TimeSeriesStats(proc_hydro_field).mean(f, range(f.size//2, f.size)),
        'field.tavg_pc': lambda f: TimeSeriesStats(proc_contact_pressure_field).mean(f, range(f.size//2, f.size)),
        'field.tavg_strain_energy': lambda f: TimeSeriesStats(proc_strain_energy).mean(f, range(f.size//2, f.size)),
        'field.tini_hydrostatic': lambda f: proc_hydro_field(f.get_state(0), f.get_control(0), f.get_prop()),
        'field.tini_vm': lambda f: proc_vm_field(f.get_state(0), f.get_control(0), f.get_prop()),
        # 'field.vswell': lambda f: f.get_prop().sub['v_swelling'],

        'time.spatial_stats_con_p': TimeSeries(make_cpressure_field_stats()),
        'time.spatial_stats_con_a': TimeSeries(make_carea_field_stats()),
        'time.spatial_stats_viscous': TimeSeries(make_wvisc_field_stats(dx_cover)),
        # 'time.spatial_stats_viscous_medial': TimeSeries(make_wvisc_field_stats(dx_medial)),
        'time.spatial_stats_vm': TimeSeries(make_svm_field_stats(dx_cover)),
        # 'time.spatial_stats_vm_medial': TimeSeries(make_svm_field_stats(dx_medial)),
        'time.spatial_state_ymom': TimeSeries(make_ymom_field_stats())
    }
    return result_name_to_postprocess

def get_model(in_fpath: str) -> Model:
    """Return the model"""
    in_fname = path.splitext(path.split(in_fpath)[-1])[0]
    params = ExpParam(in_fname)
    return setup_model(params)

if __name__ == '__main__':
    parser = ap.ArgumentParser()
    parser.add_argument("--num-proc", type=int, default=1)
    parser.add_argument("--study-name", type=str, default='none')
    parser.add_argument("--output-dir", type=str, default='out')
    parser.add_argument("--overwrite-results", type=str, action='extend', nargs='+')
    parser.add_argument("--default-dt", type=float, default=5e-5)
    parser.add_argument("--default-tf", type=float, default=0.25)
    parser.add_argument("--export-xdmf", action='store_true', default=True)
    clargs = parser.parse_args()

    TF = clargs.default_tf
    DT = clargs.default_dt

    postprocess = functools.partial(
        postprocess, overwrite_results=clargs.overwrite_results
    )

    # Pack up the emod arguments to a dict format
    _emods = np.array([[2.5, 5.0]]) * 1e3 * 10
    layer_labels = ['cover', 'body']
    EMODS = [
        {label: value for label, value in zip(layer_labels, layer_values)}
        for layer_values in _emods
    ]

    ## Run and postprocess simulations
    out_dir = clargs.output_dir
    params = make_exp_params(clargs.study_name)
    param_dicts = [param.data for param in params]
    if clargs.num_proc > 1:
        _run = functools.partial(run, out_dir=out_dir)
        with mp.Pool(processes=clargs.num_proc) as pool:
            print(f"Pool running with {clargs.num_proc:d} processors")
            in_fpaths = pool.map(_run, param_dicts, chunksize=1)
    else:
        in_fpaths = [run(params, out_dir) for params in param_dicts]

    out_fpath = f'{out_dir}/postprocess.h5'
    postprocess(out_fpath, in_fpaths, num_proc=clargs.num_proc)

    if clargs.export_xdmf:
        for param in params:
            in_fpath = f'{out_dir}/{param.to_str()}.h5'
            xdmf_path = (
                f"vis--vcov{param['vcov']:.2e}--mcov{param['mcov']:.2e}"
                f"--distribution{param['SwellingDistribution']:s}.xdmf"
            )
            # xdmf_path = 'temp.xdmf'

            model = setup_model(param)
            postprocess_xdmf(
                model, param, xdmf_path
            )
            # if not path.isfile(out_fpath):
            # else:
            #     print(f"Skipping XDMF export of existing file {out_fpath}")