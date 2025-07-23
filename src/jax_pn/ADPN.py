import jax 
import jax.experimental
import jax.numpy as jnp
import numpy as np
import basix
from scipy.sparse import lil_matrix
import scipy.sparse 

from typing import List, Iterable, Literal, Union, Dict
from numpy.polynomial.legendre import legval
from scipy.special import legendre
from numpy.polynomial.legendre import leggauss
from .FEM1D import create_dof_matrix_vertex_interior
from .PN import legendre_coeff_matrix, PN_Problem
from .Neutron import Neutron_Problem, interpolate_solution
from functools import cached_property
from dataclasses import dataclass



@dataclass(frozen=True)
class GlobalSettings:
    n_local_dofs: int     # dof per element
    n_moments: int        # number of moments (N_max + 1)
    n_global_dofs: int    # total dofs in the problem per moment
    n_elements: int       # number of elements in the mesh
    left_dof: int         # degree of freedom for the left boundary condition (usually 0)
    right_dof: int        # degree of freedom for the right boundary condition (in basix element ordering, this is the second node of the last element and has index len(nodes) - 1)
    n_dofs_per_eg : int   # includes additional boundary conditions
    n_energy_groups : int # number of energy groups in the problem

@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class MatrixSettings:
    elem_dof_matrix : jnp.ndarray # local dof matrix for each element, shape (n_elements, n_local_dofs)
    local_streaming : jnp.ndarray # local streaming matrix, shape (n_local_dofs, n_local_dofs)
    mass_matrix     : jnp.ndarray # local mass matrix, shape (n_local_dofs, n_local_dofs)
    leg_coeff_left  : jnp.ndarray # Legendre coefficients for the left boundary condition, shape (n_moments, n_moments)
    leg_coeff_right : jnp.ndarray # Legendre coefficients for the right boundary condition, shape (n_moments, n_moments)

#@jax.tree_util.register_pytree_node_class

# =============================================================================================================================================================================================
# |                                                                                Matrix Assembly Functions                                                                                  |
# =============================================================================================================================================================================================

def add_block_if(cond, block_fn, fallback_shape, dtypefloat, dtypeint):
    """Helper to conditionally add a block to the matrix.
       Else falls back to adding zeros at the (0,0) position"""
    return jax.lax.cond(
        cond,
        lambda _: block_fn(),
        lambda _: (jnp.zeros(fallback_shape, dtype=dtypeint),   # rows
                   jnp.zeros(fallback_shape, dtype=dtypeint),   # cols
                   jnp.zeros(fallback_shape, dtype=dtypefloat)),# vals
        operand=None
    )



def local_matrix_PN_single_g(
        moment_k : int,
        elem_i   : int, 
        global_settings : GlobalSettings, 
        matrix_settings : Dict,
        parameters : Dict):  
    
    n_local_dofs    = global_settings.n_local_dofs
    n_moments       = global_settings.n_moments
    n_global_dofs   = global_settings.n_global_dofs

    elem_dof_matrix = matrix_settings.elem_dof_matrix
    local_streaming = matrix_settings.local_streaming
    mass_matrix     = matrix_settings.mass_matrix    

    sigma_t_i       = parameters['sigma_t_i']
    sigma_s_k_i_gg  = parameters['sigma_s_k_i_gg']
    h_i             = parameters['h_i']
    q_i_k_j         = parameters['q_i_k_j']

    img, jmg = jnp.meshgrid(jnp.arange(n_local_dofs), jnp.arange(n_local_dofs), indexing='ij')

    # i is row, j is column (row = equation number, column = dof number)
    i = img.flatten()
    j = jmg.flatten()

    spatial_global_i = elem_dof_matrix[elem_i, i]
    spatial_global_j = elem_dof_matrix[elem_i, j]

    total_matrix_entries_local = n_local_dofs * n_local_dofs 

    def global_index(k_value, i):
        return  k_value * n_global_dofs + i


    def assemble_block(k_row, k_col, block_values):
        row_idx = global_index(k_row, spatial_global_i)
        col_idx = global_index(k_col, spatial_global_j)
        return row_idx, col_idx, block_values.flatten()
    

    def add_km1_block():        
        local_block = (moment_k / (2 * moment_k + 1)) * local_streaming
        return assemble_block(moment_k, moment_k - 1, local_block)
    
    def add_kp1_block():
        local_block = (moment_k + 1) / (2 * moment_k + 1) * local_streaming
        return assemble_block(moment_k, moment_k + 1, local_block)
    
    def add_mass_block():
        local_block = mass_matrix * (sigma_t_i[elem_i] - sigma_s_k_i_gg[elem_i, moment_k]) * h_i[elem_i]
        return assemble_block(moment_k, moment_k, local_block)


    rows_km1, cols_km1, vals_km1 = add_block_if(moment_k > 0, add_km1_block, fallback_shape=(total_matrix_entries_local,), dtypeint = img.dtype, dtypefloat=h_i.dtype)

    rows_kp1, cols_kp1, vals_kp1 = add_block_if(moment_k < n_moments - 1, add_kp1_block, fallback_shape=(total_matrix_entries_local,), dtypeint = img.dtype, dtypefloat=h_i.dtype)

    rows_mass, cols_mass, vals_mass = add_mass_block()

    b_values_j = (mass_matrix * h_i[elem_i]) @ q_i_k_j[elem_i, moment_k, :]

    row_b = global_index(moment_k, elem_dof_matrix[elem_i, jnp.arange(n_local_dofs)])


    rows = jnp.concatenate([rows_km1, rows_kp1, rows_mass])
    cols = jnp.concatenate([cols_km1, cols_kp1, cols_mass])
    vals = jnp.concatenate([vals_km1, vals_kp1, vals_mass])      
    
    return rows, cols, vals, row_b, b_values_j

def local_matrix_PN_scatter(moment_k : int,
        elem_i :int,
        global_settings : GlobalSettings, 
        matrix_settings : Dict,
        parameters : Dict):        
    '''
    Assembles the local matrix for a single element for the scattering term. This does 
    take into account the dof numbering in the single group, but the global group numbering has to be added separately. (i.e. row += g_out * dofs_per_eg, col = gin * dofs_per_eg )    
    '''
    n_local_dofs    = global_settings.n_local_dofs    
    n_global_dofs   = global_settings.n_global_dofs

    elem_dof_matrix = matrix_settings.elem_dof_matrix
    mass_matrix     = matrix_settings.mass_matrix
    
    sigma_s_k_i_gg  = parameters['sigma_s_k_i_gg']
    h_i             = parameters['h_i']

    img, jmg = jnp.meshgrid(jnp.arange(n_local_dofs), jnp.arange(n_local_dofs), indexing='ij')

    # i is row, j is column (row = equation number, column = dof number)
    i = img.flatten()
    j = jmg.flatten()

    spatial_global_i = elem_dof_matrix[elem_i, i]
    spatial_global_j = elem_dof_matrix[elem_i, j]

    total_matrix_entries_local = n_local_dofs * n_local_dofs 

    def global_index(k_value, i):
        return  k_value * n_global_dofs + i

    def assemble_block(k_row, k_col, block_values):
        row_idx = global_index(k_row, spatial_global_i)
        col_idx = global_index(k_col, spatial_global_j)
        return row_idx, col_idx, block_values.flatten()    
    
    def add_mass_block():
        local_block = mass_matrix * ( - sigma_s_k_i_gg[elem_i, moment_k]) * h_i[elem_i]
        return assemble_block(moment_k, moment_k, local_block)

    return add_mass_block()


def _append_marshak_boundary_conditions(global_settings : GlobalSettings, matrix_settings : Dict):
        
    n_moments       = global_settings.n_moments
    n_global_dofs   = global_settings.n_global_dofs
    left_dof        = global_settings.left_dof
    right_dof       = global_settings.right_dof

    leg_coeff_left  = matrix_settings.leg_coeff_left
    leg_coeff_right = matrix_settings.leg_coeff_right    

    bc_offset =n_global_dofs * n_moments 
    
    Vcc_rows = []
    Vcc_cols = []
    Vcc_vals = []
    bcc_rows = []
    bcc_vals = []

    i_bc = 0 
    
    for enforce_i in range(1, n_moments, 2): # number of boundary conditions = group * len(range(1, settings.n_moments, 2))            
        for l in range(n_moments):                                
            index_left_dof_group = l * n_global_dofs + left_dof
            index_right_dof_group = l * n_global_dofs + right_dof

            # VCC contribution
            Vcc_rows.append(bc_offset + i_bc + 0)
            Vcc_cols.append(index_left_dof_group)
            Vcc_vals.append(leg_coeff_left[enforce_i, l] * (2 * l + 1))

            Vcc_rows.append(bc_offset + i_bc + 1)
            Vcc_cols.append(index_right_dof_group)
            Vcc_vals.append(leg_coeff_right[enforce_i, l] * (2 * l + 1))

            # VCC^T contribution
            Vcc_rows.append(index_left_dof_group)
            Vcc_cols.append(bc_offset + i_bc + 0)
            Vcc_vals.append(leg_coeff_left[enforce_i, l] * (2 * l + 1))                

            Vcc_rows.append(index_right_dof_group)
            Vcc_cols.append(bc_offset + i_bc + 1)
            Vcc_vals.append(leg_coeff_right[enforce_i, l] * (2 * l + 1))
            
            bcc_rows.append(bc_offset + i_bc + 0)
            bcc_vals.append(0.0)
            bcc_rows.append(bc_offset + i_bc + 1)
            bcc_vals.append(0.0)
        i_bc += 2    
    return jnp.array(Vcc_rows), jnp.array(Vcc_cols), jnp.array(Vcc_vals) , jnp.array(bcc_rows), jnp.array(bcc_vals)


def _append_reflective_boundary_conditions(global_settings : Dict, matrix_settings : MatrixSettings):

    n_moments       = global_settings.n_moments
    n_global_dofs   = global_settings.n_global_dofs
    left_dof        = global_settings.left_dof
    right_dof       = global_settings.right_dof

    bc_offset =n_global_dofs * n_moments 
    
    Vcc_rows = []
    Vcc_cols = []
    Vcc_vals = []
    bcc_rows = []
    bcc_vals = []

    i_bc = 0 
    
    for enforce_i in range(1, n_moments, 2): # number of boundary conditions = group * len(range(1, settings.n_moments, 2))            
        
        index_left_dof_group = enforce_i * n_global_dofs + left_dof
        index_right_dof_group = enforce_i * n_global_dofs + right_dof

        Vcc_rows.append(bc_offset + i_bc + 0)
        Vcc_cols.append(index_left_dof_group)
        Vcc_vals.append(1.0)

        Vcc_rows.append(bc_offset + i_bc + 1)
        Vcc_cols.append(index_right_dof_group)
        Vcc_vals.append(1.0)

        Vcc_rows.append(index_left_dof_group)
        Vcc_cols.append(bc_offset + i_bc + 0)
        Vcc_vals.append(1.0)                

        Vcc_rows.append(index_right_dof_group)
        Vcc_cols.append(bc_offset + i_bc + 1)
        Vcc_vals.append(1.0)
        
        bcc_rows.append(bc_offset + i_bc + 0)
        bcc_vals.append(0.0)
        bcc_rows.append(bc_offset + i_bc + 1)
        bcc_vals.append(0.0)
        i_bc += 2    
    return jnp.array(Vcc_rows), jnp.array(Vcc_cols), jnp.array(Vcc_vals) , jnp.array(bcc_rows), jnp.array(bcc_vals)


def total_matrix_assembly_vacuum_bcs_single_g(        
        global_settings : Dict,
        matrix_settings : Dict,
        parameters : Dict        
        ):                
        moments = jnp.arange(global_settings.n_moments)
        elems   = jnp.arange(global_settings.n_elements)                
        rows_all, cols_all, vals_all, rows_b, vals_b      = vmap_local_matrix_PN_single_g(moments, elems, global_settings, matrix_settings, parameters)                                                

        Vcc_rows, Vcc_cols, Vcc_vals , bcc_rows, bcc_vals = marshak_jit(global_settings, matrix_settings)                            

        vals_np = jnp.concatenate([vals_all.ravel(), Vcc_vals])        
        rows_np = jnp.concatenate([rows_all.ravel(), Vcc_rows])    
        cols_np = jnp.concatenate([cols_all.ravel(), Vcc_cols])        

        vals_b_np = jnp.concatenate([vals_b.ravel(), bcc_vals])
        rows_b_np = jnp.concatenate([rows_b.ravel(), bcc_rows])
        
        return vals_np, rows_np, cols_np, vals_b_np, rows_b_np#



def total_matrix_assembly_reflective_bcs_single_g(
        global_settings : Dict,
        matrix_settings : Dict,
        parameters : Dict        
        ):
                
        moments = jnp.arange(global_settings.n_moments)
        elems   = jnp.arange(global_settings.n_elements)        
        
        rows_all, cols_all, vals_all, rows_b, vals_b      = vmap_local_matrix_PN_single_g(moments, elems, global_settings, matrix_settings, parameters)                                                

        Vcc_rows, Vcc_cols, Vcc_vals , bcc_rows, bcc_vals = reflective_jit(global_settings, matrix_settings)                            

        vals_np = jnp.concatenate([vals_all.ravel(), Vcc_vals])        
        rows_np = jnp.concatenate([rows_all.ravel(), Vcc_rows])    
        cols_np = jnp.concatenate([cols_all.ravel(), Vcc_cols])        

        vals_b_np = jnp.concatenate([vals_b.ravel(), bcc_vals])
        rows_b_np = jnp.concatenate([rows_b.ravel(), bcc_rows])
        
        return vals_np, rows_np, cols_np, vals_b_np, rows_b_np#


def total_downscatter_matrix_assembly(
        global_settings : GlobalSettings,
        matrix_settings : Dict,
        parameters : Dict):
        moments = jnp.arange(global_settings.n_moments)
        elems   = jnp.arange(global_settings.n_elements)

        rows_all, cols_all, vals_all = vmap_local_matrix_PN_scatter(moments, elems, global_settings, matrix_settings, parameters)
        return vals_all.flatten(), rows_all.flatten(), cols_all.flatten()

## Jitted versions
marshak_jit                                        = jax.jit(_append_marshak_boundary_conditions, static_argnums=(0)) # static argnum is number of moments, which is required for JAX to compile the loops.
reflective_jit                                     = jax.jit(_append_reflective_boundary_conditions, static_argnums=(0)) # static argnum is number of moments, which is required for JAX to compile the loops.
local_matrix_PN_single_g_jit                       = jax.jit(local_matrix_PN_single_g, static_argnums= (2))     # static argnums are n_local_dofs and n_moments, which are required for JAX to compile the loops and allocate arrays
vmap_local_matrix_PN_single_g                      = jax.jit(jax.vmap(jax.vmap(local_matrix_PN_single_g_jit, in_axes=(0, None, None, None, None)), in_axes=(None, 0, None, None, None)), static_argnums=(2)) # static argnums are n_local_dofs and n_moments, which are required for JAX to compile the loops and allocate arrays
total_matrix_assembly_vacuum_bcs_single_g_jit      = jax.jit(total_matrix_assembly_vacuum_bcs_single_g, static_argnums=(0)) # static argnums are n_local_dofs and n_moments, which are required for JAX to compile the loops and allocate arrays
total_matrix_assembly_reflective_bcs_single_g_jit  = jax.jit(total_matrix_assembly_reflective_bcs_single_g, static_argnums=(0)) # static argnums are n_local_dofs and n_moments, which are required for JAX to compile the loops and allocate arrays
total_downscatter_matrix_assembly_jit              = jax.jit(total_downscatter_matrix_assembly, static_argnums=(0)) # static argnums are n_local_dofs and n_moments, which are required for JAX to compile the loops and allocate arrays

local_matrix_PN_scatter_jit  = jax.jit(local_matrix_PN_scatter, static_argnums=(2)) # static argnums are n_local_dofs and n_moments, which are required for JAX to compile the loops and allocate arrays
vmap_local_matrix_PN_scatter = jax.jit(jax.vmap(jax.vmap(local_matrix_PN_scatter_jit, in_axes=(0, None, None, None, None)), in_axes=(None, 0, None, None, None)), static_argnums=(2)) # static argnums are n_local_dofs and n_moments, which are required for JAX to compile the loops and allocate arrays



def assemble_multigroup_system(global_settings : GlobalSettings,matrix_settings : MatrixSettings, parameters : Dict):     # same as Neutron_Problem.assemble_multigroup_system, but without global state.
    
    total_dofs = global_settings.n_dofs_per_eg * global_settings.n_energy_groups        

    Arows = []
    Acols = []
    Adata = []

    brows = [] 
    bdata = []

    for i in range(global_settings.n_energy_groups):

        start_row =       i * global_settings.n_dofs_per_eg
                    
        # Block-diagonal (single energy group)
        parameters_eg =  {
            'sigma_t_i'       : parameters['sigma_t_i'][:, i],
            'sigma_s_k_i_gg'  : parameters['sigma_s_k_i_gg'][:, :, i, i],
            'h_i'             : parameters['h_i'],
            'q_i_k_j'         : parameters['q_i_k_j'][:, :, :, i]
        }
        acoo_data, acoo_row, acoo_col, bcoo_data, bcoo_row = total_matrix_assembly_vacuum_bcs_single_g_jit(global_settings, matrix_settings, parameters_eg)
        diag_col = start_row

        Arows.append(acoo_row + start_row)
        Acols.append(acoo_col + diag_col)
        Adata.append(acoo_data)
        
        brows.append(bcoo_row + start_row)
        bdata.append(bcoo_data)
        
        for g_in in range(i):
            parameters_eg_ds =  {
                'sigma_t_i'       : parameters['sigma_t_i'][:, i], # does not matter
                'sigma_s_k_i_gg'  : parameters['sigma_s_k_i_gg'][:, :, i, g_in],
                'h_i'             : parameters['h_i'], 
                'q_i_k_j'         : parameters['q_i_k_j'][:, :, :, i] # does not matter
            }
            D_data, D_row, D_col = total_downscatter_matrix_assembly_jit(global_settings, matrix_settings, parameters_eg_ds )
    
            Arows.append(D_row + start_row)
            Acols.append(D_col + g_in * global_settings.n_dofs_per_eg)
            Adata.append(D_data)

    rows = np.concatenate(Arows)
    cols = np.concatenate(Acols)
    data = np.concatenate(Adata)
    brow = np.concatenate(brows)
    bdata = np.concatenate(bdata)

    # Build sparse matrix and vector
    A_total = scipy.sparse.coo_matrix((data, (rows, cols)), shape=(total_dofs, total_dofs))
    b_total = scipy.sparse.coo_matrix((bdata, (brow, np.zeros_like(brow))), shape=(total_dofs, 1))

    return A_total, b_total

# =============================================================================================================================================================================================
# |                                                                               Residual Functions                                                                                          |
# =============================================================================================================================================================================================

def local_residual_eg(
        energy_group_g : int, 
        moment_k : int,
        elem_i   : int, 
        global_settings : GlobalSettings, 
        matrix_settings : Dict,
        parameters : Dict,
        solution   : jnp.ndarray):  
    
    n_local_dofs    = global_settings.n_local_dofs
    n_moments       = global_settings.n_moments
    n_global_dofs   = global_settings.n_global_dofs

    n_dofs_per_eg   = global_settings.n_dofs_per_eg     

    elem_dof_matrix = matrix_settings.elem_dof_matrix
    local_streaming = matrix_settings.local_streaming
    mass_matrix     = matrix_settings.mass_matrix

    leg_coeff_left  = matrix_settings.leg_coeff_left
    leg_coeff_right = matrix_settings.leg_coeff_right

    sigma_t_i       = parameters['sigma_t_i']
    sigma_s_k_i_gg  = parameters['sigma_s_k_i_gg']
    h_i             = parameters['h_i']
    q_i_k_j         = parameters['q_i_k_j']

    j = jnp.arange(n_local_dofs)
    spatial_global_i = elem_dof_matrix[elem_i, j]     
    
    def global_index(energy_group_g, k_value, i):
        return  n_dofs_per_eg * energy_group_g + k_value * n_global_dofs + i
    
    def add_zero(residual):
        return residual
    
    def add_minus_one(residual):
        indices_i_km1 = global_index(energy_group_g, moment_k - 1, spatial_global_i)
        solution_km1 = solution[indices_i_km1]
        residual_km1 = moment_k / (2 * moment_k + 1) * (local_streaming @ solution_km1)
        residual = residual.at[:].add(residual_km1)
        return residual
    
    def add_plus_one(residual):
        indices_i_kp1 = global_index(energy_group_g, moment_k + 1, spatial_global_i)
        solution_kp1 = solution[indices_i_kp1]
        residual_kp1 = (moment_k + 1) / (2 * moment_k + 1) * (local_streaming @ solution_kp1)
        residual = residual.at[:].add(residual_kp1)
        return residual

    def left_dof_bc_vacuum(residual):
        enforce_is = jnp.arange(1,n_moments,2) # number of left boundary conditions
        start_idx  = n_dofs_per_eg * energy_group_g + n_global_dofs * n_moments
        indices    = jnp.arange(len(enforce_is)) * 2 + start_idx
        lagrange_multipliers = solution[indices] 
        # left dof is first node of first element, so we add it at location 0 (MAGIC NUMBER)    
        return residual.at[0].add( jnp.sum(leg_coeff_left[enforce_is, moment_k] * (2 * moment_k + 1) * lagrange_multipliers))
    
    def right_dof_bc_vacuum(residual):        
        enforce_is = jnp.arange(1,n_moments,2) # number of right boundary conditions
        start_idx  = n_dofs_per_eg * energy_group_g + n_global_dofs * n_moments + 1
        indices    = jnp.arange(len(enforce_is)) * 2 + start_idx
        lagrange_multipliers = solution[indices]                          
        # right dof is second node of last element, so we add it at location 1  (MAGIC NUMBER)    
        return residual.at[1].add( jnp.sum(leg_coeff_right[enforce_is, moment_k] * (2 * moment_k + 1) * lagrange_multipliers))
    
    indices_ik = global_index(energy_group_g,moment_k, spatial_global_i)    
    #residual = (mass_matrix @ solution[indices_ik]) * (sigma_t_i[elem_i, energy_group_g] - sigma_s_k_i_gg[elem_i, moment_k, energy_group_g, energy_group_g]) * h_i[elem_i]
    residual = (mass_matrix @ solution[indices_ik]) * (sigma_t_i[elem_i, energy_group_g]) * h_i[elem_i]

    residual = jax.lax.cond(moment_k > 0,              add_minus_one, add_zero, residual) # if moment_k > 0,     add the k - 1 block, else do nothing
    residual = jax.lax.cond(moment_k < n_moments - 1,  add_plus_one, add_zero,  residual) # if moment_k < N_max, add the k + 1 block, else do nothing    

    residual = jax.lax.cond( (elem_i == 0) ,                              left_dof_bc_vacuum, add_zero, residual)  # if elem_i == 0, apply left BC
    residual = jax.lax.cond( (elem_i == global_settings.n_elements - 1), right_dof_bc_vacuum, add_zero, residual)  # if elem_i == n_elements, apply right BC

    # Downcatter Terms (not effecient: goes beyond the maximum L order in the scatter matrix):
    # upscatter could be trivially used as well by extending the loop to include all energy groups (and ensure you remove the contribution for the current energy group g)
    def scatter_contribution(g, acc):
        indices_i_k = global_index(g, moment_k, spatial_global_i)
        term = -(mass_matrix @ solution[indices_i_k]) * sigma_s_k_i_gg[elem_i, moment_k, energy_group_g, g] * h_i[elem_i]
        return acc + term
    
    # upscatter included to have a static fori loop
    residual   = jax.lax.fori_loop(0, global_settings.n_energy_groups, scatter_contribution, residual)    
    b_values_j = (mass_matrix * h_i[elem_i]) @ q_i_k_j[elem_i, moment_k, :, energy_group_g]

    return residual - b_values_j

def residual_bc_marshak_eg(energy_group_g : int, global_settings : GlobalSettings, matrix_settings : Dict, solution : jnp.ndarray):
    
    enforced_l = jnp.arange(1, global_settings.n_moments, 2) # number of left boundary conditions
    residual   = jnp.zeros(global_settings.n_moments)

    dof_eg_offset = energy_group_g * global_settings.n_dofs_per_eg    
    
    all_l = jnp.arange(global_settings.n_moments)

    start_idx = dof_eg_offset
    n_solution_l = jnp.arange(global_settings.n_moments) * global_settings.n_global_dofs    

    solution_left_indices = start_idx + global_settings.left_dof + n_solution_l
    solution_right_indices = start_idx + global_settings.right_dof + n_solution_l

    solution_left_all_l  = solution[solution_left_indices]
    solution_right_all_l = solution[solution_right_indices]

    residual = residual.at[::2].add( jnp.sum(matrix_settings.leg_coeff_left[enforced_l, :]   * (2 * all_l[None, :] + 1) * solution_left_all_l[None,:] , axis=1))
    residual = residual.at[1::2].add( jnp.sum(matrix_settings.leg_coeff_right[enforced_l, :] * (2 * all_l[None, :] + 1) * solution_right_all_l[None,:] , axis=1))
    
    return residual

vmap_local_residual_PN_eg = jax.jit(jax.vmap(jax.vmap(jax.vmap(local_residual_eg, in_axes=(0, None, None, None, None, None, None)), in_axes=(None, 0, None, None, None, None, None)), in_axes=(None, None, 0, None, None, None, None)), static_argnums=(3)) # static argnums are n_local_dofs and n_moments, which are required for JAX to compile the loops and allocate arrays
residual_bc_marshak_eg_jit = jax.jit(residual_bc_marshak_eg, static_argnums=(1)) # static argnum is global_settings, which is required for JAX to compile the loops and allocate arrays


def residualPN(global_settings : GlobalSettings, matrix_settings, parameters_eg, solution):    
    moments = jnp.arange(global_settings.n_moments)
    elems   = jnp.arange(global_settings.n_elements)
    
    n_moments     = global_settings.n_moments    
    n_global_dofs = global_settings.n_global_dofs
    n_groups      = global_settings.n_energy_groups

    eg = jnp.arange(n_groups)

    local_residuals = vmap_local_residual_PN_eg(
        eg,
        moments,
        elems,
        global_settings,
        matrix_settings,
        parameters_eg,
        solution
    )    
    global_residual = jnp.zeros_like(solution)

    for g in range(n_groups):
        
        offset_g = global_settings.n_dofs_per_eg * g
        for k in range(n_moments):            
            offset_k = offset_g + global_settings.n_global_dofs * k

            global_dof_indices = matrix_settings.elem_dof_matrix + offset_k
            global_residual    = global_residual.at[global_dof_indices].add(local_residuals[:, k, g, :])
        
        bcs = residual_bc_marshak_eg_jit(
            g,
            global_settings,
            matrix_settings,            
            solution,            
        )    
        global_residual = global_residual.at[offset_g + global_settings.n_global_dofs * n_moments : offset_g + global_settings.n_global_dofs * n_moments + n_moments].add(bcs)

    return global_residual

residualPN_jit = jax.jit(residualPN, static_argnums=(0,)) # static argnum is global_settings, which is required for JAX to compile the loops and allocate arrays


class ADPN_Problem(PN_Problem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)        
        
        # Making sure all arrays are JAX compatible for the matrix assembly
        self.jax_n_groups         = self.sigma_s.shape[-1]
        self.jax_n_moments        = self.N_max + 1
        self.jax_n_global_dofs    = self.n_global_dofs
        self.jax_n_elements       = len(self.nodes) - 1
        self.jax_n_local_dofs     = self.dof_matrix.shape[1]
        self.jax_elem_dof_matrix  = jnp.array(self.dof_matrix)
        self.jax_mass_matrix      = jnp.array(self.mass_matrix)
        self.jax_local_streaming  = jnp.array(self.local_streaming)
        self.jax_nodes            = jnp.array(self.nodes)
        self.jax_elems            = jnp.arange(self.jax_n_elements)
        self.jax_moments          = jnp.arange(self.jax_n_moments)

        self.jax_sigma_t          = jnp.array(self.sigma_t)
        self.jax_sigma_s          = jnp.array(self.sigma_s)
        self.jax_h_i              = jnp.array(self.nodes[1:] - self.nodes[:-1])
        self.jax_q_i_k_j          = jnp.array(self.q)

        self.jax_left_coeff_matrix  = jnp.array(legendre_coeff_matrix(self.jax_n_moments,  0, 1))
        self.jax_right_coeff_matrix = jnp.array(legendre_coeff_matrix(self.jax_n_moments, -1, 0))

        self.global_settings = GlobalSettings(**{'n_local_dofs'    : self.jax_n_local_dofs,
            'n_moments'       : self.jax_n_moments,
            'n_global_dofs'   : self.jax_n_global_dofs,
            'n_elements'      : self.jax_n_elements,
            'left_dof'        : 0,
            'right_dof'       : len(self.jax_nodes) - 1,            
            'n_dofs_per_eg'   : self.dofs_per_eg,
            "n_energy_groups" : self.jax_n_groups
        })
        self.matrix_settings = MatrixSettings(**{
            'elem_dof_matrix' : self.jax_elem_dof_matrix,
            'local_streaming' : self.jax_local_streaming,
            'mass_matrix'     : self.jax_mass_matrix,
            'leg_coeff_left'  : self.jax_left_coeff_matrix,
            'leg_coeff_right' : self.jax_right_coeff_matrix
        })

        self.parameters = {
            'sigma_t_i'       : self.jax_sigma_t,
            'sigma_s_k_i_gg'  : self.jax_sigma_s,
            'h_i'             : self.jax_h_i,
            'q_i_k_j'         : self.jax_q_i_k_j
        }

                    

    def set_additional_BC_equations_per_eg(self):
        return self.N_max + 1
        
    def Assemble_Single_Energy_Group(self, energy_group : int, bc : Literal["vacuum"]):
        """
        Assemble the DPN finite element matrix and right-hand side vector for a single energy group.
        
        Parameters:
        -----------
        energy_group: int
            The index of the energy group to assemble.        
        
        Returns:
        --------
        A: jax.experimental.sparse.BCOO
            The assembled finite element matrix.
        b: jax.experimental.sparse.BCOO
            The right-hand side vector.
        """                     

        parameters_eg = {
            'sigma_t_i'       : self.jax_sigma_t[:, energy_group],
            'sigma_s_k_i_gg'  : self.jax_sigma_s[:, :, energy_group, energy_group],
            'h_i'             : self.jax_h_i,
            'q_i_k_j'         : self.jax_q_i_k_j[:, :, :, energy_group]
        }
        if bc == "vacuum":     
            vals_np, rows_np, cols_np, vals_b_np, rows_b_np = total_matrix_assembly_vacuum_bcs_single_g_jit(self.global_settings, self.matrix_settings, parameters_eg)
                      
        elif bc == "reflective":
            vals_np, rows_np, cols_np, vals_b_np, rows_b_np = total_matrix_assembly_reflective_bcs_single_g_jit(self.global_settings, self.matrix_settings, parameters_eg)
        else:
            raise ValueError(f"Unsupported boundary condition: {bc}. Only 'vacuum' is currently implemented.")
        
        return vals_np, rows_np, cols_np, (self.dofs_per_eg, self.dofs_per_eg), vals_b_np, rows_b_np, np.zeros_like(rows_b_np), (self.dofs_per_eg, 1) 
        
    def Assemble_Downscatter_Matrix(self, energy_group_out, energy_group_in):
        parameters_eg = {
            'sigma_t_i'       : self.jax_sigma_t[:, energy_group_out], # not actually used
            'sigma_s_k_i_gg'  : self.jax_sigma_s[:, :, energy_group_out, energy_group_in],
            'h_i'             : self.jax_h_i,                           
            'q_i_k_j'         : self.jax_q_i_k_j[:, :, :, energy_group_out] # not actually used
        }

        vals, rows, cols =  total_downscatter_matrix_assembly_jit(self.global_settings, self.matrix_settings, parameters_eg)
        return vals, rows, cols, (self.dofs_per_eg, self.dofs_per_eg)

    def assemble_multigroup_system(self, bc, n_energy_groups=None, parameters_eg = None):

        backup_params = {
            'sigma_t_i'       : jnp.copy(self.jax_sigma_t),
            'sigma_s_k_i_gg'  : jnp.copy(self.jax_sigma_s),
            'h_i'             : jnp.copy(self.jax_h_i),
            'q_i_k_j'         : jnp.copy(self.jax_q_i_k_j)
        }

        if parameters_eg is not None:
            self.jax_sigma_t = parameters_eg['sigma_t_i']
            self.jax_sigma_s = parameters_eg['sigma_s_k_i_gg']
            self.jax_h_i     = parameters_eg['h_i']
            self.jax_q_i_k_j = parameters_eg['q_i_k_j']
        


        result = super().assemble_multigroup_system(bc, n_energy_groups)

        self.jax_sigma_t = backup_params['sigma_t_i']
        self.jax_sigma_s = backup_params['sigma_s_k_i_gg']
        self.jax_h_i     = backup_params['h_i']     
        self.jax_q_i_k_j = backup_params['q_i_k_j']
        return result
