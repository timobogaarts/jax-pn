import jax 
import jax.experimental
import jax.numpy as jnp
import numpy as np
import basix
from scipy.sparse import lil_matrix
import scipy.sparse
from typing import List, Iterable, Literal, Union
from numpy.polynomial.legendre import legval
from scipy.special import legendre
from numpy.polynomial.legendre import leggauss
from .FEM1D import create_dof_matrix_vertex_interior
from .PN import legendre_coeff_matrix, PN_Problem
from .Neutron import Neutron_Problem, interpolate_solution
from functools import cached_property



def add_block_if(cond, block_fn, fallback_shape):
    """Helper to conditionally add a block to the matrix.
       Else falls back to adding zeros at the (0,0) position"""
    return jax.lax.cond(
        cond,
        lambda _: block_fn(),
        lambda _: (jnp.zeros(fallback_shape, dtype=jnp.int32),   # rows
                   jnp.zeros(fallback_shape, dtype=jnp.int32),   # cols
                   jnp.zeros(fallback_shape, dtype=jnp.float32)),# vals
        operand=None
    )

def local_matrix_PN_single_g(
        moment_k : int,
        elem_i :int,
        n_local_dofs : int, 
        n_moments    : int, 
        n_global_dofs: int,
        elem_dof_matrix : jnp.ndarray,        
        local_streaming : jnp.ndarray,
        mass_matrix     : jnp.ndarray,                
        sigma_t_i : jnp.ndarray,
        sigma_s_k_i_gg : jnp.ndarray,
        h_i : jnp.ndarray, 
        q_i_k_j : jnp.ndarray):  
    
    
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


    rows_km1, cols_km1, vals_km1 = add_block_if(moment_k > 0, add_km1_block, fallback_shape=(total_matrix_entries_local,))

    rows_kp1, cols_kp1, vals_kp1 = add_block_if(moment_k < n_moments - 1, add_kp1_block, fallback_shape=(total_matrix_entries_local,))

    rows_mass, cols_mass, vals_mass = add_mass_block()

    b_values_j = (mass_matrix * h_i[elem_i]) @ q_i_k_j[elem_i, moment_k, :]

    row_b = global_index(moment_k, elem_dof_matrix[elem_i, jnp.arange(n_local_dofs)])


    rows = jnp.concatenate([rows_km1, rows_kp1, rows_mass])
    cols = jnp.concatenate([cols_km1, cols_kp1, cols_mass])
    vals = jnp.concatenate([vals_km1, vals_kp1, vals_mass])      
    
    return rows, cols, vals, row_b, b_values_j

def local_matrix_PN_scatter(moment_k : int,
        elem_i :int,
        n_local_dofs : int,          
        n_global_dofs: int,
        elem_dof_matrix : jnp.ndarray,                
        mass_matrix     : jnp.ndarray,                        
        sigma_s_k_i_gg : jnp.ndarray,
        h_i : jnp.ndarray
    ):
    '''
    Assembles the local matrix for a single element for the scattering term. This does 
    take into account the dof numbering in the single group, but the global group numbering has to be added separately. (i.e. row += g_out * dofs_per_eg, col = gin * dofs_per_eg )    
    '''

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


def _append_marshak_boundary_conditions(n_moments : int, n_global_dofs, left_dof, right_dof, leg_coeff_left, leg_coeff_right):
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

            Vcc_rows.append(bc_offset + i_bc + 0)
            Vcc_cols.append(index_left_dof_group)
            Vcc_vals.append(leg_coeff_left[enforce_i, l] * (2 * l + 1))

            Vcc_rows.append(bc_offset + i_bc + 1)
            Vcc_cols.append(index_right_dof_group)
            Vcc_vals.append(leg_coeff_right[enforce_i, l] * (2 * l + 1))

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


def _append_reflective_boundary_conditions(n_moments : int, n_global_dofs, left_dof, right_dof, leg_coeff_left, leg_coeff_right):
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
        moments : jnp.ndarray,
        elems : jnp.ndarray,
        n_local_dofs : int, 
        n_moments    : int, 
        n_global_dofs: int,
        elem_dof_matrix : jnp.ndarray,        
        local_streaming : jnp.ndarray,
        mass_matrix     : jnp.ndarray,                
        sigma_t_i : jnp.ndarray,
        sigma_s_k_i_gg : jnp.ndarray,
        h_i : jnp.ndarray, 
        q_i_k_j : jnp.ndarray,
        left_dof : int,
        right_dof : int,
        leg_coeff_left : jnp.ndarray,
        leg_coeff_right : jnp.ndarray
        ):
                
        total_entries = len(elems) * n_local_dofs * n_local_dofs * 3 * n_moments # this is the number of non-zero entries in the matrix: *not* the shape of the sparse matrix
        total_bcs     = 2 *  n_moments * n_moments  # this is the number of non-zero entries in the boundary condition matrix: *not* the shape of the sparse boundary condition matrix        
        
        rows_all, cols_all, vals_all, rows_b, vals_b      = vmap_local_matrix_PN_single_g(moments, elems, n_local_dofs, n_moments, n_global_dofs, elem_dof_matrix, local_streaming, mass_matrix, sigma_t_i, sigma_s_k_i_gg, h_i, q_i_k_j)                                                

        Vcc_rows, Vcc_cols, Vcc_vals , bcc_rows, bcc_vals = marshak_jit(n_moments, n_global_dofs, left_dof = left_dof, right_dof = right_dof, leg_coeff_left = leg_coeff_left, leg_coeff_right = leg_coeff_right)                            

        vals_np = jnp.concatenate([vals_all.ravel(), Vcc_vals])        
        rows_np = jnp.concatenate([rows_all.ravel(), Vcc_rows])    
        cols_np = jnp.concatenate([cols_all.ravel(), Vcc_cols])        

        vals_b_np = jnp.concatenate([vals_b.ravel(), bcc_vals])
        rows_b_np = jnp.concatenate([rows_b.ravel(), bcc_rows])
        
        return vals_np, rows_np, cols_np, vals_b_np, rows_b_np#



def total_matrix_assembly_reflective_bcs_single_g(
        moments : jnp.ndarray,
        elems : jnp.ndarray,
        n_local_dofs : int, 
        n_moments    : int, 
        n_global_dofs: int,
        elem_dof_matrix : jnp.ndarray,        
        local_streaming : jnp.ndarray,
        mass_matrix     : jnp.ndarray,                
        sigma_t_i : jnp.ndarray,
        sigma_s_k_i_gg : jnp.ndarray,
        h_i : jnp.ndarray, 
        q_i_k_j : jnp.ndarray,
        left_dof : int,
        right_dof : int,
        leg_coeff_left : jnp.ndarray,
        leg_coeff_right : jnp.ndarray
        ):
                
        total_entries = len(elems) * n_local_dofs * n_local_dofs * 3 * n_moments # this is the number of non-zero entries in the matrix: *not* the shape of the sparse matrix
        total_bcs     = 2 *  n_moments * n_moments  # this is the number of non-zero entries in the boundary condition matrix: *not* the shape of the sparse boundary condition matrix        
        
        rows_all, cols_all, vals_all, rows_b, vals_b      = vmap_local_matrix_PN_single_g(moments, elems, n_local_dofs, n_moments, n_global_dofs, elem_dof_matrix, local_streaming, mass_matrix, sigma_t_i, sigma_s_k_i_gg, h_i, q_i_k_j)                                                
        Vcc_rows, Vcc_cols, Vcc_vals , bcc_rows, bcc_vals = reflective_jit(n_moments, n_global_dofs, left_dof = left_dof, right_dof = right_dof, leg_coeff_left = leg_coeff_left, leg_coeff_right = leg_coeff_right)                            

        vals_np = jnp.concatenate([vals_all.ravel(), Vcc_vals])        
        rows_np = jnp.concatenate([rows_all.ravel(), Vcc_rows])    
        cols_np = jnp.concatenate([cols_all.ravel(), Vcc_cols])        

        vals_b_np = jnp.concatenate([vals_b.ravel(), bcc_vals])
        rows_b_np = jnp.concatenate([rows_b.ravel(), bcc_rows])
        
        return vals_np, rows_np, cols_np, vals_b_np, rows_b_np#


## Jitted versions
marshak_jit                                        = jax.jit(_append_marshak_boundary_conditions, static_argnums=(0)) # static argnum is number of moments, which is required for JAX to compile the loops.
reflective_jit                                     = jax.jit(_append_reflective_boundary_conditions, static_argnums=(0)) # static argnum is number of moments, which is required for JAX to compile the loops.
local_matrix_PN_single_g_jit                       = jax.jit(local_matrix_PN_single_g, static_argnums= (2,3))     # static argnums are n_local_dofs and n_moments, which are required for JAX to compile the loops and allocate arrays
vmap_local_matrix_PN_single_g                      = jax.jit(jax.vmap(jax.vmap(local_matrix_PN_single_g_jit, in_axes=(0, None, None, None, None, None, None, None, None, None, None, None)), in_axes=(None, 0, None, None, None, None, None, None, None, None, None, None)), static_argnums=(2,3)) # static argnums are n_local_dofs and n_moments, which are required for JAX to compile the loops and allocate arrays
total_matrix_assembly_vacuum_bcs_single_g_jit      = jax.jit(total_matrix_assembly_vacuum_bcs_single_g, static_argnums=(2, 3)) # static argnums are n_local_dofs and n_moments, which are required for JAX to compile the loops and allocate arrays
total_matrix_assembly_reflective_bcs_single_g_jit  = jax.jit(total_matrix_assembly_reflective_bcs_single_g, static_argnums=(2, 3)) # static argnums are n_local_dofs and n_moments, which are required for JAX to compile the loops and allocate arrays


local_matrix_PN_scatter_jit  = jax.jit(local_matrix_PN_scatter, static_argnums=(2)) # static argnums are n_local_dofs and n_moments, which are required for JAX to compile the loops and allocate arrays
vmap_local_matrix_PN_scatter = jax.jit(jax.vmap(jax.vmap(local_matrix_PN_scatter_jit, in_axes=(0, None, None, None, None, None, None, None)), in_axes=(None, 0, None, None, None, None, None, None)), static_argnums=(2)) # static argnums are n_local_dofs


class ADPN_Problem(PN_Problem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)        
        
        # Making sure all arrays are JAX compatible for the matrix assembly

        self.jax_n_groups = self.sigma_s.shape[-1]        
        self.jax_n_moments = self.N_max + 1
        self.jax_n_global_dofs = self.n_global_dofs
        self.jax_n_elements = len(self.nodes) - 1
        self.jax_n_local_dofs = self.dof_matrix.shape[1]
        self.jax_elem_dof_matrix = jnp.array(self.dof_matrix)
        self.jax_mass_matrix = jnp.array(self.mass_matrix)
        self.jax_local_streaming = jnp.array(self.local_streaming)
        self.jax_nodes = jnp.array(self.nodes)
        self.jax_elems = jnp.arange(self.jax_n_elements)
        self.jax_moments = jnp.arange(self.jax_n_moments)

        self.jax_sigma_t = jnp.array(self.sigma_t)
        self.jax_sigma_s = jnp.array(self.sigma_s)
        self.jax_h_i = jnp.array(self.nodes[1:] - self.nodes[:-1])
        self.jax_q_i_k_j = jnp.array(self.q)

        self.jax_left_coeff_matrix  = jnp.array(legendre_coeff_matrix(self.jax_n_moments,  0, 1))
        self.jax_right_coeff_matrix = jnp.array(legendre_coeff_matrix(self.jax_n_moments, -1, 0))
    


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
        if bc == "vacuum":     
            vals_np, rows_np, cols_np, vals_b_np, rows_b_np = total_matrix_assembly_vacuum_bcs_single_g_jit(
                    self.jax_moments, self.jax_elems, self.jax_n_local_dofs, self.jax_n_moments, self.jax_n_global_dofs,
                    self.jax_elem_dof_matrix, self.jax_local_streaming, self.jax_mass_matrix,
                    self.jax_sigma_t[:, energy_group], self.jax_sigma_s[:, :, energy_group, energy_group],
                    self.jax_h_i, self.jax_q_i_k_j[:, :, :, energy_group],
                    left_dof=0, right_dof=len(self.jax_nodes) - 1,
                    leg_coeff_left=self.jax_left_coeff_matrix,
                    leg_coeff_right=self.jax_right_coeff_matrix
                )      
        elif bc == "reflective":
            vals_np, rows_np, cols_np, vals_b_np, rows_b_np = total_matrix_assembly_reflective_bcs_single_g_jit(
                    self.jax_moments, self.jax_elems, self.jax_n_local_dofs, self.jax_n_moments, self.jax_n_global_dofs,
                    self.jax_elem_dof_matrix, self.jax_local_streaming, self.jax_mass_matrix,
                    self.jax_sigma_t[:, energy_group], self.jax_sigma_s[:, :, energy_group, energy_group],
                    self.jax_h_i, self.jax_q_i_k_j[:, :, :, energy_group],
                    left_dof=0, right_dof=len(self.jax_nodes) - 1,
                    leg_coeff_left=self.jax_left_coeff_matrix,
                    leg_coeff_right=self.jax_right_coeff_matrix
                )
        else:
            raise ValueError(f"Unsupported boundary condition: {bc}. Only 'vacuum' is currently implemented.")
        
        return vals_np, rows_np, cols_np, (self.dofs_per_eg, self.dofs_per_eg), vals_b_np, rows_b_np, np.zeros_like(rows_b_np), (self.dofs_per_eg, 1) 
        
    def Assemble_Downscatter_Matrix(self, energy_group_out, energy_group_in):
        rows, cols, vals = vmap_local_matrix_PN_scatter(self.jax_moments, self.jax_elems, self.jax_n_local_dofs, self.jax_n_global_dofs, self.jax_elem_dof_matrix, self.jax_mass_matrix, self.jax_sigma_s[:, :, energy_group_out, energy_group_in], self.jax_h_i)
        return vals.flatten(), rows.flatten(), cols.flatten(), (self.dofs_per_eg, self.dofs_per_eg)
