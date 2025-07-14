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
from .PN import legendre_coeff_matrix
from .Neutron import Neutron_Problem, interpolate_solution
from functools import cached_property



def add_block_if(cond, block_fn, fallback_shape):
    """Helper to conditionally add a block to the matrix.
       Else falls back to adding zeros at the (0,0) position"""
    return jax.lax.cond(
        cond,
        lambda _: block_fn(),
        lambda _: (jnp.zeros(fallback_shape, dtype=jnp.int64),   # rows
                   jnp.zeros(fallback_shape, dtype=jnp.int64),   # cols
                   jnp.zeros(fallback_shape, dtype=jnp.float64)),# vals
        operand=None
    )

def global_index_PN(    
    n_moments: int,
    n_global_dofs: int,
    group: Union[int, jnp.ndarray],
    moment: Union[int, jnp.ndarray],
    dof_index: Union[int, jnp.ndarray]
) -> jnp.ndarray:
    """
    Compute the global index for a given group, moment, and dof index.
    The first three parameters are scalar constants.
    The last three parameters can be scalars or JAX arrays (vectorized).

    Parameters
    ----------
    n_groups : int
        Total number of energy groups (scalar constant)
    n_moments : int
        Total number of angular moments (scalar constant)
    n_global_dofs : int
        Total number of global degrees of freedom per moment (spatial discretisation) (scalar constant)
    group : int or jax.numpy.ndarray
        Energy group index(es) - can be vectorized
    moment : int or jax.numpy.ndarray
        Angular moment index(es) - can be vectorized
    dof_index : int or jax.numpy.ndarray
        Degree of freedom index(es) - can be vectorized (global: not local, should already have been indexed by the element -> global dof matrix)

    Returns
    -------
    jax.numpy.ndarray
        Global index(es)
    """
    return group * (n_moments) * n_global_dofs + moment * n_global_dofs + dof_index

class NSettings:
    '''
    Class to aggregate settings for the AD-PN solver.
    Ensures all variables are jax compatible.
    '''
    def __init__(self, n_groups, n_moments, n_global_dofs, n_elements,
                 elem_dof_matrix, n_local_dofs, mass_matrix,
                 local_streaming, nodes):
        self.n_groups        = n_groups
        self.n_moments       = n_moments
        self.n_global_dofs   = n_global_dofs
        self.n_elements      = n_elements
        self.n_local_dofs    = n_local_dofs

        self.elem_dof_matrix = jnp.array(elem_dof_matrix)
        self.mass_matrix     = jnp.array(mass_matrix)
        self.local_streaming = jnp.array(local_streaming)
        self.nodes           = jnp.array(nodes)

        self.h_i = self.nodes[1:] - self.nodes[:-1]




def local_element_total_mat(settings : NSettings, group_g : int, moment_k : int, elem_i :int, sigma_t_i : jnp.ndarray, sigma_s_k_i_gg : jnp.ndarray, h_i : jnp.ndarray, q_i_k_j : jnp.ndarray):  
    '''
    Compute the local element matrix for a given mass matrix, local streaming, and cross-sections.

    Parameters
    ----------
    settings : NSettings
        Settings object containing all necessary parameters.
    group_g : int
        Energy group index.
    moment_k : int
        Angular moment index.
    elem_i : int
        Element index.
    sigma_t_i : jnp.ndarray
        Total cross-section for the element, shape (n_elements, n_groups)
    sigma_s_k_i_gg : jnp.ndarray
        Scattering cross-section for the element and moment, shape (n_elements, n_moments, n_groups (out), n_groups (in))
    h_i : jnp.ndarray   
        Element width, shape (n_elements,)
    q_i_k_j : jnp.ndarray
        Source term for the element, shape (n_elements, n_moments, n_local_dofs)

    Returns
    -------
    rows : jnp.ndarray
        Row indices for the sparse matrix, shape (total_matrix_entries_local,)
    cols : jnp.ndarray
        Column indices for the sparse matrix, shape (total_matrix_entries_local,)
    vals : jnp.ndarray
        Values for the sparse matrix, shape (total_matrix_entries_local,)
    row_b : jnp.ndarray
        Row indices for the source term vector, shape (n_local_dofs,)
    q_values_j : jnp.ndarray
        Values for the source term vector, shape (n_local_dofs,)
    '''
    

    img, jmg = jnp.meshgrid(jnp.arange(settings.n_local_dofs), jnp.arange(settings.n_local_dofs), indexing='ij')

    # i is row, j is column (row = equation number, column = dof number)
    i = img.flatten()
    j = jmg.flatten()

    spatial_global_i = settings.elem_dof_matrix[elem_i, i]
    spatial_global_j = settings.elem_dof_matrix[elem_i, j]

    total_matrix_entries_local = settings.n_local_dofs * settings.n_local_dofs 

    def assemble_block(k_row, k_col, block_values):
        row_idx = global_index_PN(settings.n_moments, settings.n_global_dofs, group_g, k_row, spatial_global_i)
        col_idx = global_index_PN(settings.n_moments, settings.n_global_dofs, group_g, k_col, spatial_global_j)
        return row_idx, col_idx, block_values.flatten()
    

    def add_km1_block():        
        local_block = (moment_k / (2 * moment_k + 1)) * settings.local_streaming
        return assemble_block(moment_k, moment_k - 1, local_block)
    
    def add_kp1_block():
        local_block = (moment_k + 1) / (2 * moment_k + 1) * settings.local_streaming
        return assemble_block(moment_k, moment_k + 1, local_block)
    
    def add_mass_block():
        local_block = settings.mass_matrix * (sigma_t_i[elem_i] - sigma_s_k_i_gg[elem_i, moment_k]) * h_i[elem_i]
        return assemble_block(moment_k, moment_k, local_block)


    rows_km1, cols_km1, vals_km1 = add_block_if(moment_k > 0, add_km1_block, fallback_shape=(total_matrix_entries_local,))

    rows_kp1, cols_kp1, vals_kp1 = add_block_if(moment_k < settings.n_moments - 1, add_kp1_block, fallback_shape=(total_matrix_entries_local,))

    rows_mass, cols_mass, vals_mass = add_mass_block()

    b_values_j = (settings.mass_matrix * h_i[elem_i]) @ q_i_k_j[elem_i, moment_k, :]

    row_b = global_index_PN(settings.n_moments, settings.n_global_dofs, group_g, moment_k, settings.elem_dof_matrix[elem_i, jnp.arange(settings.n_local_dofs)])


    rows = jnp.concatenate([rows_km1, rows_kp1, rows_mass])
    cols = jnp.concatenate([cols_km1, cols_kp1, cols_mass])
    vals = jnp.concatenate([vals_km1, vals_kp1, vals_mass])      
    
    return rows, cols, vals, row_b, b_values_j


def _append_marshak_boundary_conditions(settings : NSettings, left_dof, right_dof):
    bc_offset = settings.n_global_dofs * settings.n_moments * settings.n_groups
    
    left_coeff_matrix  = legendre_coeff_matrix(settings.n_moments,  0, 1)
    right_coeff_matrix = legendre_coeff_matrix(settings.n_moments, -1, 0)
    
    Vcc_rows = []
    Vcc_cols = []
    Vcc_vals = []
    bcc_rows = []
    bcc_vals = []

    i_bc = 0 
    for group in range(settings.n_groups):
        for enforce_i in range(1, settings.n_moments, 2): # number of boundary conditions = group * len(range(1, settings.n_moments, 2))            
            for l in range(settings.n_moments):                                
                index_left_dof_group = global_index_PN(settings.n_moments, settings.n_global_dofs, group,  l, left_dof)
                index_right_dof_group = global_index_PN(settings.n_moments, settings.n_global_dofs, group, l, right_dof)

                Vcc_rows.append(bc_offset + i_bc + 0)
                Vcc_cols.append(index_left_dof_group)
                Vcc_vals.append(left_coeff_matrix[enforce_i, l] * (2 * l + 1))

                Vcc_rows.append(bc_offset + i_bc + 1)
                Vcc_cols.append(index_right_dof_group)
                Vcc_vals.append(right_coeff_matrix[enforce_i, l] * (2 * l + 1))

                Vcc_rows.append(index_left_dof_group)
                Vcc_cols.append(bc_offset + i_bc + 0)
                Vcc_vals.append(left_coeff_matrix[enforce_i, l] * (2 * l + 1))                

                Vcc_rows.append(index_right_dof_group)
                Vcc_cols.append(bc_offset + i_bc + 1)
                Vcc_vals.append(right_coeff_matrix[enforce_i, l] * (2 * l + 1))
                
                bcc_rows.append(bc_offset + i_bc + 0)
                bcc_vals.append(0.0)
                bcc_rows.append(bc_offset + i_bc + 1)
                bcc_vals.append(0.0)
            i_bc += 2

    return jnp.array(Vcc_rows), jnp.array(Vcc_cols), jnp.array(Vcc_vals) , jnp.array(bcc_rows), jnp.array(bcc_vals)




def Convert_to_BCOO(rows, cols, vals, rows_b, vals_b):
    rows_flat = rows.flatten()
    cols_flat = cols.flatten()
    coords_A = jnp.stack([rows_flat, cols_flat], axis=1)    

    n_eq = np.max(rows_flat) + 1
    n_dof = np.max(cols_flat) + 1    

    A = jax.experimental.sparse.BCOO((vals.flatten(), coords_A), shape = (n_eq, n_dof))

    vals_b_flat  = vals_b.flatten()
    rows_b_flat  = rows_b.flatten()
    cols_b_flat  = jnp.zeros_like(rows_b_flat)
    b = jax.experimental.sparse.BCOO((vals_b_flat, jnp.stack([rows_b_flat, cols_b_flat], axis=1)), shape=(n_eq, 1))

    return A, b



class ADPN_Problem(Neutron_Problem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Create and assign nsettings attribute
        self.nsettings = NSettings(
            n_groups=self.sigma_s.shape[-1],
            n_moments=self.N_max + 1,
            n_global_dofs=self.n_global_dofs,
            n_elements=len(self.nodes) - 1,
            elem_dof_matrix=self.dof_matrix,
            n_local_dofs=self.dof_matrix.shape[1],
            mass_matrix=self.mass_matrix,
            local_streaming=self.local_streaming,
            nodes=self.nodes
        )
    def Assemble_Single_Energy_Group(self, energy_group, bc : Literal["vacuum"]):
        """
        Assemble the DPN finite element matrix and right-hand side vector for a single energy group.
        
        Parameters:
        -----------
        energy_group: int
            The index of the energy group to assemble.
        bc: Literal["vacuum"]
            Boundary condition to apply.
        
        Returns:
        --------
        A: scipy.sparse.lil_matrix
            The assembled finite element matrix.
        b: scipy.sparse.lil_matrix
            The right-hand side vector.
        """

        elems = jnp.arange(self.nsettings.n_elements)
        moments = jnp.arange(self.nsettings.n_moments)

        sigma_t_i      = jnp.array(self.sigma_t[elems, energy_group])
        sigma_s_k_i_gg = jnp.array(self.sigma_s[elems[:, None], moments[None, :], energy_group, energy_group])
        h_i            = self.nsettings.h_i[elems]
        q_i_k_j        = jnp.array(self.q[elems[:, None], moments[None, :], :, energy_group])

        total_elem_jit = jax.jit(local_element_total_mat, static_argnums=(0, 1))

        vectorized_elems = jax.vmap(jax.vmap(total_elem_jit, in_axes=(None, None, None, 0, None, None, None, None)), in_axes=(None, None, 0, None, None, None, None, None))

        rows_all, cols_all, vals_all, rows_b, vals_b = vectorized_elems(self.nsettings, energy_group, moments, elems, sigma_t_i, sigma_s_k_i_gg, h_i, q_i_k_j)

        A, b = Convert_to_BCOO(rows_all, cols_all, vals_all, rows_b, vals_b)

        return A,b

        #return assemble_PN_matrix(self.element, self.nodes, self.sigma_t[:, energy_group], self.sigma_s[:, :, energy_group, energy_group], self.q[:, :, :, energy_group], self.N_max, bc, self.L_scat)
    
    def set_dofs_per_eg(self):
        """
        Set the number of degrees of freedom per energy group.
        
        Returns:
        --------
        dofs_per_eg: int
            Number of degrees of freedom per energy group.
        """
        return self.n_global_dofs * (self.N_max + 1)
        
    def Assemble_Downscatter_Matrix(self, energy_group_out, energy_group_in):
        pass 
        #return Assemble_Downscatter_PN_Matrix(self.element, self.nodes, self.sigma_s[:, :, energy_group_out, energy_group_in], self.N_max, self.L_scat)
    
    def _get_single_spatial_solution(self, k, energy_group):
        """
        Get the single spatial solution for a given k, mu_sign, and energy group.
        
        Parameters:
        -----------
        k: int
            The k moment.
        mu_sign: int
            The sign of the cosine of the angle (1 or -1).
        energy_group: int
            The index of the energy group.
        
        Returns:
        --------
        np.ndarray
            The spatial solution for the given parameters.
        """
           
                
        return self.solution[self.dofs_per_eg * energy_group + k * self.n_global_dofs : self.dofs_per_eg * energy_group +   (k + 1) * self.n_global_dofs]
