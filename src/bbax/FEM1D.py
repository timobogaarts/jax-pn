import jax 
import jax.numpy as jnp
import numpy as np
import basix
from scipy.sparse import lil_matrix
from typing import List, Iterable, Literal, Union
from numpy.polynomial.legendre import legval

def create_dof_matrix_vertex_interior(element, nodes):
    """
    Create a local to global mapping matrix for the given element and nodes.
    This function assumes the element is a 1D Lagrange element with interior nodes in the order:
    [v_0, v_1, i_0, i_1, ..., i_{num_interior-1}].

    The total global DOF matrix has as order:
    [v_0, v_1, ..., v_{num_vertices-1}, i_0, i_1, ..., i_{num_interior * n_elem - 1}]
    
    Parameters:
    -----------
    element: basix.Element
        The finite element to use for the mapping.
    nodes: np.ndarray
        Array of node positions in cm.  
    Returns:
    --------
    dof_matrix: np.ndarray
        Local to global mapping matrix of shape (n_elem, dof), where n_elem is the number of elements and dof is the number of degrees of freedom per element.
    num_global_dofs: int
        Total number of global degrees of freedom across all elements.  
    
        
    """    
    dof          = element.dim
    num_interior = dof - 2        
    num_vertices = len(nodes)
    n_elem       = num_vertices - 1

    dof_matrix = np.zeros((n_elem, dof), dtype=int)
    
    for e in range(n_elem):
        start_interior = num_vertices + e * num_interior
        v_0 = e
        v_1 = e + 1
        dof_matrix[e, :] = [v_0, v_1] + list(range(start_interior, start_interior + num_interior))
    
    return dof_matrix, num_interior * n_elem + num_vertices



def build_elements_and_materials(regions, elements_per_cm, N_max, energy_group, dof_elem):
    """
    Build the nodes, centers, and material properties for a 1D mesh based on the given regions.
    Parameters: 
    -----------
    regions: list of tuples (length, sigma_t, sigma_s, source)
        Each tuple defines a region with its length (cm), total cross-section (sigma_t), 
        scattering cross-section ([sigma_sk] for some maximum k order), and external source term (q) (in cm^-1).
    elements_per_cm: int
        Number of finite elements per centimeter.
    Returns:
    --------
    nodes: np.ndarray
        Array of node positions in cm.
    centers: np.ndarray
        Array of element center positions in cm.
    sigma_t: np.ndarray 
        Array of total cross-section values for each element in cm^-1.
    sigma_s: np.ndarray
        Array of scattering cross-section values for each element in cm^-1.
    q: np.ndarray
        Array of source term values for each element in cm^-1.
    """
    nodes = [0.0]
    sigma_t = []
    sigma_s = []
    q = []
    first = True
    for length, st, ss, src in regions:
        n_elem = int(round(length * elements_per_cm))
        region_nodes = np.linspace(nodes[-1], nodes[-1] + length, n_elem + 1)[1:]
        nodes.extend(region_nodes)
        sigma_t.extend([st[energy_group]] * n_elem)

        sigma_s_extended_nmax = np.zeros((N_max + 1))
        sigma_s_extended_nmax[:ss.shape[0]] = ss[:, energy_group, energy_group]
        
        sigma_s.extend(([sigma_s_extended_nmax]) * n_elem)
        q_total_matrix =  np.zeros((N_max + 1, dof_elem))
        q_total_matrix[0,:] = src[energy_group]  # Only zeroth order is non-zero
        q.extend([q_total_matrix] * n_elem)

    nodes = np.array(nodes)    
    sigma_t = np.array(sigma_t)    
    q = np.array(q)        
    
    return nodes, sigma_t, sigma_s, q

