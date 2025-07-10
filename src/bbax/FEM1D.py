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



def interpolate_solution(x_points, nodes, elem_dofs, solution_single_spatial, lagrange):
    """
    Interpolate the FEM solution at arbitrary points x_points.

    Parameters:
    -----------
    x_points: array(n_points)
        Points at which to interpolate the solution.
    nodes: array(n_nodes)
        Nodes of the FEM mesh.
    elem_dofs: array(n_elements, p+1)
        Local DOF indices for each element.
    solution: array(n_global_dofs)
        Global solution vector.
    lagrange: basix.Element
        Lagrange element used for interpolation.

    Is rather slow: but for 1D, it works okay.
    
    Returns: array of interpolated values at x_points
    """
    p = lagrange.degree
    values = np.zeros_like(x_points)
    
    # For each point, find which element it is in
    for i, x in enumerate(x_points):
        # Find the element containing x
        e = np.searchsorted(nodes, x) - 1
        if e < 0: e = 0
        if e >= len(elem_dofs): e = len(elem_dofs) - 1
        # Map x to reference coordinate xi in [0, 1]
        x0, x1 = nodes[e], nodes[e+1]
        xi = (x - x0) / (x1 - x0)
        # Tabulate basis at xi
        phi = lagrange.tabulate(0, np.array([[xi]]))[0, 0, :, 0]  # shape: (p+1,)
        # Get local DOF values
        u_local = solution_single_spatial[elem_dofs[e]]
        # Interpolate
        values[i] = np.dot(phi, u_local)
    return values


def build_multigroup_elements_and_materials(regions, elements_per_cm, N_max, dof_elem, energy_groups : int = None):
    """
    Build the nodes, centers, and material properties for a 1D mesh based on the given regions.
    Parameters: 
    -----------
    regions: list of tuples (length, sigma_t, sigma_s, source)
        Each tuple defines a region with its length (cm), total cross-section (sigma_t), 
        scattering cross-section ([sigma_k_gout_gin] for some maximum k order), and external source term (q) (in cm^-1).
    elements_per_cm: int
        Number of finite elements per centimeter.
    Returns:
    --------
    nodes: np.ndarray
        1D Array of node positions in cm.
    
    sigma_t: np.ndarray 
        Array of total cross-section values for each element in cm^-1:

        shape: (number_of_elements, energy_groups)

    sigma_s: np.ndarray
        Array of scattering cross-section values for each element in cm^-1.
        shape: (number_of_elements, N_max + 1, energy_groups (out), energy_groups (in))
    q: np.ndarray
        Array of source term values for each element in cm^-1.
        shape: (number_of_elements, N_max + 1, dof_per_element, energy_groups)
    """

    n_elem = [int(round(length * elements_per_cm)) for length, _, _, _ in regions]
    total_elem = sum(n_elem)

    if energy_groups is None:
        energy_groups = regions[0][1].shape[-1]

        # We could ensure that every regions has the same number of energy groups and l scattering, but we just assume for now.
    
    if regions[0][2].shape[0] < N_max + 1:
        L_slice_scat = slice(0, regions[0][2].shape[0])
    else:
        L_slice_scat = slice(0, N_max + 1)
    
    sigma_t_total = np.zeros((total_elem, energy_groups))
    sigma_s_total = np.zeros((total_elem, N_max + 1, energy_groups, energy_groups))
    q_total       = np.zeros((total_elem, N_max + 1, dof_elem, energy_groups))
    
    nodes = [0.0]    


    
    
    for i, (length, sigma_t, sigma_l_gout_gin, src_flat) in enumerate(regions):
        region_nodes = np.linspace(nodes[-1], nodes[-1] + length, n_elem[i] + 1)[1:]
        nodes.extend(region_nodes)

        elem_slice = slice(sum(n_elem[:i]), sum(n_elem[:i+1]))
        
        sigma_t_total[elem_slice, :]       = sigma_t[np.newaxis, 0:energy_groups]        
        sigma_s_total[elem_slice, L_slice_scat, :, :] = sigma_l_gout_gin[np.newaxis, L_slice_scat, 0:energy_groups, 0:energy_groups]
        
        q_total[elem_slice, 0,:, :] = src_flat[np.newaxis, np.newaxis, 0:energy_groups]  # Only zeroth order is non-zero
        
    return np.array(nodes), sigma_t_total, sigma_s_total, q_total

