import jax 
import jax.numpy as jnp
import numpy as np
import basix
from scipy.sparse import lil_matrix
from typing import List, Iterable, Literal, Union, Tuple
from numpy.typing import NDArray
from numpy.polynomial.legendre import legval

# Type aliases for arrays with specific dimensionality
Array1D = NDArray[np.floating]  # 1D array
Array2D = NDArray[np.floating]  # 2D array
Array3D = NDArray[np.floating]  # 3D array
Array4D = NDArray[np.floating]  # 4D array

# Type alias for a single region tuple
RegionTuple = Tuple[
    float,    # length in cm
    Array1D,  # sigma_t: 1D array, shape (energy_groups,)
    Array3D,  # sigma_s: 3D array, shape (L_scat, energy_groups, energy_groups)
    Array1D   # source: 1D array, shape (energy_groups,)
]

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


def build_multigroup_elements_and_materials(
    regions: List[RegionTuple], 
    N_max: int, 
    element : basix.finite_element.FiniteElement, 
    elements_per_cm: int, 
    energy_groups: int = None,
    elements_per_region : List[int] = None
) -> Tuple[Array1D, Array2D, Array4D, Array4D]:
    """
    Build the nodes, centers, and material properties for a 1D mesh based on the given regions.
    Parameters: 
    -----------
    regions: list of tuples (length, sigma_t, sigma_s, source)
        Each tuple defines a region with its length (cm), total cross-section (sigma_t, array of shape (energy_groups) ), 
        scattering cross-section (sigma_k_gout_gin, array (L_scat, energy_groups, energy_groups) for some maximum L_scat order), and external, isotropic source term (q, array(energy_groups)) (in cm^-1).
    elements_per_cm: int
        Number of finite elements per centimeter.
    N_max: int
        Maximum order of the PN method (or double PN).
    element: basix.finite_element.FiniteElement
        The finite element to use for the assembly.
    energy_groups: int, optional
        Number of energy groups. If not provided, it is inferred from the regions parameter.
    elements_per_region: list of int, optional
        Number of elements per region. Either elements_per_cm or elements_per_region should be provided, not both.
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

    # Checking that either elements_per_cm is given or elements_per_region is given
    if elements_per_cm is not None and elements_per_region is not None:
        raise ValueError("Either elements_per_cm or elements_per_region should be provided, not both.")

    # Converting all lists to numpy arrays for easier manipulation
    def ensure_numpy_arrays(region: RegionTuple) -> RegionTuple:
        """Ensure all arrays in a region tuple are numpy arrays"""
        return (
            region[0],  # length (unchanged)
            np.array(region[1]) if not isinstance(region[1], np.ndarray) else region[1],
            np.array(region[2]) if not isinstance(region[2], np.ndarray) else region[2],
            np.array(region[3]) if not isinstance(region[3], np.ndarray) else region[3]
        )
    
    regions = [ensure_numpy_arrays(region) for region in regions]

    # Checking consistency of regions 
    def Get_Eg_Region(region : RegionTuple):
        ng_sigma_t    = region[1].shape[-1]
        ng_sigma_s_in = region[2].shape[-1]
        ng_sigma_s_out= region[2].shape[-2]
        nL_sigma_s    = region[2].shape[0]
        ng_source     = region[3].shape[-1]
        if(ng_sigma_t != ng_sigma_s_in or ng_sigma_t != ng_sigma_s_out or ng_sigma_t != ng_source):
            raise ValueError("Inconsistent number of energy groups in region number {i}:\n"
                             f"Expected {ng_sigma_t} energy groups, but got:\n"
                             f"total cross-section: {ng_sigma_t}, "
                             f"scattering matrix: {ng_sigma_s_in}x{ng_sigma_s_out}, "
                             f"source term: {ng_source}")   
        return ng_sigma_t, nL_sigma_s

    ng_sigma_t, nL_sigma_s = Get_Eg_Region(regions[0])
    for i, region in enumerate(regions[1:], 1):
        region_ng, region_nL = Get_Eg_Region(region)
        if (region_ng, region_nL) != (ng_sigma_t, nL_sigma_s):
            raise ValueError(f"Inconsistent number of energy groups or L scattering orders in regions. "
                           f"Region 0 has {ng_sigma_t} energy groups and {nL_sigma_s} L orders, "
                           f"but region {i} has {region_ng} energy groups and {region_nL} L orders.")
        





    dof_elem = element.dim

    if elements_per_region is not None:
        if len(elements_per_region) != len(regions):
            raise ValueError(f"elements_per_region must have the same length as regions but got {len(elements_per_region)} vs {len(regions)}.")
        n_elem = elements_per_region
        print(n_elem)
    elif elements_per_cm is not None:
        n_elem = [int(round(length * elements_per_cm)) for length, _, _, _ in regions]
        print(n_elem)
    else:
        raise ValueError("None of elements_per_cm or elements_per_region are provided.")
    total_elem = sum(n_elem)

    # Truncating energy groups
    if energy_groups is None:
        energy_groups = regions[0][1].shape[-1]

        # We could ensure that every regions has the same number of energy groups and l scattering, but we just assume for now.
    

    # We check whether:
    #  - the scattering matrices should be truncated and thus sliced to N_max + 1
    #  - the scattering matrices should be extended to N_max + 1 (and thus the zero output array should only be set until N_max + 1)
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

