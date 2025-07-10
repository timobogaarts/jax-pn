import jax 
import jax.numpy as jnp
import numpy as np
import basix
from scipy.sparse import lil_matrix
from typing import List, Iterable, Literal, Union
from numpy.polynomial.legendre import legval
from scipy.special import legendre
from numpy.polynomial.legendre import leggauss
from .FEM1D import build_elements_and_materials, create_dof_matrix_vertex_interior



def interpolate_PN_solution(x_points, nodes, elem_dofs, solution, lagrange, N_max, k = 0):
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
    
    n_global_dofs = solution.shape[0] // (N_max + 1)

    solution_k = solution[k * n_global_dofs : (k + 1) * n_global_dofs]
    
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
        u_local = solution_k[elem_dofs[e]]
        # Interpolate
        values[i] = np.dot(phi, u_local)
    return values






def Create_PN_Matrix_From_Regions(element : basix.finite_element.FiniteElement, regions : List, elements_per_cm : int, N_max : int, bc : Literal["marshak", "reflective"], energy_group : int, L_scat : int = None):
    nodes, sigma_t, sigma_s, q = build_elements_and_materials(regions, elements_per_cm, N_max, energy_group, element.dim)
    return *assemble_PN_matrix(element, nodes, sigma_t, sigma_s, q, N_max, bc, L_scat), nodes

def assemble_PN_matrix(element : basix.finite_element.FiniteElement, nodes : np.ndarray, sigma_t : Iterable[float], sigma_s : List[np.ndarray], q : List[np.ndarray], N_max : int, bc : Literal["marshak", "reflective"], L_scat : int = None):
    '''
    Assemble the 1-Group PN finite element matrix and right-hand side vector for the 1D transport equation.

    Parameters:
    -----------
    element: basix.Element
        The finite element to use for the assembly.
    nodes: np.ndarray
        Array of node positions in cm.
    sigma_t: np.ndarray(number_of_elements)
        Arrays for total cross section in cm^-1 for each element.
    sigma_s: np.ndarray(number_of_elements, N_max + 1)]
        Ingroup scattering matrices for each element. Assumed it is extended to L_scat, and if L_scat not given, to N_max. Only up until L_scat is used (i.e. L_scat = 1, uses sigma_s[:, 0:2]).
    q: np.ndarray(number_of_elements, N_max, dof_per_element)]
        External source terms for each element. Assumed to be extended to N_max + 1
    '''
    if N_max % 2 == 0:
        raise ValueError("N_max must be odd for this PN implementation.")
    
    if L_scat is None:
        L_scat = N_max

    
    
    n_elem = len(nodes) - 1

    L_tot = N_max + 1
    
    # =============================================
    # Setting up finite element space
    # =============================================
    degree = element.degree    
    dof_elem = element.dim
    quad_deg = 2 * degree    
    points, weights = basix.make_quadrature(element.cell_type, quad_deg)

    phidphi = element.tabulate(1, points)
    phi = phidphi[0, :, :, 0]  # (n_quadrature_points, n_basis) = value at [quad_point, basis_no]
    dphi = phidphi[1, :, :, 0] # (n_quadrature_points, n_basis) = d value / dx at [quad_point, basis_no]
    
    hihj = np.einsum('qi,qj->qij', phi, phi)  # (n_qp, n_basis, n_basis)
    mass_matrix = np.tensordot(weights, hihj, axes=([0], [0])) # \int H_i H_j d\xi (no Jacobian)

    dhihj = np.einsum("qi, qj->qij", phi,  dphi)
    local_streaming = np.tensordot(weights, dhihj, axes=([0], [0])) # \int  H_i \partial_\xi H_j d\xi (no Jacobian required)
    

    
    # =============================================
    # Global Matrix Setup
    # =============================================    
    dof_matrix, n_global_dofs = create_dof_matrix_vertex_interior(element, nodes)    
        # dof_matrix: (number_of_elements, dof_per_element)
        #    maps the local dof to a global dof
        # n_global_dofs: total number of global degrees of freedom


    # =============================================
    # Global Matrix assembly
    # =============================================    

    A = lil_matrix((n_global_dofs  * L_tot, n_global_dofs * L_tot), dtype=np.float64)
    b = lil_matrix((n_global_dofs * L_tot, 1), dtype=np.float64)

    # Function that calculates the total global degree of freedom for a given element and local index and k moment    
    def total_dof(element_i, local_i, k):
        return dof_matrix[element_i, local_i] + k * n_global_dofs



    for k in range(L_tot):
        global_dof_start = k * n_global_dofs
        for i in range(n_elem):            
            
            no_dofs = len(dof_matrix[i])
            h = nodes[i+1] - nodes[i]           
            if k < L_scat + 1:
                A_local = (sigma_t[i] - sigma_s[i][k]) * mass_matrix * h             
            else:
                A_local = sigma_t[i] * mass_matrix * h
            B_local = local_streaming
            
            s_local = np.einsum("ij,kj -> ik", mass_matrix * h,  q[i])
            
            for local_i in range(no_dofs):

            
                b[total_dof(i, local_i, k), 0] += s_local[local_i, k]

                for local_j in range(no_dofs):
                    
                    # Collision term
                    A[total_dof(i, local_i, k), total_dof(i, local_j, k)] += A_local[local_i, local_j]

                    # Streaming term 
                    # Note that the equation number of the streaming term is k, so the first index should 
                    # correspond to the current k, and the second index should correspond to the different k.
                    if k != L_tot - 1:
                        A[total_dof(i, local_i, k), total_dof(i, local_j, k + 1)] +=\
                            B_local[local_i, local_j] * ( k + 1 ) / ( 2 * k + 1)                         
                    if k != 0:
                        A[total_dof(i, local_i, k), total_dof(i, local_j, k - 1)] +=\
                            B_local[local_i, local_j] * ( k ) / ( 2 * k + 1)

    # =============================================
    # Apply boundary conditions
    # =============================================
    # Strongly imposed.

    if bc == "reflective":
        _apply_reflective_bc(A, b, n_global_dofs, L_tot, left_dof = 0,  right_dof = nodes.shape[0] - 1)
    elif bc == "marshak":    
        _apply_marshak_bc(    A, b, n_global_dofs, L_tot, left_dof = 0 , right_dof = nodes.shape[0] - 1)                           
    else:
        raise ValueError(f"Unknown boundary condition: {bc}. Supported: 'reflective', 'marshak'.")
    
    return A, b


def _apply_reflective_bc(A, b, n_global_dofs, L_tot, left_dof, right_dof):        
        for k in range(1, L_tot, 2):  # odd moments only
            # Left boundary
            row = left_dof + k * n_global_dofs
            A[row, :] = 0
            A[row, row] = 1
            b[row, 0] = 0
            # Right boundary
            row = right_dof + k * n_global_dofs
            A[row, :] = 0
            A[row, row] = 1
            b[row, 0] = 0

def _apply_marshak_bc(A, b, n_global_dofs, L_tot, left_dof, right_dof):
        
        left_coeff_matrix = _legendre_coeff_matrix(L_tot, 0, 1)
        right_coeff_matrix = _legendre_coeff_matrix(L_tot, -1, 0)
        for enforce_i in range(1, L_tot, 2):  # boundary condition enforced on odd moments
            enforce_row_left  = left_dof +  enforce_i * n_global_dofs
            enforce_row_right = right_dof + enforce_i * n_global_dofs
            
            A[enforce_row_left, :] = 0
            A[enforce_row_right, :] = 0
            b[enforce_row_left, 0] = 0
            b[enforce_row_right, 0] = 0

            for l in range(L_tot):
                left_edge_row  = left_dof  + l * n_global_dofs
                right_edge_row = right_dof + l * n_global_dofs

                A[enforce_row_left, left_edge_row]  = left_coeff_matrix[enforce_i, l] * (2 * l + 1)
                A[enforce_row_right, right_edge_row] = right_coeff_matrix[enforce_i, l] * (2 * l + 1)


def _legendre_coeff_matrix(L_max, a, b):
    M = max(2*L_max, 50)  # Number of quadrature points for accuracy
    
    # Gauss-Legendre nodes and weights on [-1,1]
    nodes, weights = leggauss(M)
    
    # Transform to [a,b]
    x = 0.5 * (nodes * (b - a) + (b + a))
    w = 0.5 * (b - a) * weights
    
    # Evaluate all P_i(x) for i=0..L_max-1, shape (L_max, M)
    P = np.array([legendre(i)(x) for i in range(L_max)])
    
    # Now compute coeff matrix: integral P_i * P_j = sum_k w_k * P_i(x_k)*P_j(x_k)
    # This is P @ diag(w) @ P.T but since w is 1D vector:
    # Use broadcasting or matrix multiplication with weighting:
    W = np.diag(w)
    coeff = P @ W @ P.T
    
    return coeff
