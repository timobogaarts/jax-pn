import jax 
import jax.numpy as jnp
import numpy as np
import basix
from scipy.sparse import lil_matrix
from typing import List, Iterable, Literal, Union
from numpy.polynomial.legendre import legval
from scipy.special import legendre
from numpy.polynomial.legendre import leggauss
from .FEM1D import create_dof_matrix_vertex_interior
from .Neutron import Neutron_Problem, interpolate_solution
from functools import cached_property


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



def assemble_PN_matrix(element : basix.finite_element.FiniteElement, nodes : np.ndarray, sigma_t : Iterable[float], sigma_s : List[np.ndarray], q : List[np.ndarray], N_max : int, bc = Literal["vacuum", "reflective", "none"], L_scat : int = None):
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
    q: np.ndarray(number_of_elements, N_max + 1, dof_per_element)]
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

            A_local = mass_matrix * h

            B_local = local_streaming
            
            s_local = np.einsum("ij,kj -> ik", mass_matrix * h,  q[i])
            
            for local_i in range(no_dofs):

            
                b[total_dof(i, local_i, k), 0] += s_local[local_i, k]

                for local_j in range(no_dofs):
                    
                    # Collision term
                    A[total_dof(i, local_i, k), total_dof(i, local_j, k)] += A_local[local_i, local_j] * sigma_t[i]

                    # Scatter term
                    if k < L_scat + 1:
                        A[total_dof(i, local_i, k), total_dof(i, local_j, k)] -= A_local[local_i, local_j] * sigma_s[i][k]

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
    elif bc == "vacuum":    
        _apply_vacuum_bc(    A, b, n_global_dofs, L_tot, left_dof = 0 , right_dof = nodes.shape[0] - 1)                           
    elif bc == "none":
        pass
    else:
        raise ValueError(f"Unknown boundary condition: {bc}. Supported: 'reflective', 'vacuum'.")

    acoo = A.tocoo()
    bcoo = b.tocoo()
    
    return acoo.data, acoo.row, acoo.col, acoo.shape, bcoo.data, bcoo.row, bcoo.col, bcoo.shape


def Assemble_Downscatter_PN_Matrix(element : basix.finite_element.FiniteElement, nodes : np.ndarray, sigma_s : np.ndarray, N_max : int, L_scat : int = None):
    '''
    Assemble the downscatter PN finite element matrix for the 1D transport equation.

    Parameters:
    -----------
    element: basix.Element
        The finite element to use for the assembly.
    nodes: np.ndarray
        Array of node positions in cm.
    sigma_s: np.ndarray(number_of_elements, N_max + 1)
        Group-to-group scatter values
    N_max: int
        Maximum angular moment.
    L_scat: int, optional
        Maximum scattering order. If None, defaults to N_max.
    
    Returns:
    --------
    A: scipy.sparse.lil_matrix
        The assembled finite element matrix.
    '''
    
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
    
    hihj = np.einsum('qi,qj->qij', phi, phi)  # (n_qp, n_basis, n_basis)
    mass_matrix = np.tensordot(weights, hihj, axes=([0], [0])) # \int H_i H_j d\xi (no Jacobian)

    
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

    ADSS = lil_matrix((n_global_dofs  * L_tot, n_global_dofs * L_tot), dtype=np.float64)


    # Function that calculates the total global degree of freedom for a given element and local index and k moment    
    def total_dof(element_i, local_i, k):
        return dof_matrix[element_i, local_i] + k * n_global_dofs
    
    for k in range(L_tot):        
        global_dof_start = k * n_global_dofs
        for i in range(n_elem):            
            
            no_dofs = len(dof_matrix[i])
            h = nodes[i+1] - nodes[i]           

            A_local = mass_matrix * h                                
            
            for local_i in range(no_dofs):                        
                for local_j in range(no_dofs):                                        

                    # Scatter term
                    if k < L_scat + 1:
                        #print(k, L_scat)
                        ADSS[total_dof(i, local_i, k), total_dof(i, local_j, k)] -= A_local[local_i, local_j] * sigma_s[i][k]

    coo_mat = ADSS.tocoo()
    return coo_mat.data, coo_mat.row, coo_mat.col, coo_mat.shape




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

def _apply_vacuum_bc(A, b, n_global_dofs, L_tot, left_dof, right_dof):
        
        left_coeff_matrix = legendre_coeff_matrix(L_tot, 0, 1)
        right_coeff_matrix = legendre_coeff_matrix(L_tot, -1, 0)
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


def legendre_coeff_matrix(L_max, a, b):
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


class PN_Problem(Neutron_Problem):
    def Assemble_Single_Energy_Group(self, energy_group : int, bc : Literal["vacuum", "reflective", "none"]):
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
        return assemble_PN_matrix(self.element, self.nodes, self.sigma_t[:, energy_group], self.sigma_s[:, :, energy_group, energy_group], self.q[:, :, :, energy_group], self.N_max, bc, self.L_scat)        
    
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
        return Assemble_Downscatter_PN_Matrix(self.element, self.nodes, self.sigma_s[:, :, energy_group_out, energy_group_in], self.N_max, self.L_scat)
    
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
