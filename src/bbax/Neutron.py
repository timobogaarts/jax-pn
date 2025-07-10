from .FEM1D import create_dof_matrix_vertex_interior, build_multigroup_elements_and_materials, interpolate_solution
import basix
import abc
from scipy import sparse as sps
from scipy.sparse.linalg import spsolve
from scipy.sparse import lil_matrix
import numpy as np
from functools import lru_cache


class Neutron_Problem:

    def __init__(self, nodes, sigma_t, sigma_s, q, N_max, L_scat, element : basix.finite_element.FiniteElement):
        """
        Initialize the DPN problem with nodes, total cross-section, scattering matrix, and source term.
        
        Parameters:
        -----------
        element: basix.Element
            The finite element to use for the assembly.
        nodes: np.ndarray
            Array of node positions in cm.
        sigma_t: np.ndarray(number_of_elements, number_of_energy_groups)
            Arrays for total cross section in cm^-1 for each element and energy group.
        sigma_s: np.ndarray(number_of_elements, N_max + 1, number_of_energy_groups (out), number_of_energy_groups(in) )
            Ingroup scattering matrices for each element and energy group. Assumed it is extended to L_scat, and if L_scat not given, to N_max. Only up until L_scat is used (i.e. L_scat = 1, uses sigma_s[:, :, 0:2]).
        q: np.ndarray(number_of_elements, N_max + 1, dof_per_element, number_of_energy_groups)
            External source terms for each element and energy group. Assumed to be extended to N_max + 1
        """
        self.nodes = nodes
        self.sigma_t = sigma_t
        self.sigma_s = sigma_s
        self.q = q
        self.N_max = N_max
        self.L_scat = L_scat
        self.element = element
        self.dof_matrix, self.n_global_dofs = create_dof_matrix_vertex_interior(self.element, self.nodes)    
        self.dofs_per_eg = self.set_dofs_per_eg()

    @classmethod
    def from_regions(cls, regions, elements_per_cm, N_max, element, L_scat):
        """
        Create a DPN_Problem instance from a list of regions.
        
        Parameters:
        -----------
        regions: list of tuples (length, sigma_t, sigma_s, source)
            Each tuple defines a region with its length (cm), total cross-section (sigma_t), 
            scattering cross-section ([sigma_k_gout_gin] for some maximum k order), and external source term (q) (in cm^-1).
        elements_per_cm: int
            Number of finite elements per centimeter.
        N_max: int
            Maximum order of the DPN method.
        element: basix.Element
            The finite element to use for the assembly.
        
        Returns:
        --------
        DPN_Problem instance
        """        
        nodes, sigma_t, sigma_s, q = build_multigroup_elements_and_materials(regions, elements_per_cm, N_max, element.dim)
        return cls(nodes, sigma_t, sigma_s, q, N_max, L_scat, element)
    
    @abc.abstractmethod
    def Assemble_Single_Energy_Group(self, energy_group, bc):
        pass

    @abc.abstractmethod
    def Assemble_Downscatter_Matrix(self, energy_group_out, energy_group_in):
        pass

    @abc.abstractmethod
    def set_dofs_per_eg(self):
        pass
        
    
    @lru_cache(maxsize=128)
    def assemble_multigroup_system(self, bc, n_energy_groups=None):
        """
        Assemble the multigroup DPN finite element matrix and right-hand side vector for the 1D transport equation.
        
        Parameters:
        -----------
        bc: Literal["vacuum"]
            Boundary condition to apply.
        n_energy_groups: int, optional
            Number of energy groups. If None, uses the number of energy groups in sigma_s.
        
        Returns:
        --------
        A_total: scipy.sparse.lil_matrix
            The assembled global matrix.
        b_total: scipy.sparse.lil_matrix
            The right-hand side vector.
        """
        if n_energy_groups is None:
            n_energy_groups = self.sigma_s.shape[-1]

        total_dofs = self.dofs_per_eg * n_energy_groups

        A_total = lil_matrix((total_dofs, total_dofs), dtype=np.float64)
        b_total = lil_matrix((total_dofs, 1), dtype=np.float64)

        for i in range(n_energy_groups):
            A, b = self.Assemble_Single_Energy_Group(i, bc)
            start_row =       i * self.dofs_per_eg
            end_row   = (i + 1) * self.dofs_per_eg
            
            A_total[start_row:end_row, start_row:end_row] = A
            b_total[start_row:end_row, 0] = b
            for g_in in range(i):
                A_downscatter = self.Assemble_Downscatter_Matrix(i, g_in)
                A_total[start_row:end_row, g_in * self.dofs_per_eg:(g_in + 1) * self.dofs_per_eg] -= A_downscatter

        return A_total, b_total
    
    def Solve_Multigroup_System(self, bc, n_energy_groups=None):
        """
        Solve the multigroup DPN system.
        
        Parameters:
        -----------
        bc: Literal["vacuum"]
            Boundary condition to apply.
        n_energy_groups: int, optional
            Number of energy groups. If None, uses the number of energy groups in sigma_s.
        
        Returns:
        --------
        x: np.ndarray
            The solution vector.
        """
        A, b = self.assemble_multigroup_system(bc, n_energy_groups)
        self.solution =  spsolve(A.tocsr(), b.tocsr())
        return self.solution
    
    @abc.abstractmethod
    def _get_single_spatial_solution(self, k, mu_sign, energy_group):
        pass

    def interpolate_solution(self, x_points, **kwargs):
        """
        Interpolate the DPN solution at arbitrary points x_points.
        
        Parameters:
        -----------
        x_points: np.ndarray
            Points at which to interpolate the solution.
        k: int
            The moment to interpolate.
        mu_sign: Literal[-1, 1]
            The sign of the cosine of the angle.
        energy_group: int
            The energy group to interpolate.
        
        Returns:
        --------
        values: np.ndarray
            Interpolated values at x_points.
        """
        
        return interpolate_solution(x_points, self.nodes, self.dof_matrix, self._get_single_spatial_solution(**kwargs), self.element)
