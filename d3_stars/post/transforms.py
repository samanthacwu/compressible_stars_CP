import numpy as np
from mpi4py import MPI
import dedalus.public as d3


class SHTransformer():

    def __init__(self, nphi, ntheta, dtype=np.float64, dealias=1, radius=1):
        resolution = (nphi, ntheta)

        # Parameters
        c = d3.SphericalCoordinates('phi', 'theta', 'r')
        dealias_tuple = (dealias, dealias)
        Lmax = resolution[1]-1
        sphere_area = 4*np.pi*radius**2
        dist = d3.Distributor((c,), mesh=None, comm=MPI.COMM_SELF, dtype=dtype)
        basis = d3.SphereBasis(c.S2coordsys, resolution, radius=radius, dtype=dtype, dealias=dealias_tuple)
        phig, thetag = basis.global_grids(basis.dealias)

        self.scalar_field = dist.Field(bases=basis)
        self.vector_field = dist.VectorField(bases=basis, coordsys=c)
        self.power_scalar_op = d3.Average(self.scalar_field**2)
        self.power_vector_op = d3.Average(self.vector_field@self.vector_field)

        self.ell_values = []
        self.m_values = []

        self.slices = dict()
        for i in range(self.scalar_field['c'].shape[0]):
            for j in range(self.scalar_field['c'].shape[1]):
                groups = basis.elements_to_groups((False, False), (np.array((i,)),np.array((j,))))
                m = groups[0][0]
                ell = groups[1][0]
                key = '{},{}'.format(ell, m)
                this_slice = (slice(i, i+1, 1), slice(j, j+1, 1))
                if key not in self.slices.keys():
                    self.slices[key] = [this_slice]
                else:
                    self.slices[key].append(this_slice)

                if ell not in self.ell_values:
                    self.ell_values.append(ell)
                if m not in self.m_values:
                    self.m_values.append(m)
        self.ell_values = np.sort(self.ell_values)[:,None]
        self.m_values = np.sort(self.m_values)[None,:]

    def transform_scalar_field(self, grid_data):
        self.scalar_field['g'] = grid_data
        power_grid = self.power_scalar_op.evaluate()['g'].ravel()[0]

        out_field = np.zeros((self.ell_values.size, self.m_values.size), dtype=np.complex128)
        for i, ell in enumerate(self.ell_values.ravel()):
            for j, m in enumerate(self.m_values.ravel()):
                sl_key = '{},{}'.format(ell,m)
                if sl_key in self.slices.keys():
                    cosmphi, sinmphi = self.slices[sl_key]
                    out_field[i,j] = self.scalar_field['c'][cosmphi].ravel()[0] \
                                    + 1j*self.scalar_field['c'][sinmphi].ravel()[0]
                    if m == 0:
                        out_field[i,j] *= np.sqrt(2) #normalize so that sum(coeff*conj(coeffs)) == 4*s2_avg(scalar_field**2)
        #check power
        power_transform = np.sum(out_field * np.conj(out_field)).real
        if (np.allclose(power_transform, 0) and np.allclose(power_grid, 0)) or np.allclose(power_transform/power_grid, 1): 
            return out_field
        else:
            raise ValueError("Scalar Transform is not conserving power; ratio: {}, vals: {}, {}".format(power_transform/power_grid, power_transform, power_grid))

    def transform_vector_field(self, grid_data, normalization=1/2):
        self.vector_field['g'] = grid_data
        power_grid = self.power_vector_op.evaluate()['g'].ravel()[0]

        out_field = np.zeros((grid_data.shape[0], self.ell_values.size, self.m_values.size), dtype=np.complex128)
        for i, ell in enumerate(self.ell_values.ravel()):
            for j, m in enumerate(self.m_values.ravel()):
                sl_key = '{},{}'.format(ell,m)
                if sl_key in self.slices.keys():
                    for v  in range(grid_data.shape[0]):
                        cosmphi, sinmphi = self.slices[sl_key]
                        cosmphi = (slice(v, v+1, 1), *cosmphi)
                        sinmphi = (slice(v, v+1, 1), *sinmphi)
                        out_field[v,i,j] = self.vector_field['c'][cosmphi].ravel()[0] \
                                        + 1j*self.vector_field['c'][sinmphi].ravel()[0]
                        out_field[v,i,j] *= normalization
                        if m == 0:
                            out_field[v,i,j] *= np.sqrt(2) #normalize so that sum(coeff*conj(coeffs)) == 4*s2_avg(vector_field**2)
        #check power
        power_transform = np.sum(out_field * np.conj(out_field)).real
        if (np.allclose(power_transform, 0) and np.allclose(power_grid, 0)) or np.allclose(power_transform/power_grid, 1): 
            return out_field
        else:
            raise ValueError("Vector Transform is not conserving power; ratio: {}, vals: {}, {}".format(power_transform/power_grid, power_transform, power_grid))

