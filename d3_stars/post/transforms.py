from collections import OrderedDict

import numpy as np
import h5py
from mpi4py import MPI

import dedalus.public as d3


class SHTransformer():

    def __init__(self, nphi, ntheta, dtype=np.float64, dealias=1, radius=1):
        resolution = (nphi, ntheta)
        self.dtype = dtype
        if self.dtype not in (np.float64, np.complex128):
            raise ValueError("Invalid dtype")

        # Parameters
        c = d3.SphericalCoordinates('phi', 'theta', 'r')
        dealias_tuple = (dealias, dealias)
        Lmax = resolution[1]-1
        sphere_area = 4*np.pi*radius**2
        
        dist = d3.Distributor((c,), mesh=None, comm=MPI.COMM_SELF, dtype=dtype)
        basis = d3.SphereBasis(c.S2coordsys, resolution, radius=radius, dtype=dtype, dealias=dealias_tuple)
        self.phi, self.theta = basis.global_grids(basis.dealias)

        self.scalar_field = dist.Field(bases=basis)
        self.vector_field = dist.VectorField(bases=basis, coordsys=c)
        self.conj_vector_field = dist.VectorField(bases=basis, coordsys=c)
        self.power_scalar_op = d3.Average(self.scalar_field*np.conj(self.scalar_field))
        self.power_vector_op = d3.Average(self.conj_vector_field@self.vector_field)

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
        self.out_field = None
        self.vector = False

    def transform_scalar_field(self, grid_data, normalization=1/2):
        self.vector = False
        self.scalar_field['g'] = grid_data
        power_grid = self.power_scalar_op.evaluate()['g'].ravel()[0]

        out_field = np.zeros((self.ell_values.size, self.m_values.size), dtype=np.complex128)
        for i, ell in enumerate(self.ell_values.ravel()):
            for j, m in enumerate(self.m_values.ravel()):
                sl_key = '{},{}'.format(ell,m)
                if sl_key in self.slices.keys():
                    if self.dtype is np.complex128:
                        out_field[i,j] = self.scalar_field['c'][self.slices[sl_key][0]].ravel()
                        out_field[i,j] *= np.sqrt(2)*normalization
                    else:
                        cosmphi, sinmphi = self.slices[sl_key]
                        out_field[i,j] = self.scalar_field['c'][cosmphi].ravel()[0] \
                                        + 1j*self.scalar_field['c'][sinmphi].ravel()[0]
                        out_field[i,j] *= normalization
                        if m == 0:
                            out_field[i,j] *= np.sqrt(2) #normalize so that sum(coeff*conj(coeffs)) == 4*s2_avg(scalar_field**2)
        #check power
        power_transform = np.sum(out_field * np.conj(out_field)).real
        if (np.allclose(power_transform, 0) and np.allclose(power_grid, 0)) or np.allclose(power_transform/power_grid, 1): 
            self.out_field = out_field
            return out_field
        else:
            self.out_field = None
            raise ValueError("Scalar Transform is not conserving power; ratio: {}, vals: {}, {}".format(power_transform/power_grid, power_transform, power_grid))

    def transform_vector_field(self, grid_data, normalization=1/2):
        self.vector = True
        self.vector_field['g'] = grid_data
        self.conj_vector_field['g'] = np.conj(self.vector_field['g'])
        power_grid = self.power_vector_op.evaluate()['g'].ravel()[0]

        out_field = np.zeros((grid_data.shape[0], self.ell_values.size, self.m_values.size), dtype=np.complex128)
        for i, ell in enumerate(self.ell_values.ravel()):
            for j, m in enumerate(self.m_values.ravel()):
                sl_key = '{},{}'.format(ell,m)
                if sl_key in self.slices.keys():
                    for v  in range(grid_data.shape[0]):
                        if self.dtype is np.complex128:
                            index = (slice(v, v+1, 1), *self.slices[sl_key][0])
                            out_field[v,i,j] = self.vector_field['c'][index].ravel()
                            out_field[v,i,j] *= np.sqrt(2)*normalization
                        else:
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
            self.out_field = out_field
            return out_field
        else:
            self.out_field = None
            raise ValueError("Vector Transform is not conserving power; ratio: {}, vals: {}, {}".format(power_transform/power_grid, power_transform, power_grid))

    def get_ell_m_value(self, ell, m):
        if self.out_field is None:
            raise ValueError("Must transform field before finding ell,m value")

        index = (self.ell_values == ell)*(self.m_values == m)
        if self.vector:
            return self.out_field[:,index]
        else:
            return self.out_field[index]

class DedalusShellSHTransformer():
    """ 
    Transforms all scalar and vector fields in a dedalus output file.
    Assumes all output tasks are defined on S2.

    Relies on plotpal for file reading.
    """
    def __init__(self, nphi, ntheta, root_dir, data_dir, dtype=np.float64, dealias=1, radius=1, **kwargs):
        from plotpal.file_reader import SingleTypeReader as SR
        self.nphi = nphi
        self.ntheta = ntheta
        self.root_dir = root_dir
        self.out_dir = 'SH_transform_{}'.format(data_dir)
        self.reader = SR(root_dir, data_dir, self.out_dir, distribution='even-file', **kwargs)

        # Parameters
        self.transformer = SHTransformer(nphi, ntheta, dtype=dtype, dealias=dealias, radius=radius)

        if not self.reader.idle:
            with h5py.File(self.reader.files[0], 'r') as f:
                self.fields = list(f['tasks'].keys())
        else:
            self.fields = None

    def write_transforms(self):
        if not self.reader.idle:
            while self.reader.writes_remain():
                dsets, ni = self.reader.get_dsets(self.fields)
                file_name = self.reader.current_file_name
                file_num = int(file_name.split('_s')[-1].split('.h5')[0])
                if ni == 0:
                    file_mode = 'w'
                else:
                    file_mode = 'a'
                output_file_name = '{}/{}/{}_s{}.h5'.format(self.root_dir, self.out_dir, self.out_dir, file_num)

                with h5py.File(output_file_name, file_mode) as of:
                    sim_times = self.reader.current_file_handle['scales/sim_time'][()]
                    if ni == 0:
                        of['ells'] = self.transformer.ell_values[None,:,:,None]
                        of['ms']   = self.transformer.m_values[None,:,:,None]
                        of['time'] = sim_times[()]
                        for attr in ['writes', 'set_number', 'handler_name']:
                            of.attrs[attr] = self.reader.current_file_handle.attrs[attr]

                    outputs = OrderedDict()
                    for f in self.fields:
                        vector = False
                        task_data = dsets[f][ni,:]
                        size = task_data.size
                        shape = list(task_data.squeeze().shape)
                        if task_data.size == self.transformer.scalar_field['g'].size:
                            shape[0] = self.transformer.ell_values.shape[0]
                            shape[1] = self.transformer.m_values.shape[1]
                        elif task_data.size == self.transformer.vector_field['g'].size:
                            vector = True
                            shape[1] = self.transformer.ell_values.shape[0]
                            shape[2] = self.transformer.m_values.shape[1]
                        if ni == 0:
                            of.create_dataset(name='tasks/'+f, shape=[len(sim_times),]+shape, dtype=np.complex128)
                            if vector:
                                out_field = self.transformer.transform_vector_field(task_data)
                            else:
                                out_field = self.transformer.transform_scalar_field(task_data)
                            of['tasks/'+f][ni,:] =  out_field



