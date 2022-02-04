import dedalus.public as de
import numpy as np
import matplotlib.pyplot as plt
import logging
import time
logger = logging.getLogger(__name__)


# Domain
Nz = 64
L = 1

k_n = 10
aspect = 4
kx = k_n*2*np.pi/(aspect*L)
nu = 1
def EVP(this_nz):
    print('solving evp with nz = {}'.format(this_nz))
    z_basis = de.Chebyshev('z', this_nz, interval=(0, L))
    domain = de.Domain([z_basis], np.complex128)

    # Problem
    problem = de.EVP(domain, variables=['p', 'u', 'w', 'u_z', 'w_z'],eigenvalue='om')

    problem.parameters['kx'] = kx
    problem.parameters['nu'] = nu
    problem.substitutions['dt(A)'] = '-1j*om*A'
    problem.substitutions['dx(A)'] = '1j*kx*A'

    problem.add_equation("dx(u) + w_z = 0")
    problem.add_equation("dt(u) + dx(p) - nu*(dx(dx(u)) + dz(u_z)) = 0")
    problem.add_equation("dt(w) + dz(p) - nu*(dx(dx(w)) + dz(w_z)) = 0")
    problem.add_equation("u_z - dz(u) = 0")
    problem.add_equation("w_z - dz(w) = 0")
    problem.add_bc("left(u_z) = 0")
    problem.add_bc("right(u_z) = 0")
    problem.add_bc("left(w) = 0")
    problem.add_bc("right(w) = 0")

    # Solver
    solver = problem.build_solver()
    solver.solve_dense(solver.pencils[0])

    # Filter infinite/nan eigenmodes
    finite = np.isfinite(solver.eigenvalues)
    solver.eigenvalues = solver.eigenvalues[finite]
    solver.eigenvectors = solver.eigenvectors[:, finite]

    # Sort eigenmodes by eigenvalue
    order = np.argsort(-solver.eigenvalues.imag)
    solver.eigenvalues = solver.eigenvalues[order]
    solver.eigenvectors = solver.eigenvectors[:, order]
    
    return solver

solver1 = EVP(Nz)
solver2 = EVP(int(1.5*Nz))

cot = lambda x: np.cos(x)/np.sin(x)

good1 = []
good2 = []
cutoff = 1e-8
cutoff2 = np.sqrt(cutoff)
for i, ev1 in enumerate(solver1.eigenvalues):
    for j, ev2 in enumerate(solver2.eigenvalues):
        if np.abs((ev1 - ev2))/np.abs(ev1) < cutoff:
            print(ev1.imag, ev2.imag, np.abs((ev1 - ev2))/np.abs(ev1))
            solver1.set_state(i)
            solver2.set_state(j)
            w1 = solver1.state['w']
            u1 = solver1.state['u']
            w2 = solver2.state['w']
            w1.set_scales(3/2)
            u1.set_scales(3/2)
            ix = np.argmax(np.abs(w1['g']))
            u1['g'] /= w1['g'][ix]
            w1['g'] /= w1['g'][ix]
            ix = np.argmax(np.abs(w2['g']))
            w2['g'] /= w2['g'][ix]

            #make sure they're not out of phase
            if np.allclose(w1['g'][ix]/w2['g'][ix].real, -1):
                w2['g'] *= -1

            vector_diff = np.max(np.abs(w1['g'] - w2['g']))
            print(vector_diff)
            if vector_diff >= cutoff2:
                print(w1['g'][ix]/w2['g'][ix])
                z = solver1.domain.grid(0, scales=3/2)
                plt.plot(z, w1['g'].real)
                plt.plot(z, w2['g'].real)
                plt.show()
            else:
                good1.append(i)
                good2.append(j)
                break
#                ell = -ev1.imag
#                a = -kx**2 * (-ell + nu*kx**2)
#                b = 2*nu*kx**2 - ell
#                d = -nu
#                k2 = np.array(b/d, dtype=np.complex128)
#                print(k2, np.cos(1j))
#                c1 = -a*np.sqrt(k2)*L*cot(k2*L/2)/(2*b)
#                c2 = -a*np.sqrt(k2)*L/(2*b)
#                c3 = a*L*cot(k2*L/2)/(2*b*np.sqrt(k2))
#                c4 = a*L/(2*b)
#                func = c3 + z*c4 + (1/b)*(-a*z**2/2 + d*c1*np.cos(np.sqrt(k2)*z) + d*c2*np.sin(np.sqrt(k2)*z))
#                print(func)
#                plt.plot(z, func)
#                plt.plot(solver1.domain.grid(0, scales=3/2), w1['g'].real)
#                plt.plot(solver1.domain.grid(0, scales=3/2), u1['g'].imag)
#                plt.plot(solver1.domain.grid(0, scales=3/2), w1['g'].imag)
#                plt.plot(solver1.domain.grid(0, scales=3/2), u1['g'].real)
#                plt.plot(solver1.domain.grid(0, scales=3/2), -kx*u1['g'].imag/w1['g'].real)
#                plt.plot(solver1.domain.grid(0, scales=3/2), w1['g'].imag)
solver1.eigenvalues = solver1.eigenvalues[good1]
solver1.eigenvectors = solver1.eigenvectors[:,good1]
solver2.eigenvalues = solver2.eigenvalues[good2]
solver2.eigenvectors = solver2.eigenvectors[:,good2]

# Plot error vs exact eigenvalues
mode_number = 1 + np.arange(len(solver1.eigenvalues))
print(mode_number)
kzs = (np.pi*mode_number)/L
exact_eigenvalues = -1j * nu * (kx**2 + kzs**2)
eval_relative_error = (solver1.eigenvalues.imag - exact_eigenvalues.imag) / exact_eigenvalues.imag

print('analytic', exact_eigenvalues)
print('solved', solver1.eigenvalues.imag)
print('ratio', solver1.eigenvalues.imag/exact_eigenvalues.imag)
#print('lambda/nu > kx**2', -solver1.eigenvalues.imag/nu > kx**2, -solver1.eigenvalues.imag, kx**2)

plt.figure()
plt.semilogy(mode_number, np.abs(eval_relative_error), lw=0, marker='o')
plt.xlabel("Mode number")
plt.ylabel(r"$|\lambda - \lambda_{exact}|/\lambda_{exact}$")
plt.title(r"Eigenvalue relative error ($N_z=%i$)" % Nz)
plt.savefig('eval_error.png')
#plt.show()
