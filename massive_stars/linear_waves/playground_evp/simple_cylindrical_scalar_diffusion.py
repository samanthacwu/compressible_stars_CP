import dedalus.public as de
import numpy as np
import matplotlib.pyplot as plt
import logging
import time
logger = logging.getLogger(__name__)

from scipy.special import jv, jn_zeros

# Domain
Nr = 64
R = 1

phi_n = 2
kappa = 1e-2
def EVP(this_nr):
    print('solving evp with nr = {}'.format(this_nr))
    r_basis = de.Chebyshev('r', this_nr, interval=(0, R))
    domain = de.Domain([r_basis], np.float64)

    # Problem
    problem = de.EVP(domain, variables=['T', 'T_r'],eigenvalue='lam')

    problem.parameters['kappa'] = kappa
    problem.parameters['phi_n'] = phi_n
    problem.substitutions['dt(A)'] = '-lam*A'
    problem.substitutions['dp2(A)'] = '-phi_n**2*A'

    problem.add_equation("r**2*(dt(T) -   kappa*(dp2(T)/r**2 +  (1/r)*dr(r*T_r))) = 0", tau=False)
    problem.add_equation("T_r - dr(T) = 0")
#    problem.add_bc("left(T) = 0")
    problem.add_bc("right(T) = 0")

    # Solver
    solver = problem.build_solver()
    solver.solve_dense(solver.pencils[0])

    # Filter infinite/nan eigenmodes
    finite = np.isfinite(solver.eigenvalues)
    solver.eigenvalues = solver.eigenvalues[finite]
    solver.eigenvectors = solver.eigenvectors[:, finite]

    # Sort eigenmodes by eigenvalue
    order = np.argsort(solver.eigenvalues)
    solver.eigenvalues = solver.eigenvalues[order]
    solver.eigenvectors = solver.eigenvectors[:, order]

    return solver

solver1 = EVP(Nr)
solver2 = EVP(int(1.5*Nr))

cot = lambda x: np.cos(x)/np.sin(x)

good1 = []
good2 = []
cutoff = 1e-8
cutoff2 = np.sqrt(cutoff)
for i, ev1 in enumerate(solver1.eigenvalues):
    for j, ev2 in enumerate(solver2.eigenvalues):
        if np.abs((ev1 - ev2))/np.abs(ev1) < cutoff:
            print(ev1, ev2, np.abs((ev1 - ev2))/np.abs(ev1))
            solver1.set_state(i)
            solver2.set_state(j)
            T1 = solver1.state['T']
            T2 = solver2.state['T']
            T1.set_scales(3/2)
            ix = np.argmax(np.abs(T1['g']))
            T1['g'] /= T1['g'][ix]
            ix = np.argmax(np.abs(T2['g']))
            T2['g'] /= T2['g'][ix]

#            #make sure they're not out of phase
#            if np.allclose(T1['g'][ix]/T2['g'][ix], -1):
#                T2['g'] *= -1

            vector_diff = np.max(np.abs(T1['g'] - T2['g']))
            print(vector_diff)
            if vector_diff < cutoff2:
                print(T1['g'][ix]/T2['g'][ix])
                r = solver1.domain.grid(0, scales=3/2)
#                analytic = jv(phi_n, np.sqrt(ev1.real/kappa)*r)
#                analytic /= analytic[np.argmax(np.abs(analytic))]
#                plt.plot(r, T1['g'])
#                plt.plot(r, T2['g'])
#                plt.plot(r, analytic)
#                plt.show()
                good1.append(i)
                good2.append(j)
                break
solver1.eigenvalues = solver1.eigenvalues[good1].real
solver1.eigenvectors = solver1.eigenvectors[:,good1].real
solver2.eigenvalues = solver2.eigenvalues[good2].real
solver2.eigenvectors = solver2.eigenvectors[:,good2].real


# Plot error vs exact eigenvalues
roots = jn_zeros(phi_n, len(solver1.eigenvalues))
analytic_lam = kappa*(roots/R)**2

print('solved', solver1.eigenvalues)
print('roots', analytic_lam)
print('diff', 1 - solver1.eigenvalues/analytic_lam)
