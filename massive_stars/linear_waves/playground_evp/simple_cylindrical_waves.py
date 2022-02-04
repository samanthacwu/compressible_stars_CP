import dedalus.public as de
import numpy as np
import matplotlib.pyplot as plt
import logging
import time
logger = logging.getLogger(__name__)

from scipy.special import jv, jn_zeros, yv

from scipy.special import erf
def one_to_zero(x, x0, width=0.1):
    return (1 - erf( (x - x0)/width))/2

def zero_to_one(*args, **kwargs):
    return -(one_to_zero(*args, **kwargs) - 1)

# Domain
Nr = 64
R = 1

phi_n = 1
g = 1
S = 100
N = np.sqrt(g*S)
def EVP(this_nr):
    print('solving evp with nr = {}'.format(this_nr))
    r_basis = de.Chebyshev('r', this_nr, interval=(0, R))
    domain = de.Domain([r_basis], np.complex128)

    # Problem
    problem = de.EVP(domain, variables=['p', 'up', 'ur', 'T'],eigenvalue='om')

    problem.parameters['phi_n'] = phi_n
    problem.parameters['S']  = S
    problem.parameters['g'] = g
    problem.substitutions['dt(A)'] = '-1j*om*A'
    problem.substitutions['dp(A)'] = '1j*phi_n*A'

    problem.add_equation("r*( dp(up) + dr(r*ur) ) = 0", tau=False)
    problem.add_equation("r*(dt(up)  + dp(p)/r)     = 0")
    problem.add_equation("    dt(ur) + dr(p)  - g*T*r**2 = 0")
    problem.add_equation("    dt(T) + ur*S = 0")
#    problem.add_bc("left(ur) = 0")
    problem.add_bc("right(ur) = 0")

    # Solver
    solver = problem.build_solver()
    solver.solve_dense(solver.pencils[0])

    # Filter infinite/nan eigenmodes
    finite = np.isfinite(solver.eigenvalues)
    solver.eigenvalues = solver.eigenvalues[finite]
    solver.eigenvectors = solver.eigenvectors[:, finite]

    # Sort eigenmodes by eigenvalue
    order = np.argsort(-solver.eigenvalues.real)
    solver.eigenvalues = solver.eigenvalues[order]
    solver.eigenvectors = solver.eigenvectors[:, order]

    # If solving for waves, only choose the ones with a real oscillation frequency (they're doubled up)
    if S != 0:
        positive = solver.eigenvalues.real > 0
        solver.eigenvalues = solver.eigenvalues[positive]
        solver.eigenvectors = solver.eigenvectors[:, positive]
    
    return solver

solver1 = EVP(Nr)
solver2 = EVP(int(1.5*Nr))

print(solver1.eigenvalues)
print(solver2.eigenvalues)

cot = lambda x: np.cos(x)/np.sin(x)

good1 = []
good2 = []
cutoff = 1e-4
cutoff2 = np.sqrt(cutoff)
for i, ev1 in enumerate(solver1.eigenvalues):
    for j, ev2 in enumerate(solver2.eigenvalues):
        if np.abs((ev1 - ev2))/np.abs(ev1) < cutoff:
            print(ev1.imag, ev2.imag, np.abs((ev1 - ev2))/np.abs(ev1))
            solver1.set_state(i)
            solver2.set_state(j)
            w1 = solver1.state['ur']
            u1 = solver1.state['up']
            w2 = solver2.state['ur']
            w1.set_scales(3/2)
            u1.set_scales(3/2)
            ix = np.argmax(np.abs(w1['g']))
            u1['g'] /= w1['g'][ix]
            w1['g'] /= w1['g'][ix]
            ix = np.argmax(np.abs(w2['g']))
            w2['g'] /= w2['g'][ix]

#            #make sure they're not out of phase
#            if np.allclose(w1['g'][ix]/w2['g'][ix].real, -1):
#                w2['g'] *= -1

            vector_diff = np.max(np.abs(w1['g'] - w2['g']))
            print(vector_diff)
            if vector_diff < cutoff2:
#                print(w1['g'][ix]/w2['g'][ix])
#                r = solver1.domain.grid(0, scales=3/2)
#                plt.plot(r, w1['g'].real)
##                plt.plot(r, w2['g'].real)
#                analytic = (1/r)*jv(phi_n, phi_n*np.sqrt(g*S)*r/ev1.real)
#                analytic /= analytic[np.argmax(np.abs(analytic))]
#                plt.plot(r, analytic, label='analytic', ls='--')
#                plt.legend()
#                plt.title(ev1)
#                plt.show()
                good1.append(i)
                good2.append(j)
                break
solver1.eigenvalues = solver1.eigenvalues[good1]
solver1.eigenvectors = solver1.eigenvectors[:,good1]
solver2.eigenvalues = solver2.eigenvalues[good2]
solver2.eigenvectors = solver2.eigenvectors[:,good2]

# Plot error vs exact eigenvalues
roots = jn_zeros(phi_n, len(solver1.eigenvalues))
analytic_lam = phi_n*np.sqrt(g*S)*R/roots

print('solved', solver1.eigenvalues.real)
print('roots', analytic_lam)
print('diff', 1 - solver1.eigenvalues.real/analytic_lam)


