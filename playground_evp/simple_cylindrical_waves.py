import dedalus.public as de
import numpy as np
import matplotlib.pyplot as plt
import logging
import time
logger = logging.getLogger(__name__)


# Domain
Nr = 64
L = 1

k_n = 10
aspect = 4
kx = k_n*2*np.pi/(aspect*L)
nu = 1e-2
Pr = 1
kappa = Pr*nu
g = 1
S = 1
N = np.sqrt(g*S)
def EVP(this_nr):
    print('solving evp with nr = {}'.format(this_nr))
    r_basis = de.Chebyshev('r', this_nr, interval=(0, L))
    domain = de.Domain([r_basis], np.complex128)

    # Problem
    problem = de.EVP(domain, variables=['p', 'up', 'ur', 'T', 'up_r', 'ur_r', 'T_r'],eigenvalue='om')

    problem.parameters['kx'] = kx
    problem.parameters['nu'] = nu
    problem.parameters['kappa'] = kappa
    problem.parameters['S']  = S
    problem.parameters['g'] = g
    problem.substitutions['dt(A)'] = '-1j*om*A'
    problem.substitutions['dx(A)'] = '1j*kx*A'

    problem.add_equation("r*(dx(up) + ur_r + ur/r) = 0")
    problem.add_equation("r**2*(dt(up) + dx(p)    - nu*(dx(dx(up))/r**2 + (1/r)*(r*dr(up_r) + up_r) + (2/r**2)*dx(ur) - up/r**2)) = 0")
    problem.add_equation("r**2*(dt(ur) + dr(p)/r  - nu*(dx(dx(ur))/r**2 + (1/r)*(r*dr(ur_r) + ur_r) - (2/r**2)*dx(up) - ur/r**2) - g*T) = 0")
    problem.add_equation("r**2*(dt(T) + ur*S -   kappa*(dx(dx(T))/r**2 +  (1/r)*(r*dr(T_r)  + T_r))) = 0")
    problem.add_equation("up_r - dr(up) = 0")
    problem.add_equation("ur_r - dr(ur) = 0")
    problem.add_equation("T_r - dr(T) = 0")
    problem.add_bc("left(up_r) = 0")
    problem.add_bc("right(up_r) = 0")
    problem.add_bc("left(ur) = 0")
    problem.add_bc("right(ur) = 0")
    problem.add_bc("left(T) = 0")
    problem.add_bc("right(T) = 0")

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

    # If solving for waves, only choose the ones with a real oscillation frequency (they're doubled up)
    if S != 0:
        positive = solver.eigenvalues.real > 0
        solver.eigenvalues = solver.eigenvalues[positive]
        solver.eigenvectors = solver.eigenvectors[:, positive]
    
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
                print(w1['g'][ix]/w2['g'][ix])
                r = solver1.domain.grid(0, scales=3/2)
                plt.plot(r, w1['g'].real)
                plt.plot(r, w2['g'].real)
                plt.show()
                good1.append(i)
                good2.append(j)
                break
#solver1.eigenvalues = solver1.eigenvalues[good1]
#solver1.eigenvectors = solver1.eigenvectors[:,good1]
#solver2.eigenvalues = solver2.eigenvalues[good2]
#solver2.eigenvectors = solver2.eigenvectors[:,good2]
#
## Plot error vs exact eigenvalues
#mode_number = 1 + np.arange(len(solver1.eigenvalues))
#print(mode_number)
#krs = (np.pi*mode_number)/L
#ks = np.sqrt(kx**2 + krs**2)
#exact_eigenvalues = -(kx/ks)*N - 1j * nu * ks**2
#eval_relative_error = (solver1.eigenvalues.imag - exact_eigenvalues.imag) / exact_eigenvalues.imag
#
#print('analytic', exact_eigenvalues)
#print('solved', solver1.eigenvalues)
#print('ratio_imag', solver1.eigenvalues.imag/exact_eigenvalues.imag)
#print('ratio_real', solver1.eigenvalues.real/exact_eigenvalues.real)
##print('lambda/nu > kx**2', -solver1.eigenvalues.imag/nu > kx**2, -solver1.eigenvalues.imag, kx**2)
#
#plt.figure()
#plt.semilogy(mode_number, np.abs(eval_relative_error), lw=0, marker='o')
#plt.xlabel("Mode number")
#plt.ylabel(r"$|\lambda - \lambda_{exact}|/\lambda_{exact}$")
#plt.title(r"Eigenvalue relative error ($N_r=%i$)" % Nr)
#plt.savefig('eval_error.png')
##plt.show()
