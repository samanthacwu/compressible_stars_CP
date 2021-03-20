import dedalus.public as de
import numpy as np
import matplotlib.pyplot as plt
import logging
import time
logger = logging.getLogger(__name__)

from scipy.special import erf
def one_to_zero(x, x0, width=0.1):
    return (1 - erf( (x - x0)/width))/2

def zero_to_one(*args, **kwargs):
    return -(one_to_zero(*args, **kwargs) - 1)

# Domain
Nz = 32
L = 1

kx = 1
S0 = 1e4
nu = 1
Pr = 1
kappa = Pr*nu

stress_free = False


def EVP(this_nz):
    print('solving evp with nz = {}'.format(this_nz))
    z_basis = de.Chebyshev('z', this_nz, interval=(0, L))
    domain = de.Domain([z_basis], np.complex128)
    z = domain.grid(0)
    S = S0

    # Problem
    problem = de.EVP(domain, variables=['p', 'u', 'w', 'T', 'u_z', 'w_z', 'T_z'],eigenvalue='om')

    problem.parameters['kx'] = kx
    problem.parameters['kappa'] = kappa
    problem.parameters['nu'] = nu
    problem.parameters['S']  = S
    problem.substitutions['dt(A)'] = '-1j*om*A'
    problem.substitutions['dx(A)'] = '1j*kx*A'

    problem.add_equation("dx(u) + w_z = 0")
    problem.add_equation("dt(u) + dx(p) -  nu*(dx(dx(u)) + dz(u_z)) = 0")
    problem.add_equation("dt(w) + dz(p) -  nu*(dx(dx(w)) + dz(w_z)) - T = 0")
    problem.add_equation("dt(T) + w*S - kappa*(dx(dx(T)) + dz(T_z)) = 0")
    problem.add_equation("w_z - dz(w) = 0")
    problem.add_equation("u_z - dz(u) = 0")
    problem.add_equation("T_z - dz(T) = 0")
    problem.add_bc("left(w) = 0")
    problem.add_bc("right(w) = 0")
    if stress_free:
        print('using stress-free BCs')
        problem.add_bc("left(u_z) = 0")
        problem.add_bc("right(u_z) = 0")
    else:
        print('using no-slip BCs')
        problem.add_bc("left(u) = 0")
        problem.add_bc("right(u) = 0")
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

solver1 = EVP(Nz)
solver2 = EVP(int(1.5*Nz))

print(solver1.eigenvalues)

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
                print('failed ev', ev1, w1['g'][ix]/w2['g'][ix])
            else:
                if np.abs(ev1.real) > 1e-6:
                        
                    print(w1['g'][ix]/w2['g'][ix])

                    dz2w = w2.differentiate('z').differentiate('z')
                    dz4w = w2.differentiate('z').differentiate('z').differentiate('z').differentiate('z')
                    dz6w = w2.differentiate('z').differentiate('z').differentiate('z').differentiate('z').differentiate('z').differentiate('z')
                    print('derivs', dz2w.interpolate(z=0)['g'].max(), dz4w.interpolate(z=0)['g'].max())
                    z = solver1.domain.grid(0, scales=3/2)
#                    plt.plot(z, w2.differentiate('z')['g'])
                    plt.plot(z, nu*kx**(-2)*dz4w['g'] + (1j*ev1/kx**2 - 2*nu)* dz2w['g'])
                    plt.show()
                    plt.plot(z, w1['g'].real/w1['g'].real[np.argmax(np.abs(w1['g'].real))] - np.sin(np.pi*z), lw=4)
                    if kx == 1 and S0 == 1e4 and nu == 1 and False:
                        k1 = 3.90282 - 5.40709j
                        k2 = -0.32385 - 2.94051j
                        k3 = 4.25685 - 2.31346j

                        c01 = -1
                        c02 = -0.675693 + 0.774917j
                        c03 = -72.6807 + 14.473j

                        kz1 = k3
                        c1 = c03
                        kz2 = -kz1 
                        c2 = -c1

                        kz3 = k2
                        c3 = 0.605291 - 1.97737j
                        kz4 = -kz3
                        c4 = 0.696427 + 0.828136j

                        kz5 = k1
                        c5 = 0.0000735018 + 0.000400747j
                        kz6 = -kz5
                        c6 = 1

                        c3 = c4 = c5 = c6 = 0

                        eigvec = np.zeros_like(z, dtype=np.complex128)
                        for c, k in zip((c1, c2, c3, c4, c5, c6), (kz1, kz2, kz3, kz4, kz5, kz6)):
                            eigvec += c*np.exp(k*z)
                        maxi = np.argmax(np.abs(eigvec))
                        eigvec /= eigvec[maxi]
                        print('eigvec', eigvec)
                        plt.plot(z, eigvec)
                        plt.plot(z, np.sin(np.pi*z))
    #                plt.plot(z, w2['g'].real)
    #                plt.plot(z, dz2w['g'].real/dz2w['g'].real[np.argmax(np.abs(dz2w['g'].real))], lw=3)
    #                plt.plot(z, dz4w['g'].real/dz4w['g'].real[np.argmax(np.abs(dz4w['g'].real))], lw=2)
    #                plt.plot(z, dz6w['g'].real/dz6w['g'].real[np.argmax(np.abs(dz6w['g'].real))])
                    plt.axhline(0, c='k')
                    plt.title(ev1)
                    plt.show()
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
ks = np.sqrt(kx**2 + kzs**2)
print(ks)
#exact_eigenvalues = 0.5*(-1j*kappa*ks**2 + np.sqrt(np.array(-ks**4*kappa**2 + 4*S0*kx**2/ks**2,dtype=np.complex128)))
exact_eigenvalues = 0.5*(-1j*kappa*ks**2 - np.sqrt(np.array(-ks**4*kappa**2 + 4*S0*kx**2/ks**2,dtype=np.complex128)))
eval_relative_error = (solver1.eigenvalues.imag - exact_eigenvalues.imag) / exact_eigenvalues.imag

print('analytic', exact_eigenvalues)
print('solved', solver1.eigenvalues)
print('ratio_imag', solver1.eigenvalues.imag/exact_eigenvalues.imag)
print('ratio_real', solver1.eigenvalues.real/exact_eigenvalues.real)
print('pure dissipation?', kappa**2 > (4*S0*kx**2/ks**6))
#print('lambda/nu > kx**2', -solver1.eigenvalues.imag/nu > kx**2, -solver1.eigenvalues.imag, kx**2)

plt.figure()
plt.semilogy(mode_number, np.abs(eval_relative_error), lw=0, marker='o')
plt.xlabel("Mode number")
plt.ylabel(r"$|\lambda - \lambda_{exact}|/\lambda_{exact}$")
plt.title(r"Eigenvalue relative error ($N_z=%i$)" % Nz)
plt.savefig('eval_error.png')
#plt.show()
