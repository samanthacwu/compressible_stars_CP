import numpy as np
import matplotlib.pyplot as plt
import logging
import time

from dedalus.tools.parsing import split_equation
from dedalus.core import coords, distributor, basis, field, operators, problems, solvers, timesteppers, arithmetic

logger = logging.getLogger(__name__)

from scipy.special import erf
def one_to_zero(x, x0, width=0.1):
    return (1 - erf( (x - x0)/width))/2

def zero_to_one(*args, **kwargs):
    return -(one_to_zero(*args, **kwargs) - 1)

do_erf = True
S0 = 1e6
Nmax = 95
Lmax = 1

# Domain
R = 1
r_transition = 0.9*R
width = 0.05*R

ell = 1
g = 1
N = np.sqrt(g*S0)
dtype = np.complex128
dealias = 1

def EVP(this_Nmax):
    print('solving evp with nmax = {}'.format(this_Nmax))

    # Bases
    c    = coords.SphericalCoordinates('φ', 'θ', 'r')
    d    = distributor.Distributor((c,), mesh=None)
    b   = basis.BallBasis(c, (2*(Lmax+1), Lmax+1, this_Nmax+1), radius=R, dtype=dtype)
    b_top = b.S2_basis(radius=R)
    φ,  θ,  r = b.local_grids((dealias, dealias, dealias))
    shell_ell = b.local_ell
    shell_m = b.local_m

    weight_φ = np.gradient(φ.flatten()).reshape(φ.shape)
    weight_θ = b.local_colatitude_weights(dealias)
    weight_r = b.radial_basis.local_weights(dealias)
    weight = weight_θ * weight_φ
    volume = np.sum(weight)

    #Operators
    div       = lambda A: operators.Divergence(A, index=0)
    lap       = lambda A: operators.Laplacian(A, c)
    grad      = lambda A: operators.Gradient(A, c)
    dot       = lambda A, B: arithmetic.DotProduct(A, B)
    curl      = lambda A: operators.Curl(A)
    cross     = lambda A, B: arithmetic.CrossProduct(A, B)
    trace     = lambda A: operators.Trace(A)
    transpose = lambda A: operators.TransposeComponents(A)
    radComp   = lambda A: operators.RadialComponent(A)
    angComp   = lambda A, index=1: operators.AngularComponent(A, index=index)
    LiftTau   = lambda A: operators.LiftTau(A, b, -1)

    # Fields
    u    = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=dtype)
    p    = field.Field(dist=d, bases=(b,), dtype=dtype)
    T   = field.Field(dist=d, bases=(b,), dtype=dtype)
    t_top = field.Field(dist=d, bases=(b_top,), dtype=dtype)

    #nccs
    S      = field.Field(dist=d, bases=(b.radial_basis,), tensorsig=(c,), dtype=dtype)
    grav   = field.Field(dist=d, bases=(b.radial_basis,), tensorsig=(c,), dtype=dtype)
    if do_erf:
        S['g'][2] = S0*zero_to_one(r, r_transition, width=width)
        S['c'][:,:,:,-1] = 0 #zero out last coefficient for a better idea of what this looks like
        #Plot to check goodness of coeff expansion
        plt.plot(r.flatten(), S['g'][2][0,0,:])
        plt.yscale('log')
        plt.show()
    else:
        S['g'][2] = S0#*r**2
    grav['g'][2] = - g*r**2

    #useful operators
    E = 0.5*(grad(u) + transpose(grad(u)))
    E.store_last = True
    gradT = grad(T)
    gradT.store_last = True

    #Impenetrable, stress-free boundary conditions
    u_r_bc        = radComp(u(r=R))
    u_perp_bc_top = radComp(angComp(E(r=R), index=1))

    #Eigenvalue substitution
    omega = field.Field(name='omega', dist=d, dtype=dtype)
    ddt       = lambda A: -1j * omega * A

    # Problem
    these_locals = locals()
    def eq_eval(eq_str, namespace=these_locals):
        for k, i in namespace.items():
            try:
                locals()[k] = i 
            except:
                print('failed {}'.format(k))
        exprs = []
        for expr in split_equation(eq_str):
            exprs.append(eval(expr))
        return exprs

    problem = problems.EVP([p, u, T, t_top], omega)

    ### Ball momentum
    problem.add_equation(eq_eval("div(u) = 0"), condition="nθ != 0")
    problem.add_equation(eq_eval("ddt(u) + grad(p) + grav*T  = 0"), condition = "nθ != 0")
    ## ell == 0 momentum
    problem.add_equation(eq_eval("p = 0"), condition="nθ == 0")
    problem.add_equation(eq_eval("u = 0"), condition="nθ == 0")
    ### Ball energy
    problem.add_equation(eq_eval("ddt(T) + dot(u, S)  + LiftTau(t_top)= 0 "))

    #Velocity BCs ell != 0
    problem.add_equation(eq_eval("radComp(u(r=R))    = 0"),                      condition="nθ != 0")
    # velocity BCs ell == 0
    problem.add_equation(eq_eval("t_top     = 0"), condition="nθ == 0")

    logger.info("Problem built")
    # Solver
    print(problem.dtype)
    solver = solvers.EigenvalueSolver(problem)
    logger.info("solver built")

    for subproblem in solver.subproblems:
        this_ell = subproblem.group[1]
        if this_ell != ell:
            continue
        #TODO: Output to file.
        logger.info("solving ell = {}".format(ell))
        solver.solve_dense(subproblem)
        break

    for subsystem in solver.subsystems:
        ss_m, ss_ell, r_couple = subsystem.group
        if ss_ell == ell and ss_m == 0:
            good_subsys = subsystem
            break

    # Filter infinite/nan eigenmodes
    finite = np.isfinite(solver.eigenvalues)
    solver.eigenvalues = solver.eigenvalues[finite]
    solver.eigenvectors = solver.eigenvectors[:, finite]

    # Sort eigenmodes by eigenvalue
#    order = np.argsort(-solver.eigenvalues.real)
    order = np.argsort(-solver.eigenvalues.imag)
    solver.eigenvalues = solver.eigenvalues[order]
    solver.eigenvectors = solver.eigenvectors[:, order]

    # If solving for waves, only choose the ones with a real oscillation frequency (they're doubled up)
    if S != 0:
        positive = solver.eigenvalues.real > 0
        solver.eigenvalues = solver.eigenvalues[positive]
        solver.eigenvectors = solver.eigenvectors[:, positive]
    
    return solver, locals()

solver1, namespace1 = EVP(Nmax)
solver2, namespace2 = EVP(int(1.5*(1+Nmax)-1))

good1 = []
good2 = []
cutoff = 1e-4
cutoff2 = np.sqrt(cutoff)
for i, ev1 in enumerate(solver1.eigenvalues):
    for j, ev2 in enumerate(solver2.eigenvalues):
        if np.abs((ev1 - ev2))/np.abs(ev1) < cutoff:
            print(ev1.imag, ev2.imag, np.abs((ev1 - ev2))/np.abs(ev1))
            solver1.set_state(i, namespace1['good_subsys'])
            solver2.set_state(j, namespace2['good_subsys'])
            u1 = namespace1['u']
            u2 = namespace2['u']
            u1.set_scales((1, 1, (1.5*(1+Nmax))/(1+Nmax)))
            shape = u1['g'][2].shape
            ix = np.unravel_index(np.argmax(np.abs(u1['g'][2])), shape)
            u1['g'] /= u1['g'][2][ix]
            ix = np.unravel_index(np.argmax(np.abs(u2['g'][2])), shape)
            u2['g'] /= u2['g'][2][ix]

            #make sure they're not out of phase
            if np.allclose(u1['g'][2][ix]/u2['g'][2][ix].real, -1):
                u2['g'] *= -1

            vector_diff = np.max(np.abs(u1['g'][2] - u2['g'][2]))
#            print(vector_diff)
            if vector_diff < cutoff2:
                if np.abs(ev1.real) > 1e-10:
                    r = namespace2['r']#solver1.domain.grid(0, scales=3/2)
                    plt.plot(r.flatten(), u1['g'][2][ix[0], ix[1], :].real)
                    plt.plot(r.flatten(), u2['g'][2][ix[0], ix[1], :].real)
                    plt.title(ev1)
                    plt.show()
                good1.append(i)
                good2.append(j)
                break
solver1.eigenvalues = solver1.eigenvalues[good1]
solver1.eigenvectors = solver1.eigenvectors[:,good1]
solver2.eigenvalues = solver2.eigenvalues[good2]
solver2.eigenvectors = solver2.eigenvectors[:,good2]

print('good eigenvalues', solver1.eigenvalues)
