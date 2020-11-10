"""
Dedalus script for the Lane-Emden equation.

This is a 1D script and should be ran serially.  It should converge within
roughly a dozen iterations, and should take under a minute to run.

In astrophysics, the Lane–Emden equation is a dimensionless form of Poisson's
equation for the gravitational potential of a Newtonian self-gravitating,
spherically symmetric, polytropic fluid [1].

It is usually written as:
    dr(dr(f)) + (2/r)*dr(f) + f**n = 0
    f(r=0) = 1
where n is the polytropic index, and the equation is solved over the interval
r=[0,R], where R is the n-dependent first zero of f(r). Although the equation
is second order, it is singular at r=0, and therefore only requires a single
outer boundary condition.

Following [2], we rescale the equation by defining r=R*x:
    dx(dx(f)) + (2/x)*dx(f) + (R**2)*(f**n) = 0
    f(x=0) = 1
    f(x=1) = 0
This is a nonlinear eigenvalue problem over the interval x=[0,1], with the
additional boundary condition fixing the eigenvalue R.

References:
    [1]: http://en.wikipedia.org/wiki/Lane–Emden_equation
    [2]: J. P. Boyd, "Chebyshev spectral methods and the Lane-Emden problem,"
         Numerical Mathematics Theory (2011).

"""

import time
import numpy as np
import matplotlib.pyplot as plt

from dedalus import public as de

import logging
logger = logging.getLogger(__name__)

from scipy.special import erf
def one_to_zero(x, x0, width=0.1):
    return (1 - erf( (x - x0)/width))/2

def zero_to_one(*args, **kwargs):
    return -(one_to_zero(*args, **kwargs) - 1)

# Parameters
Nx = 256
n = 1.5
n_outer = 1.6
ncc_cutoff = 1e-6
tolerance = 1e-12
transition = 0.4


# Build domain
x_basis = de.Chebyshev('x', Nx, interval=(0, 1), dealias=2)
domain = de.Domain([x_basis], np.float64)

n_ncc = domain.new_field()
n_ncc['g'] = zero_to_one(domain.grid(0), transition, width=0.05)*(n_outer-n) + n

# Setup problem
problem = de.NLBVP(domain, variables=['f', 'fx', 'R'], ncc_cutoff=ncc_cutoff)
#problem.meta['R']['x']['constant'] = True
problem.parameters['n'] = n_ncc
#problem.parameters['n'] = n
problem.add_equation("x*dx(fx) + 2*fx = -x*(R**2)*exp(n*log(f))", tau=False)
problem.add_equation("fx - dx(f) = 0")
problem.add_equation("dx(R) = 0")
problem.add_bc("left(f) = 1")
problem.add_bc("right(f) = 0")

# Setup initial guess
solver = problem.build_solver()
x = domain.grid(0)
f = solver.state['f']
fx = solver.state['fx']
R = solver.state['R']
f['g'] = np.cos(np.pi/2 * x)*0.9
f.differentiate('x', out=fx)
R['g'] = 3

# Iterations
pert = solver.perturbations.data
pert.fill(1+tolerance)
start_time = time.time()
while np.sum(np.abs(pert)) > tolerance:
    solver.newton_iteration()
    logger.info('Perturbation norm: {}'.format(np.sum(np.abs(pert))))
    logger.info('R iterate: {}'.format(R['g'][0]))
end_time = time.time()

# Compare to reference solutions from Boyd
R_ref = {0.0: np.sqrt(6),
         0.5: 2.752698054065,
         1.0: np.pi,
         1.5: 3.65375373621912608,
         2.0: 4.3528745959461246769735700,
         2.5: 5.355275459010779,
         3.0: 6.896848619376960375454528,
         3.25: 8.018937527,
         3.5: 9.535805344244850444,
         4.0: 14.971546348838095097611066,
         4.5: 31.836463244694285264}
logger.info('-'*20)
logger.info('Iterations: {}'.format(solver.iteration))
logger.info('Run time: %.2f sec' %(end_time-start_time))
logger.info('Final R iteration: {}'.format(R['g'][0]))
if n in R_ref:
    logger.info('Error vs reference: {}'.format(R['g'][0]-R_ref[n]))


r = domain.grid(0, scales=2)
xi = r*R['g']

#plt.plot(r, f['g'])
#plt.plot(r, 1 - ((xi)**2)/6, ls='--') # n = 0
#plt.plot(r, np.sin(xi)/xi, ls='--')   # n = 1
#plt.show()

plt.show()
gamma = 5./3
T     = np.concatenate([[1,], np.copy(f['g'])])
rho   = np.concatenate([[1,], np.copy(f['g']**(n_ncc['g']))])

gradT        = np.concatenate([[0,], f.differentiate('x')['g']])
grad_ln_T    = gradT/T
grad_ln_rho  = np.concatenate([[n,], n_ncc['g']])*grad_ln_T

r_concatenate = np.concatenate([[0,], r]) / transition

#plt.plot(r_concatenate, np.log(rho))
#plt.ylim(-5, 0.1)
#plt.show()

gradS  = (1/gamma)*grad_ln_T - ((gamma-1)/gamma) * grad_ln_rho
#plt.plot(r_concatenate, gradS)
#plt.yscale('log')
#plt.show()
L = domain.new_field()
dLdr = domain.new_field()
L.set_scales(2, keep_data=True)

H_eff = np.exp(-r_concatenate**2/(2*0.4**2))
plt.plot(r_concatenate, gradS)
#plt.plot(r_concatenate, H_eff)
plt.yscale('log')
plt.show()

import h5py
with h5py.File('poly_nOuter{}.h5'.format(n_outer), 'w') as f:
    Mach = 0.01
    f['r'] = r_concatenate
    f['T'] = T
    f['ρ'] = rho
    f['grad_s0'] = gradS/Mach**2
    f['H_eff']   = H_eff
    
    
