from scipy.special import erf

def one_to_zero(x, x0, width=0.1):
    """ One minus Smooth Heaviside function (1 - H) """
    return (1 - erf( (x - x0)/width))/2

def zero_to_one(*args, **kwargs):
    """ Smooth Heaviside Function """
    return -(one_to_zero(*args, **kwargs) - 1)


