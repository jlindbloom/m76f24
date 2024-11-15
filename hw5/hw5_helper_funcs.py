import numpy as np
from scipy.integrate import quad
from scipy.optimize import root, bisect


def bimodal_logp(x, c, gamma):
    """Evaluates the log of the density function.
    """

    result = np.log(2 * gamma * (c**2 + gamma**2))
    result -= np.log(np.pi * (gamma**2 + (x - c)**2))
    result -= np.log((gamma**2 + (x + c)**2))

    return result


def bimodal_cdf(x, c, gamma):
    """Evaluates the CDF.
    """

    fac = ( (c**2) + (gamma**2) )/( np.pi*(4*(c**3) + 4*c*(gamma**2)  )  )

    p1 = 2*c*np.pi
    p2 = -2*c*np.arctan2( (c-x), gamma )
    p3 = 2*c*np.arctan2( (c+x),gamma )
    p4 = -gamma*np.log( (gamma**2) + ((c-x)**2)  )
    p5 = gamma*np.log( (gamma**2) + ((c+x)**2)  )

    return fac*(p1+p2+p3+p4+p5)


def bimodal_inv_cdf(u, c, gamma):
    """Evaluates the inverse CDF.
    """

    g = lambda z: bimodal_cdf(z, c, gamma) - u

    root_res = bisect(g, -1000*c, 1000*c, maxiter=10000)

    return root_res



def bimodal_distribution(n_samples, gamma, c):
    """Generates samples from the bimodial Cauchy distribution example.
    """

    # Draw samples
    u_samples = []
    x_samples = []
    for i in range(n_samples):
        
        # Draw random uniform
        u = np.random.uniform()
        u_samples.append(u)

        # Apply T
        x = bimodal_inv_cdf(u, c, gamma)
        x_samples.append(x)

    u_samples = np.asarray(u_samples)
    x_samples = np.asarray(x_samples)

    return x_samples

