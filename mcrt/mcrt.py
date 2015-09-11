from __future__ import division
import numpy as np
import scipy.integrate, scipy.special

dt = np.float

# Boltzmann's constant
kB = 1.3806488e-23 # J / K

# speed of light
c = 2.99792458e8 # m / s

def freq_param( num_freqs=24 ):
    """Frequency parameter x = ( nu - nu_0 ) / nu_0"""
    # the 1e-6 padding is to prevent division errors in the extinction coefficients
    return np.array( [ 2.25 * (i + 1e-6) / num_freqs for i in xrange(num_freqs-1) ] + [ np.inf ], dtype=dt ).reshape( 1, num_freqs )

def frequency( wavelength ):
    """Changes wavelength to frequency"""
    return c / wavelength

def sdu( m, T ):
    """Doppler unit for mass `m` and temperature `T`.
    The standard deviation for a doppler-broadened line centered at 'nu' is `sdu( m, T ) * nu` """
    return np.sqrt( kB * T / m / c**2 )

class lineshape( object ):
    def __init__( self, mean, std, prof_type='doppler' ):
        self.mean = mean
        self.std  = std
        self.prof_type = prof_type

    @property
    def cdf( self ):
        """Returns an array of the cdf"""
        return scipy.special.erf( freq_param() / self.std / np.sqrt(2) )

class Gas( object ):
    """Hold cross section info for each species.
    `Gas.sigma(T)` will return the temperature-dependent cross sections at T kelvin"""
    species = {}
    absorbers = {}
    scatterers = {}
    def __init__(self, name, role, wavelengths, cross_sections, mass ):
        """
        `role`           = 'absorption' or 'scattering'
        `wavelengths`    = wavelengths corresponding to `cross_sections`
        `cross_sections` = reference cross sections at 1000 K
        """
        self.name = name
        self.role = role
        self.wavelengths = wavelengths
        self.mass = mass
        if role == 'absorption':
            # self.sigma = [ ( lambda T: s ) for s in cross_sections ]
            self.sigma = cross_sections
            self.species[name] = self
            self.absorbers[name] = self
        elif role == 'scattering':
            # self.sigma = [ ( lambda T: s * np.sqrt( 1000 / T )  ) for s in cross_sections ]
            self.sigma = cross_sections
            self.scatterers[name] = self
            self.species[name] = self
        else:
            raise Exception( "role {} not recognized".format( role ) )

class Atmosphere( object ):
    """Contains atmospheric properties and methods for calculating RT."""
    def __init__( self, altitudes, angles, temperatures, neutrals_dict, oplus_density, viewing_angles, wavelengths=[832, 833, 834] ):
        # coordinates
        z = altitudes * 1e5 # km -> cm
        self.z   = z
        self.mu, self.leg_weights  = best_angles(np.cos(angles))
        # self.mu_e = -np.cos( viewing_angles )
        self.mu_e = np.cos(viewing_angles*np.pi/180.)
        self.ergodic_weights = np.ones(len(viewing_angles))

        self.N_angles = len(angles)
        self.N_layers = len(altitudes)
        self.N_lambda = len(wavelengths)

        # phase function
        self.P = P        

        # inputs
        self.neutrals = neutrals_dict
        self.oplus = oplus_density
        self.temperatures = temperatures
        self.viewing_angles = viewing_angles

        # layer thickness
        self.dz = self.z[:-1] - self.z[1:]

        m_oplus = Gas.species["O+"].mass
        self.tau, self.dtau, self.albedo, self.tau_s, self.lineshape, self.extinction = [],[],[],[],[],[]
        for L in xrange( self.N_lambda ):
            nu = frequency(wavelengths[L])
            ls = lineshape( nu, nu*sdu( m_oplus, self.temperatures[:,None] ) ) 
            spec = ls.cdf
            spec_chunks = spec[:,1:] - spec[:,:-1]
            self.lineshape.append( spec_chunks[:] )

            scatter_coeff = np.zeros( ( self.N_layers, spec_chunks.size ), dtype=dt )
            absorb_coeff  = np.zeros_like( scatter_coeff, dtype=dt )
            # optical depths -- each of these is a list with an array for each wavelength
            scatter_coeff = self.oplus[:, None] * Gas.species["O+"].sigma[L] * np.sqrt( 1000 / self.temperatures[:, None] ) * spec_chunks
            absorb_coeff = sum( self.neutrals[ gas.name ] * gas.sigma[L] for gas in Gas.absorbers.values() )[:,None] * np.ones_like(spec_chunks)

            # minus signs because z is decreasing
            tau_a, tau_s , tau = np.zeros_like(scatter_coeff), np.zeros_like(scatter_coeff), np.zeros_like(scatter_coeff)

            # geometric mean makes for a better estimate, I think. <======= FIGURE OUT IF THIS IS TRUE
            gmean_abs = np.sqrt( absorb_coeff[:-1,:] * absorb_coeff[1:,:] )
            gmean_sct = np.sqrt( scatter_coeff[:-1,:] * scatter_coeff[1:,:] )

            dtau_a, dtau_s = np.zeros_like(scatter_coeff), np.zeros_like(scatter_coeff)
            dtau_a[1:,:] = gmean_abs * abs(z[:-1]-z[1:])[:,None]
            dtau_s[1:,:] = gmean_sct * abs(z[:-1]-z[1:])[:,None]

            # initial depths should be taken from infinity
            dtau_a[0,:] = abs((z[0] - z[1]) * absorb_coeff[0,:] / np.log(absorb_coeff[1,:]/absorb_coeff[0,:]))
            dtau_s[0,:] = abs((z[0] - z[1]) * scatter_coeff[0,:] / np.log(scatter_coeff[1,:]/scatter_coeff[0,:]))

            tau_a = np.cumsum(dtau_a, axis=0)
            tau_s = np.cumsum(dtau_s, axis=0)

            tau = tau_a + tau_s
            self.tau_s.append(tau_s)
            self.tau.append( tau ) 
            
            dtau = dtau_a + dtau_s
            self.dtau.append( dtau )
            
            alb = np.zeros_like( tau )
            alb[:-1] = gmean_sct / ( gmean_sct + gmean_abs )
            self.albedo.append( alb )

            self.extinction.append(gmean_sct + gmean_abs)

        # states for easy indexing later
        self.transient = [ ( i, n ) for i in xrange(self.N_angles) for n in xrange(self.N_layers) ]
        self.ergodic   = range( len( viewing_angles ) )

        # convenience definitions
        self.tau_0 = [ t[-1,:] for t in self.tau ]

    def multiple_scatter_matrix( self, wavelength, view_height=None ):
        """The multiple scattering matrix maps the initial distribution of instensity to the final, scattered distribution"""
        n = len(self.transient)
        r = len(self.ergodic  )
        I = np.eye( n )
        Q = np.fromiter( ( self.Q( k, l, wavelength ) for k in self.transient for l in self.transient ), dtype=dt )
        Q = Q.reshape( ( n, n ) )

        if not view_height: 
            return np.linalg.inv( I -  Q ) # multiple scattering matrix, slow to calculate

        # calculate reference depth for viewing height
        self.calculate_view_depth(view_height)

        R = self.R(wavelength, view_height)

        # ( I - Q ) X = R
        X = np.linalg.solve( I - Q, R )

        return X

    def Q( self, k, l, wavelength ):
        """Transition probability for transient state `k` -> transient state `l`"""
        # angles
        i = k[0]
        j = l[0]
        # layers
        n = k[1]
        m = l[1]
        # sum is over frequency, for complete frequency redistribution
        return np.sum( self.lineshape[wavelength][n,:] * self.W( n, i, m, wavelength ) * self.Ms( i, j, m, wavelength ) )

    def W( self, n, i, m, wavelength ):
        """Exctinction probability for layer `n` -> `m` along direction `i`"""
        w = wavelength
        mu_i    = self.mu[i]
        dtau_n  = self.dtau[w][n]
        dtau_m  = self.dtau[w][m]
        c = self.leg_weights[i]
        # upwelling
        if n > m:
            if np.sign( mu_i ) == -1: return 0
            tau_n   = self.tau[w][n]
            tau_mp1 = self.tau[w][m+1]
            mu = mu_i
            return mu / dtau_n * ( 1 - np.exp( - dtau_n / mu ) ) * \
                   np.exp( -( tau_n - tau_mp1 ) / mu ) * \
                   ( 1 - np.exp( - dtau_m / mu ) ) * c
        # downwelling
        elif n < m: 
            if np.sign( mu_i ) == +1: return 0
            mu = -mu_i
            tau_m   = self.tau[w][m]
            tau_np1 = self.tau[w][n+1]
            return mu / dtau_n * ( 1 - np.exp( - dtau_n / mu ) ) * \
                   np.exp( -( tau_m - tau_np1 ) / mu ) * \
                   ( 1 - np.exp( - dtau_m / mu ) ) * c
        # n == m, recapture
        else: 
            mu = abs( mu_i )
            return 1 - mu / dtau_n * ( 1 - np.exp( - dtau_n / mu ) )  * c

    def Ms( self, i, j, n, wavelength ):
        """Scattering probability for `i` -> `j` in layer `n`"""
        return self.albedo[wavelength][n] * self.P( self.mu[i], self.mu[j] ) * self.leg_weights[i] 

    def R(self, wavelength, view_height):
        self.calculate_view_depth(view_height)
        transient_angles = self.mu
        ergodic_angles = self.mu_e
        ergodic_weights = self.ergodic_weights
        albedo = self.albedo[wavelength]
        tau = self.tau[wavelength]
        view_depth = self.view_depth[wavelength]
        view_ind = self.view_ind[wavelength]
        lineshape = self.lineshape[wavelength]
        out = R_matrix(transient_angles,ergodic_angles,ergodic_weights,albedo,tau,view_depth,view_ind,lineshape)
        return out

    def calculate_view_depth(self,view_height):
        """Returns the optical depth associated with the altitude `view_height` km."""
        h = view_height*1e5
        upper = self.z[0]
        lower = self.z[-1]
        upper_ind = 0
        lower_ind = -1
        for i,z in enumerate(self.z):
            if z > h:
                upper = z
                upper_ind = i
            else: 
                lower = z
                lower_ind = i
                break

        DZ = upper - lower
        upper_weight = (upper-h)/DZ
        lower_weight = (h-lower)/DZ
        view_depth = np.array( [ t[upper_ind,:]*upper_weight + t[lower_ind,:]*lower_weight  for t in self.tau ] )
        # define an index to serve as a proxy for the top of the atmosphere
        view_ind = [ np.max(np.where(np.sum(self.tau[w],axis=1) > np.sum(view_depth[w]))) for w in xrange(self.N_lambda) ]
        self.view_depth = view_depth
        self.view_ind = view_ind
        return self.view_depth

    def _R( self, l, e, wavelength ):
        """Transisition probability from transient state `l` -> ergodic state `e`.
        This is the viewing matrix."""
        w = wavelength
        j = l[0]
        # n = l[1]
        mu_j = self.mu[j]
        mu_e = self.mu_e[e]
        # tau = self.tau[w][n,:]
        dtau = self.dtau[w]
        # ind = np.arange(n+1) # layers with optical depth <= tau
        tau = self.tau[w]
        albedo = self.albedo[w]
        # tau_0 = self.tau_0[w]
        # upwelling
        if np.sign(mu_j) == +1:
            arr = np.vstack([ a  for a in quad_gen_T( mu_j, mu_e, tau, albedo, dtau, self.lineshape[w], self.view_depth[w], self.view_ind[w] ) ]).T
            return  np.sum( arr, axis=1 )
        # downwelling
        elif np.sign(mu_j) == -1:# and n < self.N_layers - 1:
            arr = np.vstack([ a  for a in quad_gen_R( mu_j, mu_e, tau, albedo, dtau, self.lineshape[w], self.view_depth[w], self.view_ind[w] ) ]).T
            return  np.sum( arr, axis=1 )
        else: 
            return 0.0 # assume no surface reflection for now, since this shouldn't matter for 834

def P( mu_i, mu_j ):
    """Phase function"""
    weight = 1 # ONLY FOR TWO-STREAM
    return weight / 2 # 1 / 4 / np.pi

def R1( mu_j, mu_e, tau, tau_prime, albedo, tau_ref=0):
    """Probability for diffuse reflection after one scattering."""
    # http://www.light-measurement.com/reflection-absorption/
    w0 = albedo
    # ind=0
    # for i, tp in enumerate(tau_prime):
    #     if all(tp <= tau_ref) :
    #         ind=i
    integrand = np.exp( -tau_prime * ( 1 / mu_j + 1 / mu_e ) ) * w0 * P( mu_j, mu_e )*np.array([1]) # hacky way to make sure this is an array.
    try:
        return abs( scipy.integrate.trapz( y=integrand, x=tau_prime / mu_j * np.array([1]) ) )
    except IndexError:
        return 0

def T1( mu_j, mu_e, tau, tau_prime, albedo, tau_ref=0 ):
    """Probability for diffuse transmission after one scattering"""
    w0 = albedo
    # ind=0
    # for i, tp in enumerate(tau_prime):
    #     if all(tp <= tau_ref) :
    #         ind=i
    integrand = np.exp( -tau_prime / mu_j ) * w0 * P( mu_j, mu_e ) * np.exp( -( tau - tau_prime ) / mu_e )
    try:
        return abs( scipy.integrate.trapz( y=integrand * np.array([1]), x=tau_prime * np.array([1]) / mu_j ) )
    except IndexError:
        return 0

def R_matrix(transient_angles,ergodic_angles,ergodic_weights,albedo,tau,view_depth,view_ind, lineshape):
    """Construct the viewing matrix."""
    # Assume tau applies to a single line.
    # This makes tau a ~ 100 x 23 numpy array (alt x wavelength)
    # l, e are lists of states
    # Each element l,e is the probability for a photon emitted in the state l to scatter into the state e
    n = tau.shape[0]
    output = np.zeros((len(transient_angles)*n,len(ergodic_angles)))
    topside = xrange(0,view_ind)
    bottomside = xrange(view_ind,n)
    topside_depth = abs(tau[topside,:] - view_depth)
    bottomside_depth = abs(tau[bottomside,:] - view_depth) 
    for e, mu_e in enumerate(ergodic_angles):
        weight = ergodic_weights[e]
        for i in xrange(len(lineshape[0,:])):
            for mu_j in transient_angles:
                # they way that transient is set up, the upwellng angles come first, then downwelling.
                if np.sign(mu_j) < 0: # upwelling
                    below = T_integral(mu_j, mu_e, bottomside_depth[:,i], albedo[bottomside,i])
                    above = R_integral(mu_j, mu_e, topside_depth[:,i], albedo[topside,i])
                    # if height < view_height:
                    #     # calculate transmission for depths below
                    # if height > view_height
                    #     # calculate reflection for depths above
                    # join tops and bottoms
                    # print above.shape, below.shape, weight
                    output[:n,e] += np.hstack([above, below]) * lineshape[:,i] * weight
                elif np.sign(mu_j) > 0: # downwelling
                    above = T_integral(mu_j, mu_e, topside_depth[:,i], albedo[topside,i])
                    below = R_integral(mu_j, mu_e, bottomside_depth[:,i], albedo[bottomside,i])

                    # if height < view_height:
                    #     # calculate reflection
                    # if height > view_height
                    #     # calculate transmission
                    # join tops and bottoms
                    output[n:,e] += np.hstack([above, below]) * lineshape[:,i] * weight
    return output



def T_integral(mu_J, mu_E, tau_range, albedo):
    """A not-shitty version of the diffuse transmission probability"""
    # if np.sign(mu_J) == np.sign(mu_E):
    #     return np.zeros_like(tau_range)
    mu_e, mu_j = abs(mu_E), abs(mu_J)
    tau = max(tau_range) 
    tau_prime = tau_range
    integrand = 1.0/mu_j * albedo * P(mu_j, mu_e) * np.exp(-tau/mu_e -(1.0/mu_j - 1.0/mu_e)*tau_prime) / 2.0
    return scipy.integrate.cumtrapz(integrand, tau_range, initial=0)

def R_integral(mu_J, mu_E, tau_range, albedo):
    """Takes in angles, optical depth, and albedo and returns the integrand for diffuse reflection."""
    # if np.sign(mu_J) != np.sign(mu_E):
    #     return np.zeros_like(tau_range)
    mu_e, mu_j = abs(mu_E), abs(mu_J)
    tau = min(tau_range)
    tau_prime = tau_range
    integrand = 1.0/mu_j * albedo * P(mu_j,mu_e) *\
        np.exp(tau/mu_j -tau_prime*(1.0/mu_j+1.0/mu_e) ) / 2.0
    return scipy.integrate.cumtrapz(integrand, tau_range, initial=0)

def quad_gen_T(mu_j, mu_e, tau, albedo, dtau, ls, view_depth, top ):
    """Generator for diffuse transmission integrals at each point in the broadened line spectrum."""
    N = tau.shape[1]
    tp = tau
    alb = albedo
    tau_view = top
    for line_chunk in xrange(N):
        max_ind = len(tau)
        data = [0.0]*top
        tau_view = view_depth[line_chunk]
        for bottom in xrange(top+1,max_ind-1):
            ind = xrange(top,bottom)
            tp = tau[ind,line_chunk]
            alb = albedo[ind,line_chunk]
            I = T_integrand(mu_j, mu_e, tau_view, tp, alb)
            # t_sub = np.arctan(tau_prime)
            # I2 = T_integrand(mu_j, mu_e, np.arctan(tau_view), t_sub, albedo)/(np.cos(t_sub)**2)
            data.append( trapz(I, tau_prime) * ls[line_chunk] )
        yield np.array(data)

def quad_gen_R(mu_j, mu_e, tau, albedo, dtau, ls, view_depth, top ):
    """Generator for diffuse transmission integrals at each point in the broadened line spectrum."""
    N = tau.shape[1]
    tp = tau
    alb = albedo
    for line_chunk in xrange(N):
        max_ind = len(tau)
        data = [0.0]*top
        tau_view = view_depth[line_chunk]
        for bottom in xrange(top+1,max_ind-1):
            ind = xrange(top,bottom)
            tp = tau[ind,line_chunk]
            alb = albedo[ind,line_chunk]
            I = R_integrand(mu_j, mu_e, tau_view, tp, alb)
            # t_sub = np.arctan(tau_prime)
            # I2 = T_integrand(mu_j, mu_e, np.arctan(tau_view), t_sub, albedo)/(np.cos(t_sub)**2)
            data.append( trapz(I, tau_prime) * ls[line_chunk] )
        yield np.array(data)

def best_angles(angles):
    """Takes in a range of viewing angles and assumes that those angles are silly, \
    replacing them with Legendre-Gauss quadrature of degree equal to the number of angles."""
    a, b = min(angles), max(angles)
    x, w = np.polynomial.legendre.leggauss(len(angles))
    y = ( x * (b-a) + (b+a) )/2.0
    return y, w

def ver_to_inten(volume_emission_rate, extinction_coefficient):
    intensity = volume_emission_rate / extinction_coefficient / 4.0 / np.pi
    return intensity

