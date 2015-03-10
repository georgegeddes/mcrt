from __future__ import division
import numpy as np

dt = np.float

class Gas( object ):
    """Hold cross section info for each species.
    `Gas.sigma(T)` will return the temperature-dependent cross sections at T kelvin"""
    species = {}
    absorbers = {}
    scatterers = {}
    def __init__(self, name, role, wavelengths, cross_sections ):
        """
        `role`           = 'absorption' or 'scattering' 
        `wavelengths`    = wavelengths corresponding to `cross_sections`
        `cross_sections` = reference cross sections at 1000 K
        """
        self.name = name
        self.role = role
        self.wavelengths = wavelengths
        if role == 'absorption':
            self.sigma = [ ( lambda T: s ) for s in cross_sections ]
            self.species[name] = self
            self.absorbers[name] = self
        elif role == 'scattering':
            self.sigma = [ ( lambda T: s * np.sqrt( 1000 / T )  ) for s in cross_sections ]
            self.scatterers[name] = self
            self.species[name] = self
        else:
            raise Exception( "role {} not recognized".format( role ) )



class Atmosphere( object ):
    """Contains atmospheric properties and methods for calculating RT."""
    def __init__( self, altitudes, angles, temperatures, neutrals_dict, oplus_density, viewing_angles ):
        # coordinates
        self.z   = altitudes * 1e5 # km -> cm
        self.mu  = np.cos( angles )

        self.N_angles = len(angles)
        self.N_layers = len(altitudes) - 1

        # inputs
        self.neutrals = { k : ( n[:-1] + n[1:] ) / 2 for (k, n) in neutrals_dict.iteritems() }
        self.oplus = ( oplus_density[:-1] + oplus_density[1:] ) / 2
        self.temperatures = ( temperatures[:-1] + temperatures[1:] ) / 2
        self.viewing_angles = viewing_angles

        # layer thickness
        self.dz = self.z[:-1] - self.z[1:]

        # optical depths -- each of these is a list with an array for each wavelength
        self.albedo = [ self.oplus * self.dz * np.fromiter(( sigma(T) for T in self.temperatures ), dtype=dt ) for sigma in Gas.species["O+"].sigma ]
        dtau_abs = [ sum( self.neutrals[ gas.name ] * gas.sigma[w](None) * self.dz for gas in Gas.absorbers.values() ) for w in xrange(3) ]
        self.dtau = [ self.albedo[w] + dtau_abs[w] for w in xrange(3) ]
        self.tau = [ np.array( [0.0] + [ t for t in np.cumsum( dtau ) ] ) for dtau in self.dtau ]

        # states for easy indexing later
        self.transient = [ ( i, n ) for i in xrange(self.N_angles) for n in xrange(self.N_layers) ]
        self.ergodic   = range( len( viewing_angles ) )

    @property
    def Mm(self, wavelength):
        return self.multiple_scatter_matrix( self, wavelength, viewing=False )

    def multiple_scatter_matrix( self, wavelength, viewing=True ):
        """The multiple scattering matrix maps the initial distribution of instensity to the final, scattered distribution"""
        n = len(self.transient)
        r = len(self.ergodic  )
        Q = np.fromiter( ( self.Q( k, l, wavelength ) for k in self.transient for l in self.transient ), dtype=dt )
        Q = Q.reshape( ( n, n ) )
        Mm = np.linalg.inv( np.eye( n ) -  Q )

        if not viewing: return Mm

        R = np.fromiter( ( self.R( l, e, wavelength ) for l in self.transient for e in self.ergodic ), dtype=dt ) 
        R = R.reshape( ( n, r ) )

        return Mm.dot( R )

    def Q( self, k, l, wavelength ):
        """Transition probability for transient state `k` -> transient state `l`"""
        i = k[0]
        j = l[0]
        n = k[1]
        m = l[1]
        return self.W( n, i, m, wavelength ) * self.Ms( i, j, m, wavelength )

    def W( self, n, i, m, wavelength ):
        """Exctinction probability for layer `n` -> `m` along direction `i`"""
        w = wavelength
        mu_i    = self.mu[i]
        dtau_n  = self.dtau[w][n]
        dtau_m  = self.dtau[w][m]
        # upwelling
        if n > m:
            if np.sign( mu_i ) == -1: return 0
            tau_n   = self.tau[w][n]
            tau_mp1 = self.tau[w][m+1]
            mu = mu_i
            return mu / dtau_n * ( 1 - np.exp( - dtau_n / mu ) ) * \
                   np.exp( -( tau_n - tau_mp1 ) / mu ) * \
                   ( 1 - np.exp( - dtau_m / mu ) )
        # downwelling
        elif n < m: 
            if np.sign( mu_i ) == +1: return 0
            mu = -mu_i
            tau_m   = self.tau[w][m]
            tau_np1 = self.tau[w][n+1]
            return mu / dtau_n * ( 1 - np.exp( - dtau_n / mu ) ) * \
                   np.exp( -( tau_m - tau_np1 ) / mu ) * \
                   ( 1 - np.exp( - dtau_m / mu ) )
        # n == m, recapture
        else: 
            mu = abs( mu_i )
            return 1 - mu / dtau_n * ( 1 - np.exp( - dtau_n / mu ) ) 

    def Ms( self, i, j, n, wavelength ):
        """Scattering probability for `i` -> `j` in layer `n`"""
        weight = 1 # ONLY FOR TWO-STREAM
        return self.albedo[wavelength][n] * self.P( self.mu[i], self.mu[j] ) * weight / 2

    def P( self, mu_i, mu_j ):
        """Phase function"""
        return 1 / 4 / np.pi

    def R( self, l, e, wavelength ):
        """Transisition probability from transient state `l` -> ergodic state `e`.
        This is the viewing matrix."""
        w = wavelength
        j = l[0]
        n = l[1]
        pass 

    def R1( self, j, e, tau ):
        """Probability for diffuse reflection after one scattering"""
        pass

    def T1( self, j, e, tau ):
        """Probability for diffuse transmission after one scattering"""
        pass





