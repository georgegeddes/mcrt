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
    def __init__(self, name, role, wavelengths, cross_sections, mass, cells=(), temperatures=() ):
        """
        `name`           = name of gas species
        `role`           = 'absorption' or 'scattering'
        `wavelengths`    = wavelengths corresponding to `cross_sections`
        `cross_sections` = reference cross sections at 1000 K
        `mass`           = molecular mass in kg
        `cells`          = list of atmospheric cells (altitude bins in 1D case)
        `temperatures`   = temperature of each cell in K
        """
        self.name = name
        self.role = role
        self.wavelengths = wavelengths
        self.mass = mass
        self.cells = cells
        self.temperatures = temperatures
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
