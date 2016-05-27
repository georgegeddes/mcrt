import numpy as np
import scipy.special

dt = np.float

# Boltzmann's constant
kB = 1.3806488e-23 # J / K

# speed of light
c = 2.99792458e8 # m / s

def freq_param( num_freqs=24 ):
    """
    Return a set of dimensionless chunks with equal gaussian weight.

    Parameters
    ----------
    num_freqs : int, optional
        Number of chunks to return (default is 24).

    Returns
    -------
    numpy.ndarray
        Complete set of frequency intervals.
    
    Notes
    -----
    Frequency parameter x = ( nu - nu_0 ) / nu_0.
    """
    # the 1e-6 padding is to prevent division errors in the extinction coefficients
    return np.array( [ 2.25 * (i + 1e-6) / num_freqs for i in xrange(num_freqs-1) ] + [ np.inf ], dtype=dt ).reshape( 1, num_freqs )

def frequency( wavelength ):
    """
    Converts wavelength to frequency.
    
    Parameters
    ----------
    wavelength : float

    Returns
    -------
    float
        Frequency.
    """
    return c / wavelength

def sdu( m, T ):
    """Standard doppler unit.
    
    The standard deviation for a doppler-broadened line centered at 'nu' is `sdu( m, T ) * nu`.
    
    Parameters
    ----------
    m : float
        Mass in kg of a gas molecule.
    T : float
        Temperature in K.

    Returns
    -------
    float
        One standard Doppler unit.
    """
    return np.sqrt( kB * T / m / c**2 )

class lineshape( object ):
    """
    Spectral lineshape function.
    
    Parameters
    ----------
    mean : float
        Line center frequency.
    std : float
        Standard deviation.
    """
    def __init__( self, mean, std ):
        self.mean = mean
        self.std  = std

    @property
    def cdf( self ):
        """Returns an array of the cdf"""
        return scipy.special.erf( freq_param() / self.std / np.sqrt(2) )

class Gas( object ):
    """
    Contains physical properties pertaining to a single gas species.    

    Parameters
    ----------
    name : string 
        Name of gas species
    role : {'absorption','scattering'}
        Type of interaction with radiation.
    wavelengths : list or tuple of floats 
        Wavelengths corresponding to `cross_sections`.
    cross_sections : list or tuple of floats
        Reference cross sections at 1000 K.
    mass : float
        Molecular mass in kg.
    cells : array_like
        List of atmospheric cells (altitude bins in 1D case)
    temperatures : array_like
        Temperature of each cell in K

    Attributes
    ----------
    species : dictionary
        Dictionary of all Gas instances by species name.
    absorbers : dictionary
        Dictionary of all Gas instances with role=='absorption'.
    scatterers: dictionary
        Dictionary of all Gas instances with role=='scattering'.
    """
    species = {}
    absorbers = {}
    scatterers = {}
    def __init__(self, name, role, wavelengths, cross_sections, mass, cells=(), temperatures=() ):
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
