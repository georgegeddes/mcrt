from __future__ import division
from .interpolate import ScaleHeightInterpolate as Interpolator
from .jit import jit
from scipy.integrate import quad
import numpy as np

RE=6371

class Integrator( object ):
    """
    Integrate a function of 2 variables using Gaussian quadrature.
    """
    def __init__(self):
        pass
    
    def __call__(self,f,b,i=0,**kwargs):
        arr = np.fromiter((quad(f,0,b)[i] for b in b),dtype=float)
        return arr

class Ray( object ):
    """
    Representation of a ray in cartesian or polar coordinates.

    Parameters
    ----------
    eye : 2 element array / vector
        The location of the ray's endpoint, in cartesian coordinates.
    direction : 2 element array / vector
        A vector along the direction of the ray. If it is not a unit vector, it will be normalized.
    """

    def __init__(self,eye,direction):
        self.eye = eye.reshape(-1,1)
        self.direction = direction.reshape(-1,1)
        if (self.direction**2).sum() != 1:
            self.direction /= np.sqrt(np.sum(direction**2))

        # convenience functions
        self.xy = self.cartesian_points
        self.tr = self.polar_points

    def cartesian_points(self,t):
        return _cartesian_points(t)

    def polar_points(self,t):
        x,y = self.cartesian_points(t)
        return _polar(x,y)

    def r(self,t):
        x,y = self.xy(t)
        return _radius(x,y)

@jit
def _cartesian_points(t):
    try:
        T = t.reshape(1,-1)
    except:
        T = t
    finally:
        return self.eye + self.direction*T


@jit
def _radius(x,y):
    return np.sqrt(x*x + y*y)

@jit
def _polar(x,y):
    theta = np.arctan2(y,x)
    r = _radius(x,y)
    return theta, r


class Observer( object ):
    """
    An observer with some location, field of view, and look directions
    
    Parameters
    ----------
    eye : 2 element array / vector
        Observer location relative to Earth's center.
    FOV : scalar or array
        Field of view of the observer, either as a single value or as a vector.
    viewing_angles : array
        Zenith angles in radians for the look directions.
    """ 
    _interpolate = Interpolator#scipy.interpolate.interp1d
    _integrate = quad#Integrator()
    
    def __init__(self,eye,FOV,viewing_angles):
        self.eye = eye
        self.u = eye / (eye**2).sum()
        self.FOV = FOV
        self.va = viewing_angles
    
    def _new_sightline(self,angle):
        c, s = np.cos(angle), np.sin(angle)
        r = np.array([[c,-s],[s,c]])
        direction = r.dot(self.u)
        return Ray(self.eye,direction)
    
    def integrate_sightlines(self, z, emission, extinction):
        # distance along sightline in km
        t = np.linspace(0,2*RE,num=1e3)
        dt = t[1]-t[0]
        
        # set up interpolation
        interpolate = self._interpolate
        em = interpolate(z
                         ,emission*1e15
                         ,bounds_error=False
                         ,fill_value=0
                         ,axis=0
                         ,kind='linear'
                        )
        ex = interpolate(z
                         ,extinction*1e5
                         ,bounds_error=False
                         ,fill_value=0#extinction.max()*1e15
                         ,axis=0
                         ,kind='linear'
                        )
        source = lambda r: em(r) #/ ex(r)
        
        # # set up integration
        # def gen():
        #     for angle in self.va:
        #         sightline = self._new_sightline(angle)
        #         r = sightline.polar_points(t)[1].reshape(-1,1)
        #         dtau = ex(r)*dt
        #         depth = np.cumsum(dtau,axis=0)
        #         atten = np.exp(-depth)
        #         integrand = np.sum(source(r)*atten*dt,axis=1)
        #         yield np.sum(integrand.reshape(-1))
        # adaptive quadrature:
        def gen():
            for angle in self.va:
                sightline = self._new_sightline(angle)
                r = lambda t: sightline.polar_points(t)[1].reshape(-1,1)
                #absorb = lambda t: ex(r(t))
                depth = lambda t: self._integrate(ex(r(t)),0,t)[0]
                atten = lambda t: np.exp(-depth(t))
                integrand = lambda t: source(r(t))*atten(t)
                yield self._integrate(integrand,0,np.inf)[0]
    
        # perform integration and return result
        weight = self.FOV/(4*np.pi)
        los_intensity = np.fromiter(gen(),dtype=np.float)*weight
        return los_intensity

    @classmethod
    def from_height(cls,height,FOV,viewing_angles):
        eye = np.array([0,RE+height])
        return cls(eye,FOV,viewing_angles)

def integrate_spectrum_along_z(func,a,b,err=False):
    """
    Take a function that takes a location variable, r, and returns a 1xN array
    """
    i = int(err)
