import numpy as np

class Interpolator( object ):
    def __init__(self, x, y, **kwargs):
        self.x = x
        self.y = y
        self.bottom = np.min(x)
        self.top = np.max(x)
        self.bottom_value = y[np.where(x==np.min(x))]
        self.top_value = y[np.where(x==np.max(x))]
        # This assumes a direction...
        a = x[:-1]
        b = x[1:]
        ya = y[:-1]
        yb = y[1:]
        H = np.ones_like(ya)
        # scale_height = lambda a,b,ya,yb: (b-a) / np.log(ya/(yb+1e-30))
        # note that ya[i] = yb[i-1] for i in 0,1,...,N-1
        safe_alts = np.where(ya!=yb)[0]
        # H = np.where( safe_alts, scale_height(a,b,ya,yb) ,H )
        H[safe_alts] = (b[safe_alts] - a[safe_alts]) / np.log(ya[safe_alts] / (yb[safe_alts]+1e-30))
        self.cells = [ params for params in zip(a,b,ya,H)]
        self.top_scale = H[-2:-1]
        self.bot_scale = H[0:1]

    def __call__(self,x):
        if type(x) != np.ndarray:
            if type(x) == int:
                x = float(x)
            x = np.ndarray((1,1), buffer=np.asarray(x),dtype=float)
        interpolated = np.zeros_like(x)
        f_bot = self.bottom_value * np.exp(-(self.bottom-x)/self.bot_scale)
        f_top = self.top_value * np.exp(-(x-self.top)/self.top_scale)
        interpolated = np.where(x<=self.bottom, f_bot, interpolated)
        interpolated = np.where(x>=self.top,    f_top, interpolated)
        inside = np.where((x>self.bottom) * (x<self.top))
        # inside is tougher
        f_in = lambda a, ya, H: ya*np.exp(-(x-a)/H)
        for a,b,ya,H in self.cells:
            interpolated = np.where((x<b)*(x>a),f_in(a,ya,H),interpolated)
        return interpolated

class ScaleHeightInterpolate( object ):
    """
    Interpolate a function y(x) assuming a constant scale height between data points.
    
    Parameters
    ----------
    x : array
        Support of the function to be interpolated
    y : array
        Function values correspniding to `x`.

    Notes
    -----
    
    Let the initial data points be ``yi = y(xi)``. We then assume that,
    ``y(x>xi) = A exp(-(x-a)/H)``, where ``A = y(a)`` and ``a = max(xi,xi+1)``.
    Then, ``y(a) exp(-(b - a)/H) = y(b)``, so rearranging yields ``log( y(b) / y(a) )/(a-b) = 1/H`` 
    """
    def __init__(self,x,y,**kwargs):
        ind=y[:,0]>0
        self.inner = interp1d(x[ind],np.log(y[ind])
                        ,kind='linear'
                        ,axis=0
        )
        self.x = x[ind]
        self.y = y[ind]
        #print self.x[:-1].shape, self.x.shape
        #print self.y[:-1].shape, self.y.shape
        self.top = self.x.max()
        self.bot = self.x.min()
        self.ytop = self.y[np.argmax(self.x)]
        self.ybot = self.y[np.argmin(self.x)]
        Hs = np.abs(safe_divide((self.x[1:]-self.x[:-1]).reshape(-1,1),safe_log(self.y[1:]/self.y[:-1])))
        avgs = ( Hs[:5].mean(), Hs[-5:].mean() )
        if x[0]>x[-1]:
            self.Htop, self.Hbot = avgs
        else:
            self.Hbot, self.Htop = avgs
        self.Hbot *= -1

    def __call__(self,x):
        out = [_loginterp(xi
                ,self.inner
                ,self.bot, self.top
                ,self.ybot, self.ytop
                ,self.Hbot, self.Htop
        ) for xi in x]
        #print type(out)
        #print [np.shape(a) for a in out]
        return np.asarray(out)

class LogInterpolate( object ):
    """
    Interpolate a function y(x) assuming a constant scale height between data points.
    
    Parameters
    ----------
    x : array
        Support of the function to be interpolated
    y : array
        Function values corresponding to `x`.

    Notes
    -----
    
    Let the initial data points be ``yi = y(xi)``. We then assume that,
    ``y(x>xi) = A exp(-(x-a)/H)``, where ``A = y(a)`` and ``a = max(xi,xi+1)``.
    Then, ``y(a) exp(-(b - a)/H) = y(b)``, so rearranging yields ``log( y(b) / y(a) )/(a-b) = 1/H`` 
    """
    def __init__(self,x,y,**kwargs):
        ind=y[:,0]>0
        self.inner = [ interp1d(x[ind],np.log(y[ind,i])
                        ,kind='linear'
                        ,axis=0
                            ) for i in range(y.shape[1])]
        self.x = x[ind]
        self.y = y[ind]
        #print self.x[:-1].shape, self.x.shape
        #print self.y[:-1].shape, self.y.shape
        self.top = self.x.max()
        self.bot = self.x.min()
        self.ytop = self.y[np.argmax(self.x)]
        self.ybot = self.y[np.argmin(self.x)]
        Hs = np.abs(safe_divide((self.x[1:]-self.x[:-1]).reshape(-1,1),safe_log(self.y[1:]/self.y[:-1])))
        avgs = ( Hs[:5].mean(), Hs[-5:].mean() )
        if x[0]>x[-1]:
            self.Htop, self.Hbot = avgs
        else:
            self.Hbot, self.Htop = avgs
        self.Hbot *= -1

    def __call__(self,x,i):
        out = [_loginterp(xi
                         ,self.inner[i]
                         ,self.bot
                         ,self.top
                         ,self.ybot[i]
                         ,self.ytop[i]
                         ,self.Hbot
                         ,self.Htop
                     ) for xi in x]
        #print type(out)
        #print [np.shape(a) for a in out]
        return np.asarray(out)

class LogInterpolate1D( object ):
    """
    Interpolate a function y(x) assuming a constant scale height between data points.
    
    Parameters
    ----------
    x : array
        Support of the function to be interpolated
    y : array
        Function values corresponding to `x`.

    Notes
    -----
    
    Let the initial data points be ``yi = y(xi)``. We then assume that,
    ``y(x>xi) = A exp(-(x-a)/H)``, where ``A = y(a)`` and ``a = max(xi,xi+1)``.
    Then, ``y(a) exp(-(b - a)/H) = y(b)``, so rearranging yields ``log( y(b) / y(a) )/(a-b) = 1/H`` 
    """
    def __init__(self,x,y,**kwargs):
        ind=y>0
        self.inner = interp1d(x[ind], np.log(y[ind])
                              ,kind='linear'
                              ,axis=0
        )
        self.x = x[ind]
        self.y = y[ind]
        #print self.x[:-1].shape, self.x.shape
        #print self.y[:-1].shape, self.y.shape
        self.top = self.x.max()
        self.bot = self.x.min()
        self.ytop = self.y[np.argmax(self.x)]
        self.ybot = self.y[np.argmin(self.x)]
        Hs = np.abs(safe_divide((self.x[1:]-self.x[:-1]),safe_log(self.y[1:]/self.y[:-1])))
        avgs = ( Hs[:5].mean(), Hs[-5:].mean() )
        if x[0]>x[-1]:
            self.Htop, self.Hbot = avgs
        else:
            self.Hbot, self.Htop = avgs
        self.Hbot *= -1

    def __call__(self,x):
        out = _loginterp(x
                         ,self.inner
                         ,self.bot
                         ,self.top
                         ,self.ybot
                         ,self.ytop
                         ,self.Hbot
                         ,self.Htop
        )
        #print type(out)
        #print [np.shape(a) for a in out]
        return np.asarray(out)


@np.vectorize
def _loginterp(x,f,b,t,yb,yt,Hb,Ht):
    if x<b:
        interpolated = yb*np.exp(-(x-b)/Hb)
    elif x<t:
        interpolated = np.exp(f(x))
    else:
        interpolated = yt*np.exp(-(x-t)/Ht)
    return interpolated

@np.vectorize
def safe_divide(x,y):
    """
    Maybe divide x/y.
    """
    if y!=0:
        return x/y
    else:
        return np.nan

@np.vectorize
def safe_log(x):
    """
    Maybe Log(x).
    """
    if x>0:
        return np.log(x)
    elif x==0:
        return -np.inf
    else:
        return np.nan

if __name__ == "__main__":
    print(__name__)
    x = np.random.random(100)
    x.sort()
    y = np.random.random(100)
    f = interp1d(x,y)
    Y=_loginterp(x,f,x.min(),x.max(),y.min(),y.max(),1,1)
    print(Y)

