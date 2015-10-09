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