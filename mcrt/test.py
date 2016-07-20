from .mcrt import Gas, Atmosphere
from .data.testdata import *
#import pyauric
import numpy as np
import os

data_folder = os.path.join(os.path.split(os.path.abspath(__file__))[0],'data')

def TestAtmosphere(N_angles = 2):
    """Return a realistic atmosphere for testing purposes"""
    #setup cross sections
    mag = 1e-18
    f = lambda L: [ x * mag for x in L ]
    csO  = f( [  3.89,  3.89,  3.90 ] )
    csO2 = f( [ 33.80, 14.00, 10.40 ] )
    csN2 = f( [ 0.049,  0.29, 10.10 ] )

    amu = 1.660538921e-27 # kg
    mO = 16 * amu
    mO2 = 32 * amu
    mN2 = 28 * amu

    for name, cs, mass in [ ("O", csO, mO ), ("N2", csN2, mN2 ), ("O2", csO2, mO2 ) ]:
        Gas( name, 'absorption', [832, 833, 834], cs, mass )

    wavelengths = [ 832.757e-8, 833.329e-8, 834.466e-8 ]
    # mO16 = 1.489917e10 # eV / c^2
    # #e^2 = hbar c alpha
    # hbarc = 1.97327e-5 # eV cm
    # alpha = 1.0/127.0
    # me = 5.109989e5    # eV / c^2
    # c = 2.998e10       # cm / s
    # kb = 8.6173e-5     # eV / K
    # T = 1000           # K

    # sig0_over_f = lambda x : x * np.sqrt( mO16 * hbarc * alpha / ( 2 * kb * T * c * me ) )

    # fosc = [ 0.0459, 0.0916, 0.1371 ]

    sigma_scatter = [1.68E-13, 1.12E-13, 5.61E-14] # from Vickers' thesis, Table 4.2, p. 101
    # sigma_scatter = [ sig0_over_f(w)*fosc[i] for (i, w) in enumerate(wavelengths) ]

    Gas( "O+", 'scattering', [832, 833, 834], sigma_scatter, mO )

    # setup and read AURIC stuff
    #auric = pyauric.AURICManager(data_folder)

    n_species = [ "[O]", "[O2]", "[N2]" ]

    #neutrals = auric.retrieve("atmos.dat", n_species + ['Tn (K)'])
    neutrals = {'[O]':np.asarray(nO)
                , '[O2]':np.asarray(nO2)
                , '[N2]':np.asarray(nN2)
                , 'Tn (K)':np.asarray(T)
                , 'ALT':np.asarray(altitudes_km)}
    temperatures = neutrals['Tn (K)'] 
    temperatures = temperatures * 1000.0 / np.max(temperatures)
    absorbers = { k[1:-1]:v for (k,v) in neutrals.items() if '[' in k }
    altitudes = neutrals["ALT"]#np.logspace(0,3,len(neutrals["ALT"]))[::-1]

    # O+ setup
    #oplus = auric.retrieve("ionos.dat", ['[e-]'])
    oplus = {'[e-]': np.asarray(ne)} # close enough to start

    # two streams:
    angles = np.linspace(0,np.pi,num=N_angles, endpoint=True)

    viewing_angles = np.arange(90-45, 90+45)

    atmosphere = Atmosphere( altitudes, angles, temperatures, absorbers, oplus['[e-]'], viewing_angles )

    return atmosphere

def test():
    atm = TestAtmosphere()
    M = atm.multiple_scatter_matrix(2)
    import matplotlib.pyplot as plt
    # plt.imshow(M, cmap=plt.cm.viridis)
    # plt.show()
    Q = (M[:100,:100]+M[100:,:100] + M[:100,100:] + M[100:,100:])/2
    # plt.imshow(Q)
    # plt.show()
    src = np.asarray(oii834A_photoion) + np.asarray(oii834A_e_impact)
    fin = src.dot(Q)#Q.dot(src) # this is the correct order
    # fin2= Q.dot(src)
    aur = oii834A_final
    plt.figure()
    plt.plot(altitudes_km,src, label='Source')
    plt.plot(altitudes_km,fin, label='Markov', ls='', marker = 'x')
    # plt.plot(altitudes_km,fin2, label='Markov2', ls='', marker = 'x')
    plt.plot(altitudes_km,aur, label='Featurier')
    plt.legend()
    plt.yscale('log')
    plt.ylim(1e-3,)
    plt.show()
    return Q
    

if __name__=="__main__":
    print(TestAtmosphere())
