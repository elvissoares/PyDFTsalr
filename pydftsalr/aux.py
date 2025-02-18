from numpy import pi, sinc, exp, cos, sin, piecewise
from scipy.special import spherical_jn

def sigmaLancsozFT(kx,ky,kz,kcut):
    return sinc(kx/kcut[0])*sinc(ky/kcut[1])*sinc(kz/kcut[2])

def translationFT(kx,ky,kz,a):
    return exp(1.0j*(kx*a[0]+ky*a[1]+kz*a[2]))

def w3FT(k,sigma=1.0):
    return (pi*sigma**3/6)*(spherical_jn(2,0.5*sigma*k)+spherical_jn(0,0.5*sigma*k))

def w2FT(k,sigma=1.0):
    return pi*sigma**2*spherical_jn(0,0.5*sigma*k)

def wtensFT(k,sigma=1.0):
    return pi*sigma**2*spherical_jn(2,0.5*sigma*k)

def wtensoverk2FT(k,sigma=1.0):
    return piecewise(k,[k*sigma<=1e-8,k*sigma>1e-8],[pi*sigma**4/60,lambda k:(pi*sigma**2/k**2)*spherical_jn(2,0.5*sigma*k)])

# The Fourier transform of the Yukawa potential (without core)
def YKFT(k,Z,sigma=1.0):
    return 4*pi*sigma**3*piecewise(k,[k<=1e-6,k>1e-6],[(1+Z)/Z**2,lambda k: (k*sigma*cos(k*sigma)+Z*sin(k*sigma))/(k*sigma*(Z**2+(k*sigma)**2))])