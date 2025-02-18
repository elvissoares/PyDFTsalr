import numpy as np
from .aux import w3FT, YKFT

# Author: Elvis do A. Soares
# Github: @elvissoares
# Date: 2023-10-24
# Updated: 2023-08-22

# the hard-sphere structure factor (WB version I)
def Chs(q,rhob,sigma=1.0):
    eta = np.pi*rhob*sigma**3/6
    chi0 = -(1+eta*(4+eta*(3-2*eta)))/(1-eta)**4
    chi1 = (2-eta+14*eta**2-6*eta**3)/(1-eta)**4+2*np.log(1-eta)/eta
    chi3 = -(3+5*eta*(eta-2)*(1-eta))/(1-eta)**4-3*np.log(1-eta)/eta
    qs = q*sigma
    return np.piecewise(qs,[qs<=1e-3,qs>1e-3],[np.pi*(eta-4)*sigma**3/(3*(1-eta)**4),lambda qs: 4*np.pi*sigma**3*(24*chi3-2*chi1*qs**2-(24*chi3-2*(chi1+6*chi3)*qs**2+(chi0+chi1+chi3)*qs**4)*np.cos(qs)+qs*(-24*chi3+(chi0+2*chi1+4*chi3)*qs**2)*np.sin(qs))/qs**6])

# the Yukawa potential (with core correction)
def Cyk(q,K,Z,sigma=1.0):
    cyk = np.zeros_like(q)
    for i in range(len(K)):
        cyk[:] += -K[i]*YKFT(q,Z[i],sigma) - K[i]*w3FT(q,sigma=2*sigma)
    return cyk

def CRPA(q,rho,kT,K,Z,sigma=1.0):
    beta = 1.0/kT
    return Chs(q,rho,sigma=sigma)+beta*Cyk(q,K,Z,sigma=sigma)

# The structure factor on RPA approximation
def SRPA(q,rho,kT,K,Z,sigma=1.0):
    return 1/(1-rho*CRPA(q,rho,kT,K,Z,sigma=sigma))

def DRPA(q,rho,kT,K,Z,sigma=1.0):
    return 1-rho*CRPA(q,rho,kT,K,Z,sigma=sigma)

def Shs(q,rho,sigma=1.0):
    return 1/(1-rho*Chs(q,rho,sigma=sigma))
