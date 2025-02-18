import numpy as np
from numpy import pi, log, meshgrid
import timeit
from .aux import sigmaLancsozFT, w3FT, w2FT, YKFT
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Author: Elvis do A. Soares
# Github: @elvissoares
# Date: 2023-10-01
# Updated: 2025-02-18

" The DFT model for SALR fluid on 3d geometries"

class dft3d():
    def __init__(self,gridsize):
        self.Ncell = gridsize 
        self.Ncelltot = self.Ncell[0]*self.Ncell[1]*self.Ncell[2]
        self.D = 1.0
        self.rho = torch.zeros((self.Ncell[0],self.Ncell[1],self.Ncell[2]),dtype=torch.float32, device=device)

    def Set_FluidProperties(self,K,Z,sigma=1.0):
        self.sigma = sigma
        self.Kyk = K
        self.Zyk = Z

    def Set_Diffusion(self,D=1.0):
        self.D = D
    
    def Set_Geometry(self,Lgrid):
        self.Lgrid = Lgrid

        self.delta = Lgrid/self.Ncell
        self.x = np.arange(0.0,self.Lgrid[0],self.delta[0])
        self.y = np.arange(0.0,self.Lgrid[1],self.delta[1])
        self.z = np.arange(0.0,self.Lgrid[2],self.delta[2])
        self.X,self.Y,self.Z = meshgrid(self.x,self.y,self.z,indexing ='ij')
        self.dV = self.delta[0]*self.delta[1]*self.delta[2]
        self.Vol = self.Lgrid[0]*self.Lgrid[1]*self.Lgrid[2]

        self.Ngrid = np.array([self.Ncell[0],self.Ncell[1],self.Ncell[2]])
        self.Ngridtot = self.Ngrid[0]*self.Ngrid[1]*self.Ngrid[2]

        kx = torch.fft.fftfreq(self.Ngrid[0], d=self.delta[0])*2*pi
        ky = torch.fft.fftfreq(self.Ngrid[1], d=self.delta[1])*2*pi
        kz = torch.fft.fftfreq(self.Ngrid[2], d=self.delta[2])*2*pi
        self.kcut = torch.tensor([kx.max(),ky.max(),kz.max()])

        self.Kx,self.Ky,self.Kz = torch.meshgrid(kx,ky,kz,indexing ='ij')
        self.K = torch.stack((self.Kx,self.Ky,self.Kz)).to(device)
        self.Knorm = torch.sqrt(self.Kx**2 + self.Ky**2 + self.Kz**2).cpu().numpy()
        del kx, ky, kz

        # creating arrays
        self.rho_hat = torch.zeros((self.Ngrid[0],self.Ngrid[1],self.Ngrid[2]),dtype=torch.complex64, device=device)

        self.Vext = torch.zeros((self.Ngrid[0],self.Ngrid[1],self.Ngrid[2]),dtype=torch.float32, device=device)
        
        self.c1 = torch.zeros((self.Ngrid[0],self.Ngrid[1],self.Ngrid[2]),dtype=torch.float32, device=device)
        self.c1_hat = torch.empty_like(self.rho_hat, device=device)
        self.c1hs = torch.zeros_like(self.c1, device=device)
        self.c1att = torch.zeros_like(self.c1, device=device)

        self.n0 = torch.empty_like(self.rho, device=device)
        self.n1 = torch.empty_like(self.rho, device=device)
        self.n3 = torch.empty_like(self.rho, device=device)
        self.n2 = torch.empty_like(self.rho, device=device)
        self.n2vec = torch.empty((3,self.Ngrid[0],self.Ngrid[1],self.Ngrid[2]),dtype=torch.float32, device=device)
        self.n1vec = torch.empty_like(self.n2vec, dtype=torch.float32, device=device)

        # Defining the weight functions
        self.sigmaLancsoz = torch.tensor(sigmaLancsozFT(self.Kx.cpu().numpy(),self.Ky.cpu().numpy(),self.Kz.cpu().numpy(),self.kcut.cpu().numpy()),dtype=torch.complex64, device=device)
        self.w3_hat = torch.tensor(w3FT(self.Knorm,sigma=self.sigma),dtype=torch.complex64, device=device)*self.sigmaLancsoz
        self.w2_hat = torch.tensor(w2FT(self.Knorm,sigma=self.sigma),dtype=torch.complex64, device=device)*self.sigmaLancsoz
        self.w2vec_hat = self.K*(-1.0j*self.w3_hat)

        self.uint = torch.zeros((self.Ngrid[0],self.Ngrid[1],self.Ngrid[2]),dtype=torch.float32, device=device)
        self.utyk_hat = torch.zeros((self.Ngrid[0],self.Ngrid[1],self.Ngrid[2]),dtype=torch.complex64, device=device) 
        for i in range(len(self.Kyk)):
            self.utyk_hat[:] += torch.tensor(self.Kyk[i]*YKFT(self.Knorm,self.Zyk[i],self.sigma),dtype=torch.complex64, device=device)*self.sigmaLancsoz # to avoid Gibbs phenomenum
            self.utyk_hat[:] += torch.tensor(self.Kyk[i]*w3FT(self.Knorm,sigma=2*self.sigma),dtype=torch.complex64, device=device)*self.sigmaLancsoz # to avoid Gibbs phenomenum
        self.amft = self.utyk_hat[0,0,0].real

        del self.sigmaLancsoz      

    def Set_Temperature(self,kT):
        self.kT = kT
        self.beta = 1/self.kT

    def Set_BulkDensity(self,rhob):
        self.rhob = rhob             
        self.Calculate_mu()
        
    def Set_External_Potential(self,Vext):

        self.Vext[:] = torch.tensor(Vext,dtype=torch.float32, device=device)
        self.mask = (self.Vext<16128)
        self.Vext[self.Vext>=16128] = 16128

    def Set_InitialCondition(self):

        self.rho[:] = 0.0
        self.rho[self.mask] = self.rhob
        # self.rho[self.mask] = self.rhob*torch.exp(-0.01*self.beta*self.Vext[self.mask])

        self.Update_System()

    def GetSystemInformation(self):
        print('============== The DFT 3D for Multiple Yukawa fluids ==============')
        print('The grid is',self.Ngrid)
        print('--- Geometry properties ---')
        print('Lx =', self.Lgrid[0], ' A')
        print('Ly =', self.Lgrid[1], ' A')
        print('Lz =', self.Lgrid[2], ' A')
        print('delta = ', self.delta, ' A')
        print('Vol =',self.Vol, ' A³')
    
    def GetFluidInformation(self):
        print('--- Fluid properties ---')
        print('K/kB =', self.Kyk, ' K')
        print('Z/sigma =', self.Zyk)
        print('sigma =', self.sigma, ' A')

    def GetFluidTemperatureInformation(self):
        print('Temperature =', self.kT, ' K')

    def GetFluidDensityInformation(self):
        print('Bulk Density:',self.rhob, ' particles/A³')
        print('muid:',self.muid.round(3))
        print('muhs:',self.muhs.round(3))
        print('muatt:',self.muatt.round(3))

    def Update_System(self):
        self.Calculate_FT()
        self.Calculate_weighted_densities()
        self.Calculate_c1()
        self.Calculate_Omega()

    def Calculate_FT(self):
        self.rho_hat[:] = torch.fft.fftn(self.rho)

    def Calculate_weighted_densities(self):

        # Unpack the results and assign to self.n 
        self.n3[:] = torch.fft.ifftn(self.rho_hat*self.w3_hat).real
        self.n2[:] = torch.fft.ifftn(self.rho_hat*self.w2_hat).real
        self.n2vec[:] = torch.fft.ifftn(self.rho_hat*self.w2vec_hat,dim=(1,2,3)).real

        self.n3[self.n3>=1.0] = 1.0-1e-30 # to avoid Nan on some calculations
        self.xi = (self.n2vec*self.n2vec).sum(dim=0)/((self.n2+1e-30)**2)
        self.xi[self.xi>=1.0] = 1.0-1e-30

        self.n1vec[:] = self.n2vec/(2*pi*self.sigma)

        self.n0[:] = self.n2/(pi*self.sigma**2)
        self.n1[:] = self.n2/(2*pi*self.sigma)
        
        self.phi1 = -torch.log(1-self.n3)
        self.dphi1dn3 = 1.0/(1.0-self.n3)
        self.phi2 = 1.0/(1.0-self.n3)
        self.dphi2dn3 = 1.0/(1.0-self.n3)**2
        self.phi3 = torch.where(self.n3 <=1e-3, (1.5+8*self.n3/3+15*self.n3**2/4)/(36*pi), (self.n3+(1-self.n3)**2*torch.log(1-self.n3))/(36*pi*self.n3**2*(1-self.n3)**2))
        self.dphi3dn3 = torch.where(self.n3 <=1e-3, (8/3+7.5*self.n3+72*self.n3**2/5)/(36*pi), -(self.n3*(2-5*self.n3+self.n3**2)+2*(1-self.n3)**3*torch.log(1-self.n3))/(36*pi*self.n3**3*(1-self.n3)**3))
        
        self.uint[:] = torch.fft.ifftn(self.rho_hat*self.utyk_hat).real

    def Calculate_Free_energy(self):

        self.fid = self.kT*self.rho*(torch.log(self.rho+1.0e-16)-1.0)
        self.Fid = torch.sum(self.fid)*self.dV

        self.fhs = self.kT*(self.n0*self.phi1+self.phi2*(self.n1*self.n2-(self.n1vec*self.n2vec).sum(dim=0)) + self.phi3*self.n2**3*(1-self.xi)**3)
        self.Fhs = torch.sum(self.fhs)*self.dV

        self.ftyk = 0.5*self.rho*self.uint
        self.Ftyk = torch.sum(self.ftyk)*self.dV

        self.f = self.fid + self.fhs + self.ftyk

        self.Fexc =  self.Fhs + self.Ftyk
        self.F = self.Fid + self.Fexc

    def Calculate_Omega(self):
        self.Calculate_Free_energy()
        self.Omega = self.F + torch.sum((self.Vext-self.mu)*self.rho)*self.dV

    def Calculate_c1(self):

        self.c1_hat[:] = -torch.fft.fftn(self.phi1)/(pi*self.sigma**2)*self.w2_hat #dPhidn0
        self.c1_hat[:] += -torch.fft.fftn((self.n2*self.phi2) )/(2*pi*self.sigma)*self.w2_hat #dPhidn1
        self.c1_hat[:] += -torch.fft.fftn((self.n1*self.phi2 + 3*(self.n2**2)*(1+self.xi)*((1-self.xi)**2)*self.phi3))*self.w2_hat #dPhidn2
        self.c1_hat[:] += -torch.fft.fftn((self.n0*self.dphi1dn3 +(self.n1*self.n2-(self.n1vec*self.n2vec).sum(dim=0))*self.dphi2dn3 + (self.n2**3*(1-self.xi)**3)*self.dphi3dn3) )*self.w3_hat #dPhidn3
        self.c1_hat[:] += (torch.fft.fftn( (-self.n2vec*self.phi2), dim=(1,2,3))/(2*pi*self.sigma)*self.w2vec_hat).sum(dim=0) #dPhidn1vec
        self.c1_hat[:] += (torch.fft.fftn( (-self.n1vec*self.phi2 + (- 6*self.n2*self.n2vec*(1-self.xi)**2)*self.phi3), dim=(1,2,3))*self.w2vec_hat).sum(dim=0)#dPhidn2vec

        self.c1hs[:] = torch.fft.ifftn(self.c1_hat).real

        self.c1att[:] = -self.beta*self.uint

        self.c1_hat[:] += torch.fft.fftn(self.c1att)

        self.c1[:] = self.c1hs + self.c1att

    def Calculate_mu(self):
        self.muid = self.kT*log(self.rhob)

        n3 = self.rhob*pi*self.sigma**3/6
        n2 = self.rhob*pi*self.sigma**2
        n1 = self.rhob*self.sigma/2
        n0 = self.rhob

        phi1 = -log(1-n3)
        dphi1dn3 = 1.0/(1.0-n3)
        phi2 = 1.0/(1.0-n3)
        dphi2dn3 = 1.0/(1.0-n3)**2
        phi3 = (n3+(1-n3)**2*log(1-n3))/(36*pi*n3**2*(1-n3)**2)
        dphi3dn3 = -(n3*(2-5*n3+n3**2)+2*(1-n3)**3*log(1-n3))/(36*pi*n3**3*(1-n3)**3)

        dPhidn0 = phi1
        dPhidn1 = n2*phi2
        dPhidn2 = n1*phi2 + (3*n2**2)*phi3
        dPhidn3 = n0*dphi1dn3 +(n1*n2)*dphi2dn3 + (n2**3)*dphi3dn3
        self.muhs = self.kT*(dPhidn0+dPhidn1*self.sigma/2+dPhidn2*pi*self.sigma**2+dPhidn3*pi*self.sigma**3/6)

        self.muatt = self.rhob*self.amft

        self.muexc = self.muhs + self.muatt
        self.mu = self.muid + self.muexc

    # Calculate Equilibrium using the optmization algorithms
    def Calculate_Equilibrium(self,alpha0=0.25,dt=0.1,atol=1e-6,rtol=1e-4,max_iter=9999,method='abc-fire',logoutput=False):

        starttime = timeit.default_timer()

        alpha = alpha0

        lnrho = torch.zeros((self.Ngrid[0],self.Ngrid[1],self.Ngrid[2]),dtype=torch.float32, device=device)
        lnrho[:] = torch.log(self.rho+1.0e-30) # to avoid log(0)
        self.Update_System()

        F = torch.zeros((self.Ngrid[0],self.Ngrid[1],self.Ngrid[2]),dtype=torch.float32, device=device)
        sk = torch.empty_like(F)
        F[:] = -(lnrho - self.c1 - self.beta*self.mu + self.beta*self.Vext)
        sk[:] = atol+rtol*torch.abs(self.rho)
        error = torch.norm(self.rho*F/sk)/np.sqrt(self.Ngridtot)

        if logoutput: 
            print('Iter.','Omega','error','|','alpha','dt')
            print(0,self.Omega.cpu().numpy(),error.cpu().numpy(),'|',alpha,dt)

        if method == 'picard':
            # Picard algorithm
            
            for i in range(max_iter):                
                lnrho[:] += alpha*F
                self.rho[:] = torch.exp(lnrho).cpu()
                self.Update_System()
                F[:] = -(lnrho - self.c1 - self.beta*self.mu + self.beta*self.Vext)
                
                self.Niter = i+1
                sk[:]=atol+rtol*torch.abs(self.rho)
                error = torch.norm(self.rho*F/sk)/np.sqrt(self.Ngridtot)
                if logoutput: print(self.Niter,self.Omega.cpu().numpy(),error.cpu().numpy())
                if error < 1.0: break

        elif method == 'abc-fire':
            # ABC-Fire algorithm https://doi.org/10.1016/j.commatsci.2022.111978
            Ndelay = 20
            Nnegmax = 2000
            dtmax = 10*dt
            dtmin = 0.02*dt
            Npos = 1
            Nneg = 0
            finc = 1.1
            fdec = 0.5
            fa = 0.99
            V = torch.zeros((self.Ngrid[0],self.Ngrid[1],self.Ngrid[2]),dtype=torch.float32, device=device)
            i = 0 

            while i < max_iter:

                P = torch.sum(F*V) # dissipated power
                if (P>0):
                    Npos = Npos + 1
                    if Npos>Ndelay:
                        dt = min(dt*finc,dtmax)
                        alpha = max(1.0e-10,alpha*fa)
                else:
                    Npos = 1
                    Nneg = Nneg + 1
                    if Nneg > Nnegmax: break
                    if i> Ndelay:
                        dt = max(dt*fdec,dtmin)
                        alpha = alpha0
                    lnrho[:] -= V*0.5*dt
                    V[:] = 0.0
                    self.rho[:] = torch.exp(lnrho).cpu()
                    self.Update_System()

                V[:] += F*0.5*dt
                V[:] = (1/(1-(1-alpha)**Npos))*((1-alpha)*V + alpha*F*torch.norm(V)/torch.norm(F))
                lnrho[:] += dt*V
                self.rho[:] = torch.exp(lnrho).cpu()
                self.Update_System()
                F[:] = -(lnrho - self.c1 - self.beta*self.mu + self.beta*self.Vext)
                V[:] += F*0.5*dt

                self.Niter = i+1
                i += 1
                sk[:]=atol+rtol*torch.abs(self.rho)
                error = torch.norm(self.rho*F/sk)/np.sqrt(self.Ngridtot)

                if logoutput: print(self.Niter,self.Omega.cpu().numpy(),error.cpu().numpy(),'|',alpha,dt)
                if error < 1.0 and self.Niter> Ndelay: break
                if torch.isnan(error):
                    print('DFT::ABC-FIRE: The system is out fo equilibrium!')
                    break
                
            del V

        del F  
        torch.cuda.empty_cache()

        self.Nabs = self.rho.sum()*self.dV
        
        if logoutput:
            print("Time to achieve equilibrium:", timeit.default_timer() - starttime, 'sec')
            print('Number of iterations:', self.Niter)
            print('error:', error.cpu().numpy())
            print('---- Equilibrium quantities ----')
            print('Fid/Vol =',self.Fid.cpu().numpy()/self.Vol)
            print('Fexc/Vol =',self.Fexc.cpu().numpy()/self.Vol)
            print('beta*F/Vol =',self.beta*(self.F/self.Vol).cpu().numpy())
            print('Omega/Vol =',self.Omega.cpu().numpy()/self.Vol)
            print('mu =',self.mu)
            print('Nbulk =',self.rhob*self.Vol)
            print('Nabs =',self.Nabs.cpu().numpy())
            print('rhomean =',self.rho.mean().cpu().numpy())
            print('================================')

    # Calculate Equilibrium using the optmization algorithms
    def Calculate_Canonical_Equilibrium(self,alpha0=0.25,dt=0.1,atol=1e-6,rtol=1e-4,max_iter=9999,method='abc-fire',logoutput=False):

        starttime = timeit.default_timer()

        alpha = alpha0

        Nabs = self.rho.sum()*self.dV
        Nbulk = self.rhob*self.Vol

        lnrho = torch.zeros((self.Ngrid[0],self.Ngrid[1],self.Ngrid[2]),dtype=torch.float32, device=device)
        lnrho[:] = torch.log(self.rho+1.0e-30) # to avoid log(0)
        self.Update_System()
        self.mu = torch.sum(self.kT*(lnrho-self.c1)*self.rho)*self.dV/Nbulk

        F = torch.zeros((self.Ngrid[0],self.Ngrid[1],self.Ngrid[2]),dtype=torch.float32, device=device)
        sk = torch.empty_like(F)
        F[:] = -(lnrho - self.c1 - self.beta*self.mu + self.beta*self.Vext)
        sk[:] = atol+rtol*torch.abs(self.rho)
        error = torch.norm(self.rho*F/sk)/np.sqrt(self.Ngridtot)

        if logoutput: print(0,self.F.cpu().numpy(),Nabs.cpu().numpy()/self.Vol,self.mu.cpu().numpy(),'|',error.cpu().numpy(),dt)

        if method == 'abc-fire':
            # ABC-Fire algorithm https://doi.org/10.1016/j.commatsci.2022.111978
            Ndelay = 20
            Nnegmax = 2000
            dtmax = 10*dt
            dtmin = 0.02*dt
            Npos = 1
            Nneg = 0
            finc = 1.1
            fdec = 0.5
            fa = 0.99
            V = torch.zeros((self.Ngrid[0],self.Ngrid[1],self.Ngrid[2]),dtype=torch.float32, device=device)
            i = 0 

            while i < max_iter:

                Nabs = self.rho.sum()*self.dV

                P = torch.sum(F*V) # dissipated power
                if (P>0):
                    Npos = Npos + 1
                    if Npos>Ndelay:
                        dt = min(dt*finc,dtmax)
                        alpha = max(1.0e-10,alpha*fa)
                else:
                    Npos = 1
                    Nneg = Nneg + 1
                    if Nneg > Nnegmax: break
                    if i> Ndelay:
                        dt = max(dt*fdec,dtmin)
                        alpha = alpha0
                    lnrho[:] -= V*0.5*dt
                    V[:] = 0.0
                    self.rho[:] = torch.exp(lnrho).cpu()
                    self.Update_System()

                V[:] += F*0.5*dt
                V[:] = (1/(1-(1-alpha)**Npos))*((1-alpha)*V + alpha*F*torch.norm(V)/torch.norm(F))
                lnrho[:] += dt*V
                self.rho[:] = torch.exp(lnrho).cpu()
                self.Update_System()
                self.mu =  torch.sum(self.kT*(lnrho-self.c1)*self.rho)*self.dV/Nbulk
                F[:] = -(lnrho - self.c1 - self.beta*self.mu + self.beta*self.Vext)
                V[:] += F*0.5*dt

                self.Niter = i+1
                i += 1
                sk[:]=atol+rtol*torch.abs(self.rho)
                error = torch.norm(self.rho*F/sk)/np.sqrt(self.Ngridtot)
                
                if logoutput: print(self.Niter,self.F.cpu().numpy(),Nabs.cpu().numpy()/self.Vol,self.mu.cpu().numpy(),'|',error.cpu().numpy(),alpha,dt)
                if error < 1.0 and self.Niter> Ndelay: break
                if torch.isnan(error):
                    print('DFT::ABC-FIRE: The system is out fo equilibrium!')
                    break
                
            del V

        del F  
        torch.cuda.empty_cache()

        if logoutput:
            print("Time to achieve equilibrium:", timeit.default_timer() - starttime, 'sec')
            print('Number of iterations:', self.Niter)
            print('error:', error.cpu().numpy())
            print('---- Equilibrium quantities ----')
            print('Fid =',self.Fid.cpu().numpy())
            print('Fexc =',self.Fexc.cpu().numpy())
            print('Omega =',self.Omega.cpu().numpy())
            # print('Nbulk =',self.Nbulk)
            print('Nabs =',Nabs.cpu().numpy())
            print('================================')

    def Set_Dynamics(self,dt=1e-4):
        self.h = dt*self.sigma**2/self.D # in scales of diffusion
        # The linear terms of PDE
        Loperator_k = torch.tensor(-self.D*self.Knorm**2,dtype=torch.float32,device=device)
        self.Tlinear_k = torch.exp(self.h*Loperator_k) 
        # Dealising matrix
        dealias = ((torch.abs(self.K[0]) < self.kcut[0]*2.0/3.0 )*(torch.abs(self.K[1]) < self.kcut[1]*2.0/3.0 )*(torch.abs(self.K[2]) < self.kcut[2]*2.0/3.0 )).to(device)
        # Defining the time marching operators arrays
        self.Tnon_k = dealias*self.h*torch.where(self.Tlinear_k == 1.0,1.0,(self.Tlinear_k -1.0)/torch.log(self.Tlinear_k ))

        self.Noperator_k = torch.zeros_like(self.rho_hat,device=device)

        self.Calculate_FT()

    # Compute a new time step using the exponential integrators in pseudo spectral methods
    def Calculate_TimeStep(self,ti,tf):

        Nsteps = int((tf-ti)/self.h)
        
        self.Calculate_FT()

        for i in range(Nsteps):
            self.Calculate_weighted_densities()
            self.Calculate_c1()
            
            # calculate the nonlinear operator (with dealising)
            self.Noperator_k[:] = -1.0j*self.D*self.K[0]*torch.fft.fftn(self.rho*torch.fft.ifftn(1.0j*self.K[0]*self.c1_hat).real)
            self.Noperator_k[:] += -1.0j*self.D*self.K[1]*torch.fft.fftn(self.rho*torch.fft.ifftn(1.0j*self.K[1]*self.c1_hat).real)
            self.Noperator_k[:] += -1.0j*self.D*self.K[2]*torch.fft.fftn(self.rho*torch.fft.ifftn(1.0j*self.K[2]*self.c1_hat).real)

            # updating in time
            self.rho_hat[:] = self.rho_hat*self.Tlinear_k + self.Noperator_k*self.Tnon_k 

            # IFT to next step
            self.rho[:] = torch.fft.ifftn(self.rho_hat).real 
            self.rho[self.rho<1.0e-30] = 1.0e-30 # to avoid zeros

        # Update Free-energy and so on
        self.Update_System()


