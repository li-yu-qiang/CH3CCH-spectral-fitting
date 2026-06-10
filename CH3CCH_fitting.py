import numpy as np
from astropy import constants
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

c=constants.c.cgs.value # Speed of light (cm/s)
kB=constants.k_B.cgs.value # Boltzmann coefficient (erg/K)
h=constants.h.cgs.value # Planck constant (erg*s)

colspecs=[(0,13),(24,35),(37,47),(48,51),(61,63),(63,65)]
spec_df=pd.read_fwf('c040502.cat',header=None,colspecs=colspecs)
spec_df=spec_df.replace({'A':'10','B':'11'},regex=True)
spec_df[4]=spec_df[4].astype('Int64')
spec_df=spec_df.to_numpy()

freq_all=spec_df[:,0]
A_all=10**spec_df[:,1]
El_all=spec_df[:,2]*h*c/kB
g_upper_all=spec_df[:,3]
J_upper_all=spec_df[:,4]
K_upper_all=spec_df[:,5]
Eu_all=El_all+h*freq_all*10**6/kB

gI=np.zeros((K_upper_all.shape[0],1))

for i in range (K_upper_all.shape[0]):
    if (K_upper_all[i]%3==0)&(K_upper_all[i]!=0):
        gI[i,0]=2
    else:
        gI[i,0]=1

def partition(T):
    q=2*(2*(J_upper_all[:]-1)+1)*gI[:,0]*np.e**(-El_all[:]/T)
    Q=np.sum(q)
    return Q

def para(J,K0,K1):
    index=np.where(J_upper_all==J)
    arr=spec_df[index[0],:]
    index=np.where((arr[:,-1]>=K0)&(arr[:,-1]<=K1))
    arr=arr[index[0],:]
    arr=arr[::-1]
    return arr

El_all=spec_df[:,2]*h*c/kB
g_upper_all=spec_df[:,3]
J_upper_all=spec_df[:,4]
K_upper_all=spec_df[:,5]
Eu_all=El_all+h*freq_all*10**6/kB

def multi_gaussian(vel,T_rot,logN_tot,sigma_v,v_0,p):
    N_tot=10**logN_tot
    nu_0s=p[:,0]*1e6
    A_us=10**p[:,1]
    E_us=p[:,2]*h*c/kB+h*nu_0s/kB
    g_us=p[:,3]
    phi=1/np.sqrt(2*np.pi)/(sigma_v)
    Z=partition(T_rot)
    v_obs=c/10**5*(1-nu_0s/nu_0s[0])+v_0
    model=0
    for i in range(nu_0s.shape[0]):
        intensities=N_tot*g_us[i]*np.exp(-E_us[i]/T_rot)/Z
        model+=intensities*np.exp(-0.5*((v_obs[i]-vel)/sigma_v)**2)*(h*c**3*A_us[i])/(8*np.pi*nu_0s[i]**2*1e3*kB*T_rot)*phi
    return model

def fitting(J,K0,K1,velocity,intensity,v_0_init,FWHM_init,Trot=[15,100],Ntot=[13,16],sigmav=[0,10],v0=[-10,10],source_name=None,source=None):
    bounds=([Trot[0],Ntot[0],sigmav[0],v0[0]],[Trot[1],Ntot[1],sigmav[1],v0[1]])
    p=para(J,K0,K1)
    p0=[30,14,FWHM_init/(2*np.sqrt(2*np.log(2))),v_0_init]
    popt,pcov=curve_fit(
        lambda vel,T_rot,N_tot,sigma_v,v_0:multi_gaussian(vel,T_rot,N_tot,sigma_v,v_0,p),
        velocity,
        intensity,
        p0=p0,
        maxfev=10000,
        bounds=bounds,
        )
    T_rot_fit,N_tot_fit,sigma_v_fit,v_0_fit=popt
    errors=np.sqrt(np.diag(pcov))
    f=plt.figure(figsize=(8,6))
    ax=plt.gca()
    plt.plot(velocity,intensity,drawstyle='steps-mid',label="Spectrum",color="black",lw=1)
    plt.plot(velocity,multi_gaussian(velocity,*popt,p=p),label="Best-fit model",color="red",lw=2)
    plt.text(0.02,0.95,source_name,transform=ax.transAxes,color='r',fontsize=12)
    plt.text(0.02,0.90,r'$T_\mathrm{{rot}}$= {:.2f}$\pm${:.2f} K'.format(T_rot_fit,errors[0]),color='r',transform=ax.transAxes,fontsize=12)
    plt.text(0.02,0.85,r'$V_\mathrm{{lsr}}$ = {:.2f}$\pm${:.2f} km/s'.format(v_0_fit,errors[3]),color='r',transform=ax.transAxes,fontsize=12)
    plt.text(0.02,0.80,r'$\Delta V$ = {:.2f}$\pm${:.2f} km/s'.format(sigma_v_fit*2*np.sqrt(2*np.log(2)),errors[2]*2*np.sqrt(2*np.log(2))),color='r',transform=ax.transAxes,fontsize=12)
    plt.xlabel("Velocity (km/s)",fontsize=14)
    plt.ylabel(r"$T_\mathrm{{A}}$ (K)",fontsize=14)
    plt.legend(fontsize=12)
    plt.savefig(source+'.pdf')
    f.clear()
    plt.close()

data=np.loadtxt('G03519-0074_CH3CCH.dat')
velocity=data[:,0]
eta_mb=0.225
intensity=data[:,1]/eta_mb
result=fitting(J=5,K0=0,K1=2,velocity=velocity,intensity=intensity,v_0_init=34,FWHM_init=3,v0=[30,40],
               sigmav=[3/(2*np.sqrt(2*np.log(2)))-1,3/(2*np.sqrt(2*np.log(2)))+1],source_name='G035.19-00.74',source='G035')