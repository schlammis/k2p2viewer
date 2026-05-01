import numpy as np
import scipy.special

def airDensity(temp,press,relHumid):     # in degC, hPa, %
    Ma= 28.96546e-3
    #Z = 0.999603
    Z = compressibility(temp,press,relHumid)
    xv =relwaterPress(temp,press,relHumid)
    R=  8.314510
    T=273.15 + temp
    p = press* 100
    ratio = 0.3780
    rho_air = p * Ma/(Z*R*T)*(1.0-xv*ratio )
    return rho_air
    
    
def buoyCorr(temp,press,relHumid, DensityOfMass):
    md = DensityOfMass
    rho_air = airDensity(temp,press,relHumid)
    buoycorr=1.0-rho_air/md
    return buoycorr

def satWaterPress(temp): # temp ind degC
    T=273.15 + temp
    A= 1.2378847e-5
    B=-1.9121316e-2
    C=33.93711047
    D=-6.3431645e3
    psv = np.exp(A*T*T+B*T+C+D/T)
    return psv
    
def waterPress(temp,press,relHumid):     # in degC, hPa, %
    p = press*100
    rH = relHumid/100
    alpha = 1.00062
    beta = 3.14e-8
    gamma=5.6e-7
    f = alpha+beta*p+gamma*temp*temp
    #f=1.0
    sp = satWaterPress(temp)
    return sp*f*rH
    

def relwaterPress(temp,press,relHumid):     # in degC, hPa, %
    wP = waterPress(temp,press,relHumid)
    p = press*100
    return wP/p


def compressibility(temp,press,relHumid):     # in degC, hPa, %
    xv =  relwaterPress(temp,press,relHumid)
    p = press*100
    T=273.15 + temp
    a0 =1.58123e-6
    a1=-2.9331e-8
    a2=1.1043e-10
    b0 = 5.707e-6
    b1=-2.051e-8
    c0=1.9898e-4
    c1=-2.376e-6
    d=1.83e-11
    e=-0.765e-8
    
    Z = 1-p/T*(a0+a1*temp+a2*temp*temp+(b0+b1*temp)*xv+(c0+c1*temp)*xv*xv)+p*p/(T*T)*(d+e*xv*xv)
    return Z
    


def FitLikeACanadianOrthoMaster(data,zmin=-2.5,zmax=2.5,\
                                order=4,z0=0,retVals=False,usez=False):
     #
    #Fitting like a Canadian. 
    #
    # V = V0 + b0 *v + b1*v*x + b2*v*x**2 + b2*v*x**3

    #X  =np.matrix((np.ones(len(t)),v,v*x,v*x**2)).T

    t   = data[:,0]
    zin = data[:,1]
    v   = data[:,2]
    V   = data[:,3]
    S   = data[:,4]
    G   = data[:,5]
    mypol = scipy.special.eval_chebyt
    z     =  (zin-zmin)/(0.5*(zmax-zmin))-1
    z0s   =  (z0-zmin)/(0.5*(zmax-zmin))-1
    X = np.zeros((len(t),len(set(S))+2+order))
    #X[:,0]=t-min(t)
    if usez:
        X[:,0] = z
    else:
        X[:,0] = np.ones(len(z))
    #
    X[:,1] = t-min(t)
    for i in range(1,order+1):
        #X = np.c_[X,v*scipy.special.eval_legendre(i,x)]
        X[:,i+1] = v*mypol(i,z)
        
    i=i+2
    oss=S[0]
    for n,s in enumerate(S):
        if s!=oss:
            i=i+1
        X[n,i] = v[n]
        oss=s
    X  =np.matrix(X)
    C =((X.T*X).I)
    fit_pars=C*X.T*np.matrix(V).T
    fit_vals =np.array( X*fit_pars)[:,0]  
    C2 = np.dot(V-fit_vals,V-fit_vals)
    NDF = len(V)
    uncert =np.sqrt( C2/NDF)
    vals = np.array(fit_pars)[order+2:]
    cor=0
    for i in range(1,order+1):
        cor += fit_pars[i+1,0]*mypol(i,z0s)
        
    ts =[]
    grp =[]
    for s in set(S):
        ts.append(np.mean([i for i,j in zip(t,S) if j==s]))
        grp.append(np.mean([i for i,j in zip(G,S) if j==s]))

    ts = np.array(ts)
    Blz0=np.array(vals+cor)[:,0]
    grp = np.array(grp)
    if retVals==False:
        return np.c_[ts,Blz0,grp],C2,NDF,fit_pars
    else:
        return np.c_[ts,Blz0,grp],C2,NDF,fit_pars,fit_vals

def calcProfile(pars,order,z,zmin=-5,zmax=5,withOffset=False):
    rz = (z-zmin)/(0.5*(zmax-zmin))-1
    mypol = scipy.special.eval_chebyt
    ret = np.zeros(len(rz))
    i=1
    for p in np.array(pars)[2:order+2,0]:
        ret += p*mypol(i,rz)
        i=i+1
    if withOffset:
        ret = ret + pars[order+2,0]
    return ret

def csinefit(t,y,T,hars=[1,2,3,4]):
    """
    Assume the base function is A*sin(wt+phi) = Acos(phi)sin(wt)+ Asin(phi)cos(wt)
    Hence C = Asin(phi) and S =A cos(phi)  C/S = tan(phi)
    So the best fit function is S*sin(wt)+C*cos(wt)
    Re[ (C-i*S)*exp(i wt) ]  = C*cos(wt) + S sin(i wt)
    
    """
    w= 2*np.pi/T
    wt = t*w
    O = np.ones(len(y))
    X = np.array(O)
    for i in  hars:
        C = np.cos(wt*i)
        S = np.sin(wt*i)
        X = np.vstack((X,C,S))
    X = np.matrix(X.T)
    C = (X.T*X).I    # covariance matrix
    fit_pars=C*X.T*np.matrix(y).T
    fit_vals =np.array( X*fit_pars)[:,0]  

    C2 = np.dot(y-fit_vals,y-fit_vals)
    Ndf = len(t)-1-2*len(hars)
    sigma2 = C2/Ndf
    CS = C*sigma2   # scaled covariance matrix
    off = fit_pars[0,0]   
    Ccof = np.array(  [ fit_pars[i*2-1,0]- 1j*fit_pars[i*2,0] for i in range(1,1+len(hars))] )
    return off,Ccof,C2

def calcsine(t,T,off,Ccof,hars=[1,2,3,4]):
    w= 2*np.pi/T
    wt = t*w
    y = off* np.ones(len(t))
    for h,C in zip(hars,Ccof):
        y+= np.real(C* np.exp(1j*w*t*h))
    return y
                     
def deatten(T,tau,Ccof,hars=[1,2,3,4]):
    mul = np.array([ 1.0/np.sinc(i/T*tau) for i in hars])
    pha = np.array([ np.exp(-1j*i*np.pi/T*tau) for i in hars])
    return mul*pha*Ccof

def CanadianFitMachine(data,zmin=-2.5,zmax=2.5,order=4,z0=0,retVals=False,zorder=4):
    """
    Impoved version of FitLikeACanadianOrthoMaster. The following changes were made:
    The order in the matrix (and hence the returned) parameters were changed to make it
    easier to add things at the rear. 
    The first NS entries are BL0. NS is the number of sets.
    The next NS entried are the Bl) drift in each set
    Then we have order number of parameters for the  fit polynomials.
    Then we have extra fit paramters. Such as a voltage offset V0, a voltage drift
    a dependence of the voltage on z.
    """

    t   = data[:,0]
    zin = data[:,1]
    v   = data[:,2]
    V   = data[:,3]
    S   = data[:,4]
    G   = data[:,5]
    mypol = scipy.special.eval_chebyt
    #mypol = scipy.special.eval_legendre
    # Scaled quantities
    z     =  (zin-zmin)/(0.5*(zmax-zmin))-1
    z0s   =  (z0-zmin)/(0.5*(zmax-zmin))-1
    ts    = (t - min(t))
    NS = len(set(S))
    NExtra = 2+zorder
    X = np.zeros((len(t),2*NS+order+NExtra))

    sett0={}
    for s in set(S):
        ix = np.where(S==s)[0]
        sett0[s] = np.mean(ts[ix])
    i=0
    oss=S[0]
    for n,s in enumerate(S):
        if s!=oss:
            i=i+1
        X[n,i] = v[n]
        oss=s
    oss=S[0]
    i=i+1
    for n,s in enumerate(S):
        if s!=oss:
            i=i+1
        X[n,i] = v[n]*(ts[n]-sett0[s])
        oss=s
    ipol =i
    GSI = np.array(z**0)
    for ii in range(1,order+1):
        GSI=np.vstack((GSI,z**ii))
    GSO = np.array(GSI)
    for ii in range(1,order+1):
        for jj in range(0,ii):
            s= np.dot(GSI[ii,:],GSO[jj,:])/np.dot(GSO[jj,:],GSO[jj,:])
            GSO[ii,:] = GSO[ii,:]-s*GSO[jj,:]
    for j in range(1,order+1):
        X[:,i+j] = v*mypol(j,z)
    #for j in range(1,order+1):
        #X[:,i+j] = v*GSO[j,:]
    i=i+j+1
    X[:,i] = ts
    i=i+1
    X[:,i] = np.ones(len(ts))
    for j in range(1,zorder+1):
        X[:,i+j] = mypol(j,z)
    
    X  =np.matrix(X)
    C =((X.T*X).I)
    fit_pars=C*X.T*np.matrix(V).T
    fit_vals =np.array( X*fit_pars)[:,0]  
    C2 = np.dot(V-fit_vals,V-fit_vals)
    NDF = len(V)
    uncert =np.sqrt( C2/NDF)
    vals = np.array(fit_pars)[0:NS]
    cor=0
    for j in range(1,order+1):
        cor += fit_pars[ipol+j,0]*mypol(j,z0s)   
    ts =[]
    grp =[]
    for s in set(S):
        ts.append(np.mean([i for i,j in zip(t,S) if j==s]))
        grp.append(np.mean([i for i,j in zip(G,S) if j==s]))

    ts = np.array(ts)
    Blz0=np.array(vals+cor)[:,0]
    grp = np.array(grp)
    if retVals==False:
        return np.c_[ts,Blz0,grp],C2,NDF,fit_pars
    else:
        return np.c_[ts,Blz0,grp],C2,NDF,fit_pars,fit_vals





def FitLikeACanadianOrthoMasterNew(data,zmin=-2.5,zmax=2.5,\
                                order=4,z0=0,maxhars=7,\
                                    f0= 0.5,tau=0.1,mode='raw'):
    """   
    mode=='raw':  fits the raw data as before:
    mode=='sine': fits a sine to the data and then Canadian fit. harmonics are fit up to ahrmonics, f0 is the frequency of the sine
    mode=='atten':fits a sine to the data  and the ettnuates the coefficients and then Canadian fit.
                     harmonics are fit up to ahrmonics, f0 is the frequency of the sine, tau is dvm and ifo aperture
    """
    if mode not in ['raw','sine','atten']:
        print("mode invalid. Has to be 'raw', 'sine' or 'atten'")
        return
              
    if mode=='raw':
        return FitLikeACanadianOrthoMaster(data,zmin=zmin,zmax=zmax,order=order,z0=z0)

    if maxhars<order+1:
        print('Warning: maxhars should be at least order+1')

    hars = range(1,1+maxhars)        
    # The following are the raw variables
    t   = data[:,0]
    zin = data[:,1]
    v   = data[:,2]
    V   = data[:,3]
    S   = data[:,4]
    G   = data[:,5]

    # The following are the sine fited variables
    ts   = np.array(t)
    zins = np.array(zin)
    vs  = np.array(v)
    Vs   = np.array(V)
    Ss   = np.array(S)
    Gs   = np.array(G)
    
    if mode=='atten':
        # The followwing variables are the attenuated variables
        t_   = np.array(t)
        zin_ = np.array(zin)
        v_   = np.array(v)
        V_   = np.array(V)
        S_   = np.array(S)
        G_   = np.array(G)

    T0=1.0/f0
    for s in set(S):
        ix = np.where(S==s)[0]
        pos_off,pos_cof,pos_C2    = csinefit(t[ix],zin[ix],T0,hars=hars)
        velo_off,velo_cof,velo_C2 = csinefit(t[ix],v[ix],T0,hars=hars)
        volt_off,volt_cof,volt_C2 = csinefit(t[ix],V[ix],T0,hars=hars)
        zins[ix] = calcsine(t[ix],T0,pos_off,pos_cof,hars=hars)
        vs[ix] = calcsine(t[ix],T0,velo_off,velo_cof,hars=hars)
        Vs[ix] = calcsine(t[ix],T0,volt_off,volt_cof,hars=hars)   
        if mode== 'atten':   
            apos_cof = deatten(T0,tau,pos_cof,hars=hars)
            avelo_cof = deatten(T0,tau,velo_cof,hars=hars)
            avolt_cof = deatten(T0,tau,volt_cof,hars=hars)
            zin_[ix] = calcsine(t[ix],T0,pos_off,apos_cof,hars=hars)
            v_[ix] = calcsine(t[ix],T0,velo_off,avelo_cof,hars=hars)
            V_[ix] = calcsine(t[ix],T0,volt_off,avolt_cof,hars=hars)     

    if mode=='sine': 
        datas = np.vstack((ts,zins,vs,Vs,Ss,Gs)).T
        return FitLikeACanadianOrthoMaster(datas,zmin=zmin,zmax=zmax,order=order,z0=z0)
    data_ = np.vstack((t_,zin_,v_,V_,S_,G_)).T
    return  FitLikeACanadianOrthoMaster(data_,zmin=zmin,zmax=zmax,order=order,z0=z0)
    
def calcProfileNew(pars,order,NS,z,zmin=-5,zmax=5,withOffset=False):
    rz = (z-zmin)/(0.5*(zmax-zmin))-1
    mypol = scipy.special.eval_chebyt
    ret = np.zeros(len(rz))
    i=1
    for p in np.array(pars)[2*NS:2*NS+order,0]:
        ret += p*mypol(i,rz)
        i=i+1
    if withOffset:
        ret = ret + pars[order+1,0]
    return ret


def fit_sine(t,y,T,hars=[1,2,3,4]):
    """
    Assume the base function is A*sin(wt+phi) = Acos(phi)sin(wt)+ Asin(phi)cos(wt)
    Hence C = Asin(phi) and S =A cos(phi)  C/S = tan(phi)
    """
    w= 2*np.pi/T
    wt = t*w
    O = np.ones(len(y))
    X = np.array(O)
    for i in  hars:
        C = np.cos(wt*i)
        S = np.sin(wt*i)
        X = np.vstack((X,C,S))
    X = np.matrix(X.T)
    C = (X.T*X).I    # covariance matrix
    fit_pars=C*X.T*np.matrix(y).T
    fit_vals =np.array( X*fit_pars)[:,0]  
    C2 = np.dot(y-fit_vals,y-fit_vals)
    Ndf = len(t)-1-2*len(hars)
    sigma2 = C2/Ndf
    CS = C*sigma2   # scaled covariance matrix
    off = fit_pars[0,0]
    amp=[]
    phase=[]
    for i in range(1,1+len(hars)):
        amp.append(np.sqrt(fit_pars[i*2-1,0]**2+fit_pars[i*2,0]**2))
        phase.append(np.arctan2(fit_pars[i*2-1,0],fit_pars[i*2,0]))
    return off,amp,phase,C2,fit_vals
#    return np.array(fit_pars)[:,0],fit_vals,C2

def findT0(t,data,hars,T0guess,dT=1):
    T0mi = T0guess-dT
    T0ma = T0guess+dT
    c2a=[]
    TT =np.linspace(T0mi,T0ma,10)
    for tt in TT:
        _,_,_,C2,_=fit_sine(t,data,tt,hars=hars)
        c2a.append(C2)
    ix =np.argmin(c2a)
    T0 = TT[ix]
    if ix>=8:
        return findT0(t,data,hars,T0guess+dT/2,dT)
    if ix<=1:
        return findT0(t,data,hars,T0guess-dT/2,dT)
    if dT>0.002:
        return findT0(t,data,hars,T0guess,dT=dT/5)
    else:
        return T0


def findmin2(t,y,tmin,tmax,hars=[1,2,3,4]):
    C2=[]
    TT = np.linspace(0,tmax-tmin,5)
    for tt in TT:
        o,a,p,c2,_=fit_sine(t,y,tmin+tt,hars)
        C2.append(c2)
    pf=np.polyfit(TT,C2,2)
    # y = ax^2+bx+c -> 2*ax+b=0 x= -b/(2a)
    return tmin-pf[1]/2/pf[0]

def myatten(T,tau,hars,amps,phase):
    f0= 1./T
    nea=[]
    nep=[]
    for h,a,p in zip(hars,amps,phase):
        f =h*f0
        nea.append(a/np.sinc(f*tau))
        nep.append(p+np.pi*f*tau)
    return nea,nep


def calcAP(t,T,hars,O,amps,phase):
    y = np.ones(len(t))*O
    w= 2*np.pi/T
    wt =w*t
    for h,a,p in zip(hars,amps,phase):
        y += a*np.sin(wt*h+p)
    return y


