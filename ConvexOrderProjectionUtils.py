from pysdot import PowerDiagram
from pysdot import OptimalTransport
from pysdot.radial_funcs import RadialFuncUnit
import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse.linalg import spsolve
from scipy.sparse import triu, csr_matrix


def cluster(X,masses = None,tol=1e-6):
    
    distance = cdist(X,X,"euclidean")
    
    mask = distance < tol
    upper_distance_sparse0 = csr_matrix(triu(mask,k=0).astype(int))
    upper_distance_sparse = csr_matrix(triu(mask,k=1).astype(int))
    indices = np.asarray(upper_distance_sparse.sum(axis=0)).flatten() ==0
    cluster_map_sparse = csr_matrix(upper_distance_sparse0)[indices, :]
    cluster_map_norm = csr_matrix( cluster_map_sparse.T/ cluster_map_sparse.sum(axis=1).flatten()).T

    Xcluster = np.array([cluster_map_norm@X[:,0].copy(),cluster_map_norm@X[:,1].copy()]).T 
    

        
    if masses is None:
        masses = np.sum(upper_distance_sparse[indices,:],axis=1)+1
        masses = masses/np.sum(masses)
    else:
        masses = cluster_map_sparse@masses
    
    return Xcluster, masses, cluster_map_sparse+0.0


def frank_wolfe(y,x0,lmo, dot_product, tolerance, t_max= 100 ):
    """
    Frank Wolfe algorithm with away step for the minimization of |y-x|^2/2 with y in A
    
    y : point to be projected on A
    x0 : initial guess in A
    lmo : linear minimization oracle
    """
    
   
    xt = x0.copy()
    xt_history = [x0.copy()]
    fun_history = [dot_product(y-x0,y-x0)/2.]
    active_set = [x0.copy()]
    alpha_set = [1.]
    
    t = 0 
    while t <= t_max:
        t +=1

        st = lmo(xt-y)
        dtFW = st - xt
        
        FWgap = dot_product(-(xt-y),dtFW)
        
        print('Frank-Wolfe gap : ', FWgap )
        
        if dot_product(-(xt-y),dtFW) < tolerance: 
            print ("Success")
            break
        else:
            dt = dtFW
            gamma_max = 1

        gammat = np.max((np.min((dot_product(y-xt,dt)/dot_product(dt,dt),gamma_max)),0))
        xt = xt + gammat*dt
   
        xt_history.append(xt)   
        fun_history.append(dot_product(xt-y,xt-y)/2)
        print('\riteration ',t, ', function value ', fun_history[-1], end="", flush=True) 
        
    return xt_history, fun_history

def get_centroids(domain,X,masses, rescale = True, R0 = 1.):
    
    if rescale : 
        # fit data in unit box
        X = X - np.reshape(np.mean(X,axis=0), (1,2))
        X = X/np.max(np.linalg.norm(X,axis=0))*.25*R0 + .5*R0
    

    ot = OptimalTransport(domain=domain, positions=  X, masses=masses,
                          weights=np.zeros(len(masses)), verbosity=1,linear_solver = 'Scipy')
    ot.adjust_weights()
    errorot = np.linalg.norm(ot.pd.integrals()-masses)
        
        
    print('OT error',errorot)
      
    return ot.get_centroids()

def convex_order_projection_frank_wolfe(domain, Y, X0, R0=1., masses = None,  niter= 600, tol =1e-5):
    """
    Solves Convex order projection via Frank Wolfe with away step
    Y : data
    X0 : initial point
    """ 

    N = len(Y[:,0])
    print('seeds', N)

    if masses is None: masses = 1/N*np.ones(N)
    
    dot_product = lambda X0, X1 : np.sum(X0*X1*masses.reshape((N,1))) 
    
    lmo = lambda X : get_centroids(domain,-X,masses,R0)
    
    xt_history, fun_history = frank_wolfe(Y,X0,lmo, dot_product, tol , t_max= niter )
    
    return xt_history, fun_history

def convex_order_projection(domain, Z, Y, masses0 = None, tau=None, lowerbound = None, niter= 600,show = False, lengthscale = 1.,tol =1e-5, tolcluster = 1e-6, tolfun = 1e-12):
    """
    Miminises dual problem for regularized projection on convex order:
            Z^2/2N -  ZY/N - w^2(Z,domain)/2 
    """
    
    if show:
        fig,ax = plt.subplots()    
        display0 = ax.scatter(Z[:,0], Z[:,1], c=Z[:,0])
        ax.axis('equal')
        fig.show()
        fig.canvas.draw()
        ax.axis([-1, 1.1, -1, 1.1])

    
    error_vec =[]
    fun_vec = []
    N = len(Y[:,0])
    print('seeds', N)
    Zavg = Z.copy()
    
  
        
    if tau is None: 
        #Use strong convexity for initial step
        tau = 2.
     
    
    
    for i in range(niter):
        
        Zold = Z.copy()
        Zrescale = (Z-np.min(Z)+.02)/(np.max(Z)-np.min(Z)+.02)*lengthscale
        Zcluster, masses, cluster_map = cluster(Zrescale, masses = masses0,tol=tolcluster)
        #Zcluster = Zrescale.copy()
        if len(Zcluster)<len(Z):
            print('collapse!')
         
        ot = OptimalTransport(domain=domain, positions= Zcluster+np.zeros(Zcluster.shape),
                               masses=masses, weights=np.zeros(len(masses)), verbosity=1,linear_solver = 'Scipy')
        ot.adjust_weights()
        
        errorot = np.linalg.norm(ot.pd.integrals()-masses)
        
        print('OT error',errorot)
        if errorot> 1e-6: 
            break
        
        Bcluster = ot.get_centroids()

        B = cluster_map.T@Bcluster
        minusgrad = -Z +Y-B
        
        masses = cluster_map.T@masses.reshape((len(masses),1))
        fun = ( (np.sum(Z*Z*masses)/2 - np.sum(Y*Z*masses)) + np.sum(B*Z*masses))
        if lowerbound is not None:
            #Use Polyak step
            taui = (fun - lowerbound)/np.sum(minusgrad*minusgrad*masses)
        else:    
            taui = tau/(i+1)
        Z= Z +taui*minusgrad
        Zavgold = Zavg.copy()
        Zavg = (Zavg*(i+2)*(i+1) + 2*Z*(i+2))/(i+2)/(i+3) 
       
        error = np.linalg.norm(np.linalg.norm(Zavgold-Zavg))
        #Xavg = Xavgn.copy()
        error_vec.append(error)
        fun_vec.append(fun)
        print("Average error: ", error, ", Iteration: ", i)
        if error <tol:  break
        if show:
            display0.set_offsets(Z)
            fig.canvas.draw()
            fig.canvas.flush_events()
    return Zcluster, masses, error_vec,fun_vec

