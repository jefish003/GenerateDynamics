# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 13:06:30 2023

@author: jefis
"""
import numpy as np
import networkx as nx
from copy import deepcopy
from scipy.integrate import odeint
from sklearn.neighbors import KernelDensity
from matplotlib import pyplot as plt
from scipy.linalg import schur,expm
from scipy.spatial.distance import pdist
from tqdm import tqdm

class laplacian_dynamics:
    
    def __init__(self):
        self.graph = None
        self.init_cond = None
    
    
    def set_graph(self,G):
        """A function to set the graph. Note it is assumed to be a networkx graph...
        
        Inputs: G - a networkx graph.
        """
        self.graph = G
    
    def set_init_cond(self,init_cond):
        """For setting a particular initial condition"""
        self.init_cond=init_cond

    def return_graph(self):
        return self.graph
    
    def return_init_cond(self):
        return self.init_cond

    def convert_adjacency(self,A,Type='DiGraph'):
        """To convert an adacency matrix to a networkx graph and store the graph
        internally
        
        Inputs: A - (n x n) Adjacency matrix in numpy array form. Note must be acceptable 
                    form for networkx to convert to a graph
                Type - either 'Graph' or 'DiGraph' depending on the type of graph
        
        Outputs: 
                graph - The networkx graph. 
        """
        
        if Type == 'DiGraph':
            self.graph = nx.DiGraph(A)
        
        elif Type == 'Graph':
            self.graph = nx.Graph(A)
        
        else:
            raise ValueError("Only Type = 'DiGraph' or Type = 'Graph' is allowed")
        
        return self.graph

    def linear_dynamics(self,x,t,A):
        x = np.dot(A,x)
        return np.array(x).flatten()

    def rossler(self,x,t,LH,params):
        a,b,c = params
        lenx = len(x)
        dx = deepcopy(x)
        X = x[0:lenx:3]
        Y = x[1:lenx:3]
        Z = x[2:lenx:3]
        dx[0:lenx:3] = -Y-Z
        dx[1:lenx:3] = X+a*Y
        dx[2:lenx:3] = b+Z*(X-c)
        return np.array(dx-np.dot(LH,x)).flatten()
        
    def lorenz(self,x,t,LH,params):
        a,b,c = params
        lenx = len(x)
        dx = deepcopy(x)
        X = x[0:lenx:3]
        Y = x[1:lenx:3]
        Z = x[2:lenx:3]
        dx[0:lenx:3] = a*(Y-X)
        dx[1:lenx:3] = X*(b-Z)-Y
        dx[2:lenx:3] = X*Y-c*Z
        return np.array(dx-np.dot(LH,x)).flatten()

    def brusselator(self,x,t,LH,params):
        a,b = params
        lenx = len(x)
        dx = deepcopy(x)
        X = x[0:lenx:2]
        Y = x[1:lenx:2]
        dx[0:lenx:2] = 1-(a+1)*X+b*Y*X**2
        dx[1:lenx:2] = a*X-b*Y*X**2
        
        return np.array(dx-np.dot(LH,x)).flatten()            

    def vanderpol(self,x,t,LH,params):
        a = params[0]
        lenx = len(x)
        dx = deepcopy(x)
        X = x[0:lenx:2]
        Y = x[1:lenx:2]
        dx[0:lenx:2] = Y
        dx[1:lenx:2] = -X+a*(1-X**2)*Y
        
        return np.array(dx-np.dot(LH,x)).flatten()     

    def wienbridge(self,x,t,LH,params):
        a,b,c = params
        lenx = len(x)
        dx = deepcopy(x)
        X = x[0:lenx:2]
        Y = x[1:lenx:2]
        dx[0:lenx:2] = -X+Y-(a*Y-b*Y**3+c*Y**5)
        dx[1:lenx:2] = -(-X+Y-(a*Y-b*Y**3+c*Y**5))-Y
        
        return np.array(dx-np.dot(LH,x)).flatten()      

    def convert_graph_to_laplacian(self,G):
        A = nx.adjacency_matrix(G).todense()
        L = self.get_Laplacian(A)
        return L             

    def get_Laplacian(self,A,return_eigvals=False):
        L = np.matrix(np.zeros((A.shape[0],A.shape[0])))
        np.fill_diagonal(L,np.sum(A,1))
        L = L-A
        if not return_eigvals:
            return L
        else:
            E = np.linalg.eigvals(L)
            return L,E
    

    def continuous_time_linear_dynamics(self,G=None,tmax=100,timestep=0.1,init_cond_type='normal',init_cond=None,init_cond_params=[0,1],init_cond_offset=0):
        """Generate continuous time Laplacian (i.e. diffusive) type linear dynamics
        Inputs: 
               G - a networkx graph, if set to None then the internally stored graph will be used
               tmax - the max value to integrate to
               timestep - the integration time steps
               init_cond_type - the type of random initial conditions (can be 'normal' or 'uniform' for instance, see docs for available types)
               init_cond - you can specify the initial condition here, if this is specified then init_cond_type will be ignored, if this is None then the initial condition will be generated according to init_cond_type
               init_cond_parameters - a list of the parameters for the distribution type. For instance if init_cond_type = 'normal' then two parameters should be specified, the mean and the variance
               init_cond_offset - offset to be added to all initial condition values (to shift the distribution...)
        Outputs: 
                sol - the solution after integration
                t   - the vector of integration time points
               
        """
        if G is None:
            if self.graph is not None:
                G = deepcopy(self.graph)
            else:
                raise ValueError("No graph to perform dynamics on...")
        
        else:
            self.graph=G
        L = self.convert_graph_to_laplacian(G)
        t = np.arange(0,tmax,step=timestep)
        n = L.shape[0]
        lenx0 = n
        if init_cond_type =='normal':
            x0 = np.random.normal(init_cond_params[0],init_cond_params[1],n)
        elif init_cond_type =='uniform':
            x0 = np.random.uniform(init_cond_params[0],init_cond_params[1],lenx0)+init_cond_offset
        
        elif init_cond_type == 'laplace':
            x0 = np.random.uniform(init_cond_params[0],init_cond_params[1],lenx0)+init_cond_offset
        
        elif init_cond_type == 'exponential':
            x0 = np.random.exponential(init_cond_params[0],lenx0)+init_cond_offset
        
        elif init_cond_type == 'rayleigh':
            x0 = np.random.exponential(init_cond_params[0],lenx0)+init_cond_offset
        
        elif init_cond_type == 'beta':
            x0 = np.random.beta(init_cond_params[0],init_cond_params[1],lenx0)+init_cond_offset
        
        elif init_cond_type == 'gamma':
            x0 = np.random.beta(init_cond_params[0],init_cond_params[1],lenx0)+init_cond_offset             
        
        elif init_cond_type == 'gumbel':
            x0 = np.random.gumbel(init_cond_params[0],init_cond_params[1],lenx0)+init_cond_offset

        elif init_cond_type == 'chisquare':
            x0 = np.random.chisquare(init_cond_params[0],lenx0)+init_cond_offset

        elif init_cond_type == 'logistic':
            x0 = np.random.logistic(init_cond_params[0],init_cond_params[1],lenx0)+init_cond_offset
  
        elif init_cond_type == 'lognormal':
            x0 = np.random.lognormal(init_cond_params[0],init_cond_params[1],lenx0)+init_cond_offset                
        
        elif init_cond_type == 'pareto':
            x0 = np.random.pareto(init_cond_params[0],lenx0)+init_cond_offset                
        
        elif init_cond_type == 'f':
            x0 = np.random.f(init_cond_params[0],init_cond_params[1],lenx0)+init_cond_offset                
        
        elif init_cond_type == 'vonmises':
            x0 = np.random.vonmises(init_cond_params[0],init_cond_params[1],lenx0)+init_cond_offset               
        
        elif init_cond_type == 'wald':
            x0 = np.random.wald(init_cond_params[0],init_cond_params[1],lenx0)+init_cond_offset                

        elif init_cond_type == 'weibull':
            x0 = np.random.weibull(init_cond_params[0],lenx0)+init_cond_offset

        elif init_cond_type == 'zipf':
            x0 = np.random.zipf(init_cond_params[0],lenx0)+init_cond_offset
        
        else:
            raise ValueError("This init_cond_type is not implemented see docs for types which are currently implemented")                    
        sol = odeint(self.linear_dynamics,x0,t,args=(-L,))
        return sol,t
    
    def generate_initial_condition(self,lenx0,method='random',init_cond_type='normal',init_cond_params=[0,1],init_cond_offset=0,p_norm=2,scale=1):
        """Method for generating initial conditions.
            
            Inputs: lenx0 - the size of the initial condition vector
                    method - the method for generating initial conditions. 'random' gives random ic's, normalized gives normalized ic's 
                    init_cond_type - the type of random numbers to draw from, for example to draw from a normal distribution 'normal'
                    init_cond_params - the parameters for the init_cond_type distribution. Must be an iterable
                    init_cond_offset - how much to offset the initial condition by. 
                    p_norm - if method is 'normalized', the value of the p_norm to normalize by
                    scale - if method is 'normalized' this will scale the initial condition to be of a certain size.
                    
            Outputs:
                    x0 - the initial condition.
        """
        
        if init_cond_type =='normal':
            x0 = np.random.normal(init_cond_params[0],init_cond_params[1],lenx0)+init_cond_offset
        
        elif init_cond_type =='uniform':
            x0 = np.random.uniform(init_cond_params[0],init_cond_params[1],lenx0)+init_cond_offset
        
        elif init_cond_type == 'laplace':
            x0 = np.random.uniform(init_cond_params[0],init_cond_params[1],lenx0)+init_cond_offset
        
        elif init_cond_type == 'exponential':
            x0 = np.random.exponential(init_cond_params[0],lenx0)+init_cond_offset
        
        elif init_cond_type == 'rayleigh':
            x0 = np.random.exponential(init_cond_params[0],lenx0)+init_cond_offset
        
        elif init_cond_type == 'beta':
            x0 = np.random.beta(init_cond_params[0],init_cond_params[1],lenx0)+init_cond_offset
        
        elif init_cond_type == 'gamma':
            x0 = np.random.beta(init_cond_params[0],init_cond_params[1],lenx0)+init_cond_offset             
        
        elif init_cond_type == 'gumbel':
            x0 = np.random.gumbel(init_cond_params[0],init_cond_params[1],lenx0)+init_cond_offset

        elif init_cond_type == 'chisquare':
            x0 = np.random.chisquare(init_cond_params[0],lenx0)+init_cond_offset

        elif init_cond_type == 'logistic':
            x0 = np.random.logistic(init_cond_params[0],init_cond_params[1],lenx0)+init_cond_offset
  
        elif init_cond_type == 'lognormal':
            x0 = np.random.lognormal(init_cond_params[0],init_cond_params[1],lenx0)+init_cond_offset                
        
        elif init_cond_type == 'pareto':
            x0 = np.random.pareto(init_cond_params[0],lenx0)+init_cond_offset                
        
        elif init_cond_type == 'f':
            x0 = np.random.f(init_cond_params[0],init_cond_params[1],lenx0)+init_cond_offset                
        
        elif init_cond_type == 'vonmises':
            x0 = np.random.vonmises(init_cond_params[0],init_cond_params[1],lenx0)+init_cond_offset               
        
        elif init_cond_type == 'wald':
            x0 = np.random.wald(init_cond_params[0],init_cond_params[1],lenx0)+init_cond_offset                

        elif init_cond_type == 'weibull':
            x0 = np.random.beta(init_cond_params[0],init_cond_params[1],lenx0)+init_cond_offset

        elif init_cond_type == 'zipf':
            x0 = np.random.beta(init_cond_params[0],init_cond_params[1],lenx0)+init_cond_offset
        
        else:
            raise ValueError("This init_cond_type is not implemented see docs for types which are currently implemented")  
        
        if method=='normalized':
            x0 = (x0/np.linalg.norm(x0,p_norm))*scale
        
        elif method =='random':
            do_nothing =1
        
        else:
            raise ValueError("This method for initial conditions is not allowed. See docs for allowable methods.")
        
        return x0
                    
    
    def continuous_time_nonlinear_dynamics(self,G=None,L=None,tmax=100,timestep=0.1,method='random',init_cond_type='normal',init_cond=None,init_cond_params=[0,1],init_cond_offset=0,p_norm=2,scale=1,dynamics_type='Rossler',dynamics_params=[0.2,0.2,7],coupling_matrix=None,coupling_strength=1):
        """Generate continuous time Laplacian (i.e. diffusive) type non-linear dynamics
        Inputs: 
               G - a networkx graph, if set to None then the internally stored graph will be used
               L - the graph Laplacian, note this will be ignored if G is not None
               tmax - the max value to integrate to
               timestep - the integration time steps
               method - the method for drawing initial conditions. See docs for more information
               init_cond_type - the type of random initial conditions (can be 'normal' or 'uniform' for instance, see docs for available types)
               init_cond - you can specify the initial condition here, if this is specified then init_cond_type will be ignored, if this is None then the initial condition will be generated according to init_cond_type
               init_cond_parameters - a list of the parameters for the distribution type. For instance if init_cond_type = 'normal' then two parameters should be specified, the mean and the variance
               init_cond_offset - how much to offset the initial condition by, added to each entry of the initial condition
               p_norm - the value of p in the p-norm if method = 'normalized'
               scale - how much to scale the initial condition by if method = 'normalized'
               dynamics_type - the type of dynamics to use, for instance 'Rossler' will give Rossler type dynamics, see docs for allowable dynamics types
               dynamics_params - a list of the parameters for the dynamics type, for instance 'Rossler' has 3 parameters to specify
               coupling_matrix - if coupling_function_type is matrix then this is the matrix
               coupling_strength - the coupling strength
        Outputs: 
                sol - the solution after integration
                t   - the vector of integration time points
               
        """        
        if G is None:
            if self.graph is not None:
                G = deepcopy(self.graph)
            elif L is not None:
                do_nothing = 1
            else:
                raise ValueError("No graph to perform dynamics on...")
        else:
            self.graph = G
        
        if G is not None:
            L = self.convert_graph_to_laplacian(G)
        n = len(L)
        t = np.arange(0,tmax,step=timestep)
        

        
        if dynamics_type =='Rossler':
            lenx0 = 3*n
            if init_cond is None:
                self.init_cond = self.generate_initial_condition(lenx0,method=method,init_cond_type=init_cond_type,init_cond_params=init_cond_params,init_cond_offset=init_cond_offset,p_norm=p_norm,scale=scale)
            
            else:
                self.init_cond = init_cond
            if coupling_matrix is None:
                #assume coupling only through the x component
                H = np.eye(3)
                H[1,1] = 0
                H[2,2] = 0
                
            else:
                H = coupling_matrix
            
            x0 = self.init_cond
            LH = coupling_strength*np.kron(L,H)
            sol = odeint(self.rossler,x0,t,args=(LH,dynamics_params)) 
            
        elif dynamics_type=='Lorenz':
            lenx0 = 3*n
            if init_cond is None:
                self.init_cond = self.generate_initial_condition(lenx0,method=method,init_cond_type=init_cond_type,init_cond_params=init_cond_params,init_cond_offset=init_cond_offset,p_norm=p_norm,scale=scale)
            else:
                self.init_cond = init_cond  
                    
            if coupling_matrix is None:
                #assume coupling only through the x component
                H = np.eye(3)
                H[1,1] = 0
                H[2,2] = 0
                
            else:
                H = coupling_matrix
            
            x0 = self.init_cond
            LH = coupling_strength*np.kron(L,H)
            sol = odeint(self.lorenz,x0,t,args=(LH,dynamics_params))   
            
        elif dynamics_type=='VanDerPol':
            lenx0 = 2*n
            if init_cond is None:
                self.init_cond = self.generate_initial_condition(lenx0,method=method,init_cond_type=init_cond_type,init_cond_params=init_cond_params,init_cond_offset=init_cond_offset,p_norm=p_norm,scale=scale)
            else:
                self.init_cond = init_cond
                    
            if coupling_matrix is None:
                #assume coupling only through the x component
                H = np.eye(2)
                H[1,1] = 0
                
            else:
                H = coupling_matrix
            
            x0 = self.init_cond
            LH = coupling_strength*np.kron(L,H)
            sol = odeint(self.vanderpol,x0,t,args=(LH,dynamics_params)) 
            
        elif dynamics_type=='Wienbridge':
            lenx0 = 2*n
            if init_cond is None:
                self.init_cond = self.generate_initial_condition(lenx0,method=method,init_cond_type=init_cond_type,init_cond_params=init_cond_params,init_cond_offset=init_cond_offset,p_norm=p_norm,scale=scale)
            else:
                self.init_cond = init_cond
                    
            if coupling_matrix is None:
                #assume coupling only through the x component
                H = np.eye(2)
                H[1,1] = 0
                
            else:
                H = coupling_matrix
            
            x0 = self.init_cond
            LH = coupling_strength*np.kron(L,H)
            sol = odeint(self.wienbridge,x0,t,args=(LH,dynamics_params))            

        elif dynamics_type=='Brusselator':
            lenx0 = 2*n
            if init_cond is None:
                self.init_cond = self.generate_initial_condition(lenx0,method=method,init_cond_type=init_cond_type,init_cond_params=init_cond_params,init_cond_offset=init_cond_offset,p_norm=p_norm,scale=scale)
            else:
                self.init_cond = init_cond
                    
            if coupling_matrix is None:
                #assume coupling only through the x component
                H = np.eye(2)
                H[1,1] = 0
                
            else:
                H = coupling_matrix
            
            x0 = self.init_cond
            LH = coupling_strength*np.kron(L,H)
            sol = odeint(self.brusselator,x0,t,args=(LH,dynamics_params))            
 

        
        else:
            raise ValueError("Dynamics must be of allowed type, see docs for allowed types")
        
        return sol,t
    
    
    def nonlinear_find_time_to_sync(self,x,t,d,Tol=1e-6):
        """For finding the time to synchronization.
            
           Inputs:
                  x - the (m x D) dimensional time series. Here D = nd where n
                      n is the number of oscillators and d is the dimension of 
                      each oscillator. m is the number of time samples
                 
                 t - a vector mapping each time sample to a particular integration
                     time.
                
                d - the dimension of an isolated oscillator.
                
                Tol - The tolerance for stating the synchronization has been 
                      attained. If this tolerance is never reached then sync_time = inf
         
           Outputs:
                   sync_time - the time to synchronization, may be inf if sync is never 
                               reached in the time series...
        """
        sync_time = np.inf
        if d == 1:
            for i in range(x.shape[0]):
                dist = np.max(pdist(x[i,:]))
                if dist<Tol:
                    sync_time = t[i]
                    break
        
        elif d>1:
            Dim = x.shape[1]
            
            for i in range(x.shape[0]):
                concat = x[i,0:Dim:d].reshape(-1,1)
                for j in range(1,d):
                    concat = np.concatenate((concat,x[i,j:Dim:d].reshape(-1,1)),axis=1)
                dist = np.max(pdist(concat))
                #print(dist)
                if dist<Tol:
                    sync_time = t[i]
                    break 
        
        else:
            raise ValueError("Dimension (d) of oscillators cannot be negative")
                    
         
        return sync_time
            
    def average_ctnd(self,G=None,L=None,tmax=100,timestep=0.1,init_cond_type='normal',init_cond=None,init_cond_params=[0,1],init_cond_offset=0,dynamics_type='Rossler',dynamics_params=[0.2,0.2,7],coupling_matrix=None,coupling_strength=1,num_average=100,d=3,Tol=1e-6):
        """Generate continuous time Laplacian (i.e. diffusive) type non-linear dynamics and find the average time to synchronization.
        Inputs: 
               G - a networkx graph, if set to None then the internally stored graph will be used
               L - the graph Laplacian, note this will be ignored if G is not None
               tmax - the max value to integrate to
               timestep - the integration time steps
               init_cond_type - the type of random initial conditions (can be 'normal' or 'uniform' for instance, see docs for available types)
               init_cond - you can specify the initial condition here, if this is specified then init_cond_type will be ignored, if this is None then the initial condition will be generated according to init_cond_type
               init_cond_parameters - a list of the parameters for the distribution type. For instance if init_cond_type = 'normal' then two parameters should be specified, the mean and the variance
               init_cond_offset - how much to offset the initial condition by, added to each entry of the initial condition
               dynamics_type - the type of dynamics to use, for instance 'Rossler' will give Rossler type dynamics, see docs for allowable dynamics types
               dynamics_params - a list of the parameters for the dynamics type, for instance 'Rossler' has 3 parameters to specify
               coupling_matrix - if coupling_function_type is matrix then this is the matrix
               coupling_strength - the coupling strength
               num_average - the number of trials to average over
               d - the number of dimensions of the oscillator used
               Tol - the synchronization tolerance
        Outputs: 
                Mean - the mean time to synchronization, excluding the times it failed to synchronize
                numfail   - the number of times it failed to synchronize
               
        """                    
        av_list = []
        numfail = 0
        for i in tqdm(range(num_average)):
            x,t = self.continuous_time_nonlinear_dynamics(G,L,tmax,timestep,init_cond_type,init_cond,init_cond_params,init_cond_offset,dynamics_type,dynamics_params,coupling_matrix,coupling_strength)
            sync_time = self.nonlinear_find_time_to_sync(x,t,d,Tol)
            if sync_time !=np.inf:
                av_list.append(sync_time)
            
            else:
                numfail = numfail+1
            
        if len(av_list) == 0:
            Mean = np.inf
        
        else:
            av_list = np.array(av_list)
            Mean = np.mean(av_list)
        
        return Mean,numfail
            
        

class divergences:
    def __init__(self):
        self.init_state = 0
    
    
    def generate_distributions_from_dynamics(self,x,y,Type,numbins,kernel,bandwidth=None,bandwidth_compare=None):
        """A class for estimating a pair of distributions from data, using either the empirical histogram method
        or kernel density estimation (kde). The binning will be decided jointly for x and y rather than individually.
        
        Inputs: 
                x - a 1d vector (or array) with n-samples
                y - a second 1d vector or array with n-samples
                Type - the method for estimation either 'empirical' or 'kde'
                numbins - the number of bins to estimate the distribution over (both methods need this)
                kernel - if Type is 'kde', this is the type of kernel. see docs for supported kernels
                bandwidth - if Type is kde this will be the bandwidth for the variable 'x', if bandwidth is
                            None then the rule of thumb for gaussian kde is employed
                bandwidth_comparision - same as bandwidth but for variable y
        
        Outputs: 
                P - the estimated distribution associated with the variable x
                Q - the estimated distribution associated with the variable y
        """
        
        if Type == 'empirical':
            min_binx = np.min(x)
            max_binx = np.max(x)
            min_biny = np.min(y)
            max_biny = np.max(y)
            min_bin = np.min([min_binx,min_biny])
            max_bin = np.max([max_binx,max_biny])
            flr = np.floor(min_bin)
            ceil = np.ceil(max_bin)  
            bins = np.linspace(flr,ceil,numbins+1)
            hist1 = np.histogram(x,bins=bins)
            P = hist1[0]/np.sum(hist1[0])
            hist1 = np.histogram(y,bins=bins)
            Q = hist1[0]/np.sum(hist1[0])
            
        elif Type == 'kde':
            if kernel == 'gaussian':
                if bandwidth is None:
                    #rule of thumb bandwidth
                    sigma = np.std(x)
                    n = len(x)
                    bandwidth = 1.06*sigma*n**(-1.5)
                
                if bandwidth_compare is None:
                    
                    sigma = np.std(y)
                    n = len(y)
                    bandwidth_comparison = 1.06*sigma*n**(-1.5)
                
              
            else:
                raise ValueError("Currently the only supported kernel type for kde is gaussian, support for other kernels in the future is expected soon")
            kde = KernelDensity(kernel=kernel,bandwidth=bandwidth).fit(x)
            min_kde1 = np.min(x)
            max_kde1 = np.max(x)
            min_kde2 = np.min(y)
            max_kde2 = np.max(y)
            min_kde = np.min([min_kde1,min_kde2])
            max_kde = np.max([max_kde1,max_kde2])
            space = np.linspace(min_kde,max_kde,numbins+2)[1:-1].reshape(-1,1)
            log_dens = kde.score_samples(space)
            #print("This is the log_dens: ", log_dens)
            P = np.exp(log_dens)
            #now for the comparison dist...
            kde = KernelDensity(kernel=kernel,bandwidth=bandwidth_comparison).fit(y)
            log_dens = kde.score_samples(space) 
            Q = np.exp(log_dens)  
        else:
            raise ValueError("Sorry only empirical and kde methods are available for estimation of distribution shape")            
            
        
        return P,Q

    def compute_univariate_hellinger_divergence_dynamics(self,x,y,Type='empirical',numbins=20,kernel='gaussian',bandwidth=None,bandwidth_comparison=None):
        """For computing the univariate Hellinger divergence.
        
           Inputs: 
                  x - a univariate variable with n samples
                  y - a second univariate variable with n samples
                  Type - The estimation technique either empirical or kernel density estimation (kde)
                  numbins - The number of bins to estimate the distributions over
                  kernel - The type of kernel. See docs for allowed kernels
                  bandwidth - if Type is kde then the bandwidth for the x variable, if this is set to None 
                              then the bandwidth will be estimated using the rule of thumb
                  bandwidth_comparison - same as bandwidth but for the y variable
        
            Outputs:
                    Hellinger_distance - the estimated Hellinger divergence
                    
        """
        
        if len(x.shape)==1:
            x = x.reshape(-1,1)
        if len(y.shape)==1:
            y = y.reshape(-1,1)
        P,Q = self.generate_distributions_from_dynamics(x,y,Type,numbins,kernel,bandwidth,bandwidth_comparison)
         
        Hellinger_distance = (1/np.sqrt(2))*np.linalg.norm(np.sqrt(P)-np.sqrt(Q))
        return Hellinger_distance

    def compute_univariate_kullback_leibler_divergence_dynamics(self,x,y,Type='empirical',numbins=20,kernel='gaussian',bandwidth=None,bandwidth_comparison=None):
        """For computing univariate Kullback-Leibler divergence. 
        
           Inputs:
                  x - a univariate variable with n samples
                  y - a second univariate variable with n samples
                  Type - the type of estimator to use either empirical or kernel density estimation (kde)
                  numbins - the number of bins to use for the estimate
                  kernel - the kernel to use for estimation. See docs for a list of available kernels
                  bandwidth - if Type is kde the bandwidth to use for estimation with x variable. If set to None the rule
                              of thumb will be used.
                  bandwidth_comparison - same as bandwidth but for y variable.
        
         Outputs:
                 KL_divergence - the estimated Kullback-Leibler divergence
                  
        """
        
        if len(x.shape)==1:
            x = x.reshape(-1,1)
        if len(y.shape)==1:
            y = y.reshape(-1,1)
        P,Q = self.generate_distributions_from_dynamics(x,y,Type,numbins,kernel,bandwidth,bandwidth_comparison)
        Start = P*(np.log(P/Q))
        #Define 0*log(0) as 0...
        Start[np.where(np.isnan(Start))]=0
        Start[np.where(np.isinf(Start))]=0
        KL_divergence = np.sum(Start)
        return KL_divergence
    
    def compute_univariate_total_variation_divergence_dynamics(self,x,y,Type='empirical',numbins=20,kernel='gaussian',bandwidth=None,bandwidth_comparison=None):
        """For computing univariate total variation divergence. 
        
           Inputs:
                  x - a univariate variable with n samples
                  y - a second univariate variable with n samples
                  Type - the type of estimator to use either empirical or kernel density estimation (kde)
                  numbins - the number of bins to use for the estimate
                  kernel - the kernel to use for estimation. See docs for a list of available kernels
                  bandwidth - if Type is kde the bandwidth to use for estimation with x variable. If set to None the rule
                              of thumb will be used.
                  bandwidth_comparison - same as bandwidth but for y variable.
        
         Outputs:
                 TV_divergence - the estimated total variation divergence
                  
        """
        if len(x.shape)==1:
            x = x.reshape(-1,1)
        if len(y.shape)==1:
            y = y.reshape(-1,1)
        P,Q = self.generate_distributions_from_dynamics(x,y,Type,numbins,kernel,bandwidth,bandwidth_comparison)
        TV_divergence = 0.5*np.linalg.norm(P-Q,1)
        return TV_divergence        

    def generate_multivariate_distribution(self,x,Type='empirical',numbins=20,kernel='gaussian',bandwidth=None):
        """For estimating the multivariate distribution.
           
           Inputs: 
                  x - a (n x d) matrix, where n is the number of samples and d is the dimension
                  Type - the type of estimate.
                  numbins - the number of bins to be used for the esimation
                  kernel - if Type is kde, the type of kernel to be used. See docs for available kernels
                  bandwidth - the bandwidth to use for estimation. If bandwidth is none
            
            Outputs:
                    P - the estimated probability for each bin
        
           """
        
        if len(x.shape)==1:
            x = x.reshape(-1,1)        
        if Type == 'empirical':
            min_bin = np.min(x)
            max_bin = np.max(x)
            flr = np.floor(min_bin)
            ceil = np.ceil(max_bin)  
            bins = np.linspace(flr,ceil,numbins+1)
            Digital = np.digitize(x,bins)
            Urows,Counts = np.unique(Digital,return_counts=True,axis=0)
            P = Counts/x.shape[0]
            return P
        
           

    def compute_mi_binning(self,x,y,Type='empirical',numbins=20,kernel='gaussian',bandwidth=None,bandwidth_comparison=None,bandwidth_xy=None):
        """Compute mutual information by various types of binning (we will use kde in a 'binning' fashion, though it does not fit the name binning...)
        
            Inputs:
                   x - the first variable to compute the mutual information of (may have more than 1 dimension)
                   y - the second variable to compute the mutual information of (may have more than 1 dimension)
                   Type - the type of mutual information estimator to use, 'empirical' will be traditional binning, 'kde' will estimate the distribution at a certain number of bin edges using kde
                   numbins - the number of bins to use (both for regular binning and kde)
                   kernel - if Type == 'kde' then this is the kernel to use. see docs for available kernels
                   bandwidth - if Type is 'kde' then this is the bandwidth to use for the first variable (x), if None the rule of thumb will be used
                   bandwidth_comparison - same as bandwidth but for the second variable (y)
                   bandwidth_xy - same as bandwidth but for the joint variable (xy)
            
            Outputs:
                     MI - the mutual information as estimated by the method.
                   
        
        """
        if len(x.shape)==1:
            x = x.reshape(-1,1)
        if len(y.shape)==1:
            y = y.reshape(-1,1) 
        
        xy = np.concatenate((x,y),axis=1)
        Px = self.generate_multivariate_distribution(x,Type,numbins,kernel,bandwidth)
        Py = self.generate_multivariate_distribution(y,Type,numbins,kernel,bandwidth_comparison)
        Pxy = self.generate_multivariate_distribution(xy,Type,numbins,kernel,bandwidth_xy)
        Hx = -np.sum(Px*np.log(Px))
        Hy = -np.sum(Py*np.log(Py))
        Hxy = -np.sum(Pxy*np.log(Pxy))
        return Hx+Hy-Hxy
    
    def compute_cmi_binning(self,x,y,z=None,Type='empirical',numbins=20,kernel='gaussian',bandwidth=None,bandwidth_comparison=None,bandwidth_xy=None):
        """For estimating the conditional mutual information using binning.

            Inputs: 
                   x - the first (possibly multidimensional) variable
                   y - the second variable
                   z - the variable to condition on, if None then the mutual information between x and y is returned
                   Type - the type of mutual information estimator to use, 'empirical' will be traditional binning, 'kde' will estimate the distribution at a certain number of bin edges using kde
                   numbins - the number of bins to use (both for regular binning and kde)
                   kernel - if Type == 'kde' then this is the kernel to use. see docs for available kernels
                   bandwidth - if Type is 'kde' then this is the bandwidth to use for the first variable (x), if None the rule of thumb will be used
                   bandwidth_comparison - same as bandwidth but for the second variable (y)
                   bandwidth_xy - same as bandwidth but for the joint variable (xy)

            Outputs: 
                    CMI - the estimated conditional mutual information                    
        """
        
        if z is None:
            return self.compute_mi_binning(x,y,Type='empirical',numbins=20,kernel='gaussian',bandwidth=None,bandwidth_comparison=None,bandwidth_xy=None)
        
        else:
            if len(x.shape)==1:
                x = x.reshape(-1,1)
            if len(y.shape)==1:
                y = y.reshape(-1,1)
            
            if len(z.shape)==1:
                z = z.reshape(-1,1)
            
            xz = np.concatenate((x,z),axis=1)
            yz = np.concatenate((y,z),axis=1)
            xyz = np.concatenate((x,y,z),axis=1)
            Pz = self.generate_multivariate_distribution(x,Type,numbins,kernel,bandwidth)
            Pxz = self.generate_multivariate_distribution(xz,Type,numbins,kernel,bandwidth)
            Pyz = self.generate_multivariate_distribution(yz,Type,numbins,kernel,bandwidth)
            Pxyz = self.generate_multivariate_distribution(xyz,Type,numbins,kernel,bandwidth)
            Hz = -np.sum(Pz*np.log(Pz))
            Hxz = -np.sum(Pxz*np.log(Pxz))
            Hyz = -np.sum(Pyz*np.log(Pyz))
            Hxyz = -np.sum(Pxyz*np.log(Pxyz))
            return Hxz+Hyz-Hz-Hxyz

    def compute_histogram_mi(self,X,Y,numdecimals=0):
        """This is done using our 'proprietary method' of rounding for the binning.
            
            Inputs: 
                   X - the first (possibly multidimensional) variable
                   Y - the second (possibly multidimensional) variable
                   numdecimals - the number of decimal places to round to, if this is negative then it will round to 10's or 100's (or so on ) places....
            
            Outputs:
                    MI - the estimated mutual information
        """
        if len(X.shape)==1:
            X = X.reshape(-1,1)
        if len(Y.shape)==1:
            Y = Y.reshape(-1,1)
        
        X = np.round(X,numdecimals)
        Y = np.round(Y,numdecimals)
        XY = np.concatenate((X,Y),axis=1)
        UrowsX, CountsX = np.unique(X,return_counts=True,axis=0)
        UrowsY, CountsY = np.unique(Y,return_counts=True,axis=0)
        PX = CountsX/np.sum(CountsX)
        PY = CountsY/np.sum(CountsY)
        Urows, Counts = np.unique(XY,return_counts=True,axis=0)
        PXY = Counts/np.sum(Counts)
        return -np.sum(PX*np.log(PX))-np.sum(PY*np.log(PY))+np.sum(PXY*np.log(PXY))

    def compute_cmi_histogram(self,X,Y,Z=None,numdecimals=0):
        """Using decimal rounding binning to estimate conditional mutual information.
        
            Inputs:
                   X - the first (possibly multidimensional) variable
                   Y - the second (possibly multidimensional) variable
                   Z - the conditing (possibly multidimensional) variable, if None the mutual information between X and Y is returned
                   numdecimals - the number of decimal places to round to for binning. If negative it will round to 10's or 100's or so on.
        
            Outputs:
                    CMI - the estimated conditional mutual information
                   
        """
        if len(X.shape)==1:
            X = X.reshape(-1,1)
        if len(Y.shape)==1:
            Y = Y.reshape(-1,1)         
        if Z is None:
            return self.compute_histogram_mi(X,Y,numdecimals)
        
        else:
            if len(Z.shape)==1:
                Z = Z.reshape(-1,1)
            X = np.round(X,numdecimals)
            Y = np.round(Y,numdecimals)
            Z = np.round(Z,numdecimals)
            XYZ = np.concatenate((X,Y,Z),axis=1)
            XZ = np.concatenate((X,Z),axis=1)
            YZ = np.concatenate((Y,Z),axis=1)
            UrowsZ, CountsZ = np.unique(Z,return_counts=True,axis=0)
            UrowsXZ, CountsXZ = np.unique(XZ,return_counts=True,axis=0)
            UrowsYZ, CountsYZ = np.unique(YZ,return_counts=True,axis=0)
            UrowsXYZ, CountsXYZ = np.unique(XYZ,return_counts=True,axis=0)
            PZ = CountsZ/np.sum(CountsZ)
            PXZ = CountsXZ/np.sum(CountsXZ)
            PYZ = CountsYZ/np.sum(CountsYZ)
            PXYZ = CountsXYZ/np.sum(CountsXYZ)
            HZ = -np.sum(PZ*np.log(PZ))
            HXZ = -np.sum(PXZ*np.log(PXZ))
            HYZ = -np.sum(PYZ*np.log(PYZ))
            HXYZ = -np.sum(PXYZ*np.log(PXYZ))
            return HXZ+HYZ-HZ-HXYZ

    def estimate_network(self,X,method='histogram',shift=0,Type='empirical',numbins=20,kernel='gaussian',bandwidth=None,bandwidth_comparison=None,bandwidth_xy=None):
        n = X.shape[0]
        A = np.zeros((n,n))
        for i in range(n):
            x = X[shift:-1,i]
        if method == 'histogram':
            for i in range(n):
                x = X[shift:-1,i]
                
            
        
        elif method == 'binning':
            call_binning  = 1
        
        else:
            raise ValueError("You have chosen an invalid method, see docs for available methods")
            
    
class master_stability:

    def __init__(self):
        self.init_state = 0
        
    def rossler(self,x,t,LH,params):
        a,b,c = params
        lenx = len(x)
        dx = deepcopy(x)
        X = x[0:lenx:3]
        Y = x[1:lenx:3]
        Z = x[2:lenx:3]
        dx[0:lenx:3] = -Y-Z
        dx[1:lenx:3] = X+a*Y
        dx[2:lenx:3] = b+Z*(X-c)
        return np.array(dx-np.dot(LH,x)).flatten()
        
    def lorenz(self,x,t,LH,params):
        a,b,c = params
        lenx = len(x)
        dx = deepcopy(x)
        X = x[0:lenx:3]
        Y = x[1:lenx:3]
        Z = x[2:lenx:3]
        dx[0:lenx:3] = a*(Y-X)
        dx[1:lenx:3] = X*(b-Z)-Y
        dx[2:lenx:3] = X*Y-c*Z
        return np.array(dx-np.dot(LH,x)).flatten()

    def brusselator(self,x,t,LH,params):
        a,b = params
        lenx = len(x)
        dx = deepcopy(x)
        X = x[0:lenx:2]
        Y = x[1:lenx:2]
        dx[0:lenx:2] = 1-(a+1)*X+b*Y*X**2
        dx[1:lenx:2] = a*X-b*Y*X**2
        
        return np.array(dx-np.dot(LH,x)).flatten()            

    def vanderpol(self,x,t,LH,params):
        a = params[0]
        lenx = len(x)
        dx = deepcopy(x)
        X = x[0:lenx:2]
        Y = x[1:lenx:2]
        dx[0:lenx:2] = Y
        dx[1:lenx:2] = -X+a*(1-X**2)*Y
        
        return np.array(dx-np.dot(LH,x)).flatten()     

    def wienbridge(self,x,t,LH,params):
        a,b,c = params[0]
        lenx = len(x)
        dx = deepcopy(x)
        X = x[0:lenx:2]
        Y = x[1:lenx:2]
        dx[0:lenx:2] = -X+Y-(a*Y-b*Y**3+c*Y**5)
        dx[1:lenx:2] = -(-X+Y-(a*Y-b*Y**3+c*Y**5))-Y
        
        return np.array(dx-np.dot(LH,x)).flatten()      




class laplacian_pseudospectra:
    
    def __init__(self):
        
        self.NumGridPoints = 100
        self.Matrix = None
        self.ReturnDict = {}
        self.Contour = None

    
    def set_num_grid_points(self,NumSplits):
        self.NumGridPoints = NumSplits
    
    def set_matrix(self,Matrix):
        self.Matrix = Matrix
    
    def set_contour(self,Contour):
        self.Contour = Contour
    
    def return_num_grid_points(self):
        return self.NumGridPoints
    
    def return_matrix(self):
        return self.Matrix
    
    def return_contour(self):
        return self.Contour
     
    
        
    def pseudospectra(self,Matrix=None,NumGridPoints = None,Grid=None,MaxGrid=None,MaxGridMultiplier=None):
        """Grid must be a dictionary containing a matrix X and corresponding matrix Y to calculate the pseudospectra over
        entries of the dictionary must be labeled 'X' and 'Y', notice the capitalization! Also X and Y must be the appropriate shapes!
        
        Returns a dictionary containing X,Y and Z to be plotted as contours"""
        if Matrix is not None: 
            self.Matrix = Matrix
            if self.Matrix is not None:
                print("A matrix is already stored, resetting to matrix you passed!")
        
        else:
            if self.Matrix is None:
                raise ValueError("There must be a matrix set in order to find it's pseudospectra!")
        
        if NumGridPoints is not None:
            self.NumGridPoints = NumGridPoints
        if Grid is None:
            MaxEig = np.max(np.abs(np.linalg.eigvals(self.Matrix)))
            if MaxGrid is None:
                if MaxGridMultiplier is None:
                    M = 1.25*MaxEig
                else:
                    M = MaxGridMultiplier*MaxEig
            
            else:
                M = MaxGrid
            print("This is M: ",M)
            Linspace = np.linspace(-M,M,self.NumGridPoints)
            X,Y = np.meshgrid(Linspace,Linspace,indexing='ij')
            OriginalShapeX = X.shape
            X = X.flatten()
            Y = Y.flatten()
            
        else:
            X = Grid['X']
            Y = Grid['Y']
            OriginalShapeX = X.shape         
            X = X.flatten()
            Y = Y.flatten()
            if len(X)!=len(Y):
                raise ValueError("X and Y are not the same shape!")
        
        
        Z = np.zeros(len(X))
        I = np.identity(self.Matrix.shape[0])
        for i in range(len(X)):
            z = complex(X[i],Y[i])
            U,Sigma,V = np.linalg.svd(z*I-self.Matrix)
            Z[i] = Sigma[-1]
        
        
        X = X.reshape(OriginalShapeX)
        Y = Y.reshape(OriginalShapeX)
        Z = Z.reshape(OriginalShapeX)
        self.ReturnDict = {}
        self.ReturnDict['X'] = X
        self.ReturnDict['Y'] = Y
        self.ReturnDict['Z'] = Z
        
        return self.ReturnDict
          
    def plot_pseudospectra(self,Matrix=None,NumGridPoints=None,Grid=None,MaxGridMultiplier=None,Contour=None,color='purple',return_contour_data = True,close_when_done=False,axis=None,fig=None):
        if axis is None:
            fig,ax = plt.subplots()
        if axis is not None and fig is not None:
            ax = axis
            
        if axis is not None and fig is None:
            raise ValueError("If axis is defined, fig must be defined as well (fig cannot be None in this case!)")
        if NumGridPoints is not None:
            self.NumGridPoints=NumGridPoints       
        
        self.pseudospectra(Matrix,NumGridPoints,Grid,MaxGridMultiplier)
        if Contour is not None or self.Contour is not None:
            if Contour is not None:
                self.Contour = Contour
            cs = ax.contour(self.ReturnDict['X'],self.ReturnDict['Y'],self.ReturnDict['Z'],levels=self.Contour,colors=color)
        else:
            cs = ax.contour(self.ReturnDict['X'],self.ReturnDict['Y'],self.ReturnDict['Z'],colors=color)        
                        
        
        if return_contour_data == True:
            return cs,fig,ax
        
        if close_when_done:
            plt.show()
            plt.close()
        
    
    def laplacian_pseudo(self,Matrix=None,NumGridPoints=None,Grid=None,MaxGridMultiplier=None,Tol=1e-15):
        if Matrix is not None: 
            self.Matrix = Matrix
            if self.Matrix is not None:
                print("A matrix is already stored, resetting to matrix you passed!")
        
        else:
            if self.Matrix is None:
                raise ValueError("Need a matrix to generate Laplacian Pseudospectra!")
            
        
        T,Z = schur(self.Matrix,output='complex')
        Entries = np.arange(0,T.shape[0])
        DiagT = np.diag(T)
        Wh = np.where(np.abs(DiagT)<=10**-15)[0]
        Diff = np.setdiff1d(Entries,Wh)
        T = T[np.matrix(Diff).T,Diff]
        T[np.abs(T)<Tol] = 0
        self.pseudospectra(T,NumGridPoints,Grid,MaxGridMultiplier=MaxGridMultiplier)
        
        return self.ReturnDict
            
    def plot_lap_pseudo(self,Matrix=None,NumGridPoints=None,Grid=None,MaxGridMultiplier=None,Contour=None,color='purple',return_contour_data = True,close_when_done=False,axis=None,fig=None,Tol=1e-15):
        
        if axis is None:
            fig,ax = plt.subplots()
        if axis is not None and fig is not None:
            ax = axis
            
        if axis is not None and fig is None:
            raise ValueError("If axis is defined, fig must be defined as well (fig cannot be None in this case!)")
        if NumGridPoints is not None:
            self.NumGridPoints=NumGridPoints
        self.laplacian_pseudo(Matrix,self.NumGridPoints,Grid,MaxGridMultiplier=MaxGridMultiplier,Tol=Tol)
        if Contour is not None or self.Contour is not None:
            if Contour is not None:
                self.Contour = Contour
            cs = ax.contour(self.ReturnDict['X'],self.ReturnDict['Y'],self.ReturnDict['Z'],levels=self.Contour,colors=color)
        else:
            cs = ax.contour(self.ReturnDict['X'],self.ReturnDict['Y'],self.ReturnDict['Z'],colors=color)        
                        
        
        if return_contour_data == True:
            return cs,fig,ax
        
        if close_when_done:
            plt.show()
            plt.close()
        
        
    def estimate_kreiss_constant(self,NumGridPoints=None,maxiter=100,MaxGridMultiplier=None,SearchRegionMax=10,tol=1e-5):
        Iter = 0
        if NumGridPoints is not None:
            self.NumGridPoints = NumGridPoints
        
        Epsilon = 1/self.NumGridPoints
        Region = np.array([Epsilon,SearchRegionMax])
        Epsilon = np.mean(Region)
        self.Contour = [Epsilon]
        cs,fig,ax = self.plot_lap_pseudo(MaxGridMultiplier=MaxGridMultiplier)
        
        for item in cs.collections:
            for i in item.get_paths():
                v = i.vertices
                x = v[:,0]
        plt.close('all')        
        Estimate = -np.min(x)/Epsilon
        Region2 = deepcopy(Region)
        Region2[0] = Epsilon
        Epsilon2 = np.mean(Region2)
        self.Contour = [Epsilon2]
        cs,fig,ax = self.plot_lap_pseudo(MaxGridMultiplier=MaxGridMultiplier)
        for item in cs.collections:
            for i in item.get_paths():
                v = i.vertices
                x = v[:,0]
        plt.close('all')        
        Estimate2 = -np.min(x)/Epsilon2  
        if Estimate2>Estimate:
            Region = deepcopy(Region2)
            Region[0] = Epsilon2
            Estimate = Estimate2
        else:
            Region[1] = Epsilon2
        
        Diff = np.abs(np.array([Epsilon-Epsilon2]))
        while Iter<maxiter and Diff>tol:
            Iter = Iter+1
            Epsilon = np.mean(Region)
            print(Iter,Epsilon,Estimate)
            self.Contour = [Epsilon]
            cs,fig,ax = self.plot_lap_pseudo(MaxGridMultiplier=MaxGridMultiplier)
            plt.close('all')
            for item in cs.collections:
                for i in item.get_paths():
                    v = i.vertices
                    x = v[:,0]
            plt.close('all')        
            Estimate2 = -np.min(x)/Epsilon
            if Estimate2>Estimate:
                Region[0] = Epsilon
                Estimate = Estimate2
            
            else:
                Region[1] = Epsilon
            
            if Epsilon < 1/self.NumGridPoints:
                break
            
        
        return Estimate

    def plot_two_contours(self,Matrix,Matrix2,NumGridPoints=100,MaxGridMultiplier=None,Grid=None,Contour=None,Vlinewidth=1,contour1color='red',contour2color='green',legendentries=['$L_1$','$L_2$','Stability Cutoff'],nolegend=True,Type='lap_pseudo',Tol=1e-15):
        if Type=='lap_pseudo':
            legend_list = []      
            cs,fig,ax = self.plot_lap_pseudo(Matrix=Matrix,NumGridPoints=NumGridPoints,Grid=Grid,MaxGridMultiplier=MaxGridMultiplier,Contour=Contour,color=contour1color,return_contour_data=True,close_when_done=False,Tol=Tol)
            legend_list.append(cs.collections[0])
            cs2,fig2,ax2 = self.plot_lap_pseudo(Matrix=Matrix2,NumGridPoints=NumGridPoints,Grid=Grid,MaxGridMultiplier=MaxGridMultiplier,Contour=Contour,color=contour2color,return_contour_data=True,close_when_done=False,axis=ax,fig=fig,Tol=Tol) 
            legend_list.append(cs2.collections[0])
            plt.axvline(x=0,linewidth=Vlinewidth)#,label=legendentries[2])
            if not nolegend:
                plt.legend(legend_list,legendentries[0:2])
            plt.title('$\epsilon = {}$'.format(self.Contour[0]))
            plt.xlabel("$Re(\Phi_{\epsilon}($" +legendentries[0]+'),' + '$\Phi_{\epsilon}($'+legendentries[1] + '))')
            plt.ylabel("$Im(\Phi_{\epsilon}($" +legendentries[0] +'),' + '$\Phi_{\epsilon}($'+legendentries[1] + '))')
            
        elif Type=='pseudo':
            legend_list = []      
            cs,fig,ax = self.plot_pseudospectra(Matrix=Matrix,NumGridPoints=NumGridPoints,Grid=Grid,MaxGridMultiplier=MaxGridMultiplier,Contour=Contour,color=contour1color,return_contour_data=True,close_when_done=False)
            legend_list.append(cs.collections[0])
            cs2,fig2,ax2 = self.plot_lap_pseudo(Matrix=Matrix2,NumGridPoints=NumGridPoints,Grid=Grid,MaxGridMultiplier=MaxGridMultiplier,Contour=Contour,color=contour2color,return_contour_data=True,close_when_done=False,axis=ax,fig=fig) 
            legend_list.append(cs2.collections[0])
            plt.axvline(x=0,linewidth=Vlinewidth)#,label=legendentries[2])
            if not nolegend:
                plt.legend(legend_list,legendentries[0:2])
            plt.title('$\epsilon = {}$'.format(self.Contour[0]))
            plt.xlabel("$Re(\Phi_{\epsilon}($" +legendentries[0]+'),' + '$\Phi_{\epsilon}($'+legendentries[1] + '))')
            plt.ylabel("$Im(\Phi_{\epsilon}($" +legendentries[0] +'),' + '$\Phi_{\epsilon}($'+legendentries[1] + '))')        
            #ax = plt.gca()
            #ax.legend(legendentries)
        
        
    def nilpotent_exp_norms(self,Matrix,Matrix2=None,t=0.05):
        Dict = {}
        if Matrix2 is None:
            T,Z = schur(Matrix,output='complex')
            n = T.shape[0]
            D = np.diag(np.diagonal(Matrix))
            N = T-D
            Dict['Matrix_N'] =np.linalg.norm(expm(N*t)-np.identity(n))
            Dict['Matrix_D'] = np.linalg.norm(expm(D*t))
            
        else:
            T,Z = schur(Matrix,output='complex')
            n = T.shape[0]
            D = np.diag(np.diagonal(T))
            N = T-D 
            Dict['Matrix_N'] =np.linalg.norm(expm(N*t)-np.identity(n))
            Dict['Matrix_D'] = np.linalg.norm(expm(D*t))            
            T,Z = schur(Matrix2,output='complex')
            D = np.diag(np.diagonal(T))
            N = T-D
            Dict['Matrix2_N'] =np.linalg.norm(expm(N*t)-np.identity(n))
            Dict['Matrix2_D'] = np.linalg.norm(expm(D*t))            
        
        return Dict
    
    def generate_induced_laplacian(self,Matrix=None):
        if Matrix is not None: 
            self.Matrix = Matrix
            if self.Matrix is not None:
                print("A matrix is already stored, resetting to matrix you passed!")
        
        else:
            if self.Matrix is None:
                raise ValueError("Need a matrix to generate Laplacian Pseudospectra!")
            
        
        T,Z = schur(self.Matrix,output='complex')
        D = np.diag(np.diagonal(T))
        Lhat = np.dot(np.dot(Z,D),Z.T)
        return Lhat