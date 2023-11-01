# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 13:06:30 2023

@author: jefis
"""
import numpy as np
import networkx as nx
from copy import deepcopy
from scipy.integrate import odeint

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
        a,b,c = params[0]
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
            x0 = np.random.beta(init_cond_params[0],init_cond_params[1],lenx0)+init_cond_offset

        elif init_cond_type == 'zipf':
            x0 = np.random.beta(init_cond_params[0],init_cond_params[1],lenx0)+init_cond_offset
        
        else:
            raise ValueError("This init_cond_type is not implemented see docs for types which are currently implemented")                    
        sol = odeint(self.linear_dynamics,x0,t,args=(-L,))
        return sol,t
    
    def continuous_time_nonlinear_dynamics(self,G=None,tmax=100,timestep=0.1,init_cond_type='normal',init_cond=None,init_cond_params=[0,1],init_cond_offset=0,dynamics_type='Rossler',dynamics_params=[0.2,0.2,7],coupling_matrix=None,coupling_strength=1):
        """Generate continuous time Laplacian (i.e. diffusive) type non-linear dynamics
        Inputs: 
               G - a networkx graph, if set to None then the internally stored graph will be used
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
        Outputs: 
                sol - the solution after integration
                t   - the vector of integration time points
               
        """        
        if G is None:
            if self.graph is not None:
                G = deepcopy(self.graph)
            
            else:
                raise ValueError("No graph to perform dynamics on...")
        
        n = len(G)
        t = np.arange(0,tmax,step=timestep)
        L = self.convert_graph_to_laplacian(G)

        
        if dynamics_type =='Rossler':
            lenx0 = 3*n
            if init_cond is None:
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
                    
            if coupling_matrix is None:
                #assume coupling only through the x component
                H = np.eye(3)
                H[1,1] = 0
                H[2,2] = 0
                
            else:
                H = coupling_matrix
            
            LH = coupling_strength*np.kron(L,H)
            sol = odeint(self.rossler,x0,t,args=(LH,dynamics_params)) 
            
        elif dynamics_type=='Lorenz':
            lenx0 = 3*n
            if init_cond is None:
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
                    
            if coupling_matrix is None:
                #assume coupling only through the x component
                H = np.eye(3)
                H[1,1] = 0
                H[2,2] = 0
                
            else:
                H = coupling_matrix
            
            LH = coupling_strength*np.kron(L,H)
            sol = odeint(self.lorenz,x0,t,args=(LH,dynamics_params))   
            
        elif dynamics_type=='VanDerPol':
            lenx0 = 2*n
            if init_cond is None:
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
                    
            if coupling_matrix is None:
                #assume coupling only through the x component
                H = np.eye(3)
                H[1,1] = 0
                H[2,2] = 0
                
            else:
                H = coupling_matrix
            
            LH = coupling_strength*np.kron(L,H)
            sol = odeint(self.vanderpol,x0,t,args=(LH,dynamics_params)) 
            
        elif dynamics_type=='Wienbridge':
            lenx0 = 2*n
            if init_cond is None:
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
                    
            if coupling_matrix is None:
                #assume coupling only through the x component
                H = np.eye(3)
                H[1,1] = 0
                H[2,2] = 0
                
            else:
                H = coupling_matrix
            
            LH = coupling_strength*np.kron(L,H)
            sol = odeint(self.wienbridge,x0,t,args=(LH,dynamics_params))            

        elif dynamics_type=='Brusselator':
            lenx0 = 2*n
            if init_cond is None:
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
                    
            if coupling_matrix is None:
                #assume coupling only through the x component
                H = np.eye(3)
                H[1,1] = 0
                H[2,2] = 0
                
            else:
                H = coupling_matrix
            
            LH = coupling_strength*np.kron(L,H)
            sol = odeint(self.brusselator,x0,t,args=(LH,dynamics_params))            
 

        
        else:
            raise ValueError("Dynamics must be of allowed type, see docs for allowed types")
        
        return sol,t
    
    
    
    
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
