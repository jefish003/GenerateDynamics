# GenerateDynamics
A package for generating some common dynamics on networks, as well as some types of analysis.

Getting started: Packages you will need: numpy, networkx, copy, scipy

Introduction to the methods:
A main class is the laplacian_dynamics class. This can be used for diffusive coupling on top of the network. To get started we will first import the package and initialize laplacian_dynamics

```
from GenerateDynamics import laplacian_dynamics
ld = laplacian_dynamics()
```

Now we have a couple of options for getting a graph initialized, one we can explicitely set the graph using set_graph, or the convert_adjacency options
```
#G is a networkx Graph or DiGraph
ld.set_graph(G)
```
or
```
#A is either a 2d numpy array or a 2d matrix
ld.convert_adjacency(A)
```
These methods will internally store the graph that has been set.
Alternatively, we can set the graph during a call to generate the dynamics, for instance
```
#G is a networkx Graph or DiGraph
x,t = ld.continuous_time_linear_dynamics(G=G)
```
will generate continuous time linear consensus dynamics, using the graph Laplacian (i.e dx/dt = -Lx). NOTE: if a graph has been stored, but the option above is used (G=G), then the internally stored graph will be overwritten by this new graph G.

The options for continuous_time_linear_dynamics are as follows:

G - the graph you want to run dynamics on top of (default None, this is assumed to have already been set by one of the methods above)

tmax - (default tmax = 100) the time to integrate until (starting time is always assumed to be t = 0)

timestep - (default timestep = 0.1) the timestep for integration

init_cond_type - (default 'normal') the distribution to draw the initial condition from. See below for a list of options

init_cond - (default None) if you wish to use your own initial condition you can specify it here, and init_cond_type and init_cond_params will then be ignored

init_cond_params - (default [0,1]) an iterable containing the parameters for the distribution in init_cond_type. (NOTE: MUST BE AN ITERABLE!!!!)

init_cond_offset - (default 0) a constant offset to be added to each value of the initial condition, this is a way to create a shifted distribution if desired


The options for init_cond_type are as follows:

'normal' - the two parameter normal distribution. The first parameter is the mean and the second is the standard deviation

'uniform' - the two parameter uniform distribution. The first parameter is the minimum possible value and the second is the maximum

'laplace' - the two parameter Laplace distribution. The first parameter is the mean, the second parameter is a scale paramter, sometimes referred to as the "diversity"

'exponential' - the one parameter exponential distribution.  The parameter is the mean of the distribution.

'rayleigh' - the one parameter Rayleigh distribution. This is the scale parameter of the distribution

'beta' - the two parameter beta distribution. The two parameters are shape parameters for the distribution.

'gamma' - the two parameter gamma distribution. The first parameter is the shape parameter and the second is the scale parameter

'gumbel' - the two parameter Gumbel distribution. The first parameter is the location parameter, the second is the scale parameter.

'chisquare' - the one parameter Chi^2 distribution. The parameter must be an integer, and is the mean of the distribution

'logistic' - the two parameter logistic distribution. The first parameter is the mean, the second parameter is the scale parameter.

'lognormal' - the two parameter lognormal distribution. The first parameter is the logarithm of the mean, and the second is the logarithm of the standard deviation.

'pareto' - the one parameter pareto distribution. The one parameter is the shape parameter.

'f' - the two parameter f distribution. The first parameter is a pos. integer, the degrees of freedom of the numerator. The second parameter degrees of freedom of the denominator.

'vonmises' - the two parameter Von Mises distribution. The first parameter is the mean (wrapped onto the circle...) and the second parameter is the measure of concentration on the circle.

'wald' - the two parameter Wald distribution (aka inverse Gaussian). The first parameter is the mean, the second parameter is the shape parameter.

'weibull' - the one parameter Weibull distribution. The parameter is the shape parameter.

