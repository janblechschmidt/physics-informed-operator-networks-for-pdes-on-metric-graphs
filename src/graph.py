import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import jax.numpy as jnp


class Graph(object):

    def __init__(self):

        self.initialized = False

        # Default setting of time-dependent flux
        self.time_dependent_flux = False

    def buildGraph(self):

        # Define networkx multigraph
        self.G = nx.MultiDiGraph(self.A)

        # Get number of vertices
        self.n_v = self.A.shape[0]

        # Determine lists of edges and lengths
        self._determineEdgeList()

        # Determine list of vertices and incoming as well as outgoing edges
        self._determineVertexList()

        # Determine graph layout if necessary
        if not hasattr(self, 'pos'):
            self.pos = nx.kamada_kawai_layout(self.G)
        else:
            if isinstance(self.pos, np.ndarray):
                self.pos = self.pos_array_to_dict(self.pos)
            elif not isinstance(self.pos, dict):
                raise ValueError('Check pos argument.')
        # Determine bounding boxes
        self.bbox = {"min": np.inf * np.ones((2,)),
                     "max": -np.inf * np.ones((2,))}
        for i, p in self.pos.items():
            self.bbox["min"] = np.min([p, self.bbox["min"]], axis=0)
            self.bbox["max"] = np.max([p, self.bbox["max"]], axis=0)

        self.initialized = True

    def _determineEdgeList(self):
        """Determine edge matrix and weight vector.
        This could also be accomplished by a loop over `G.edges`:
            for e in G.edges:
                print(e)
        """

        self.E = []
        self.W = []

        for i in range(self.n_v):
            for j in range(self.n_v): # previously from i+1
                aij = self.A[i, j]
                if aij > 0:
                    self.E.append([i, j])
                    self.W.append(aij)

        # Get number of edges
        self.ne = len(self.E)

    def _determineVertexList(self):

        self.Vin = [[] for _ in range(self.n_v)]
        self.Vout = [[] for _ in range(self.n_v)]

        self.inflowNodes = []
        self.outflowNodes = []

        for i, e in enumerate(self.E):
            # Unpack edge
            vin, vout = e
            self.Vin[vout].append(i)
            self.Vout[vin].append(i)

        for i in range(self.n_v):
            if self.Vin[i] and (not self.Vout[i]):
                self.outflowNodes.append(i)

            if (not self.Vin[i]) and self.Vout[i]:
                self.inflowNodes.append(i)
        self.outflowNodes = np.array(self.outflowNodes)
        self.inflowNodes = np.array(self.inflowNodes)

        self.dirichletNodes = np.union1d(self.outflowNodes, self.inflowNodes)
        self.innerVertices = np.setdiff1d(np.arange(self.n_v), self.dirichletNodes)

        # WARNING: This only works for constant parameters
        #self.dirichletNodes = np.where(
        #    (self.dirichletAlpha != 0) | (self.dirichletBeta != 0))[0]
        #self.innerVertices = np.setdiff1d(
        #    np.arange(self.n_v), self.dirichletNodes)

    def plotGraph(self, **kwargs):

        if not hasattr(self, 'node_color'):
            self.node_color = [] 
            for i in range(self.n_v):
                if i in self.inflowNodes:
                    self.node_color.append('#207D4A')
                elif i in self.outflowNodes:
                    self.node_color.append('maroon')
                else:
                    self.node_color.append('#5d8aa8')
            #self.node_color = #np.zeros(self.n_v)
            #self.node_color[self.inflowNodes] = ''
            #self.node_color[self.outflowNodes] = -1
        if not hasattr(self, 'edge_color'):
            self.edge_color = np.zeros(self.ne)
            for i, e in enumerate(self.E):
                if e[0] in self.inflowNodes:
                    self.edge_color[i] = 1
                elif e[1] in self.outflowNodes:
                    self.edge_color[i] = -1
                    
        nx.draw(self.G,
                pos=self.pos,
                #with_labels=True,
                node_color=self.node_color,
                edge_color=self.edge_color,
                #cmap='coolwarm',
                #vmin=-1,
                #vmax=1,
                **kwargs)
        if 'ax' in kwargs:
            ax = kwargs.get('ax')
        else:
            ax = plt.gca()
        ax.set_title(self.title)

    def pos_array_to_dict(self, pos):
        pos_dict = dict()
        for i in range(pos.shape[0]):
            pos_dict[i] = pos[i, :]
        return pos_dict

    def f(self, u, i):
        return self.v[i] * u * (1 - u)

    def df(self, u, i):
        return self.v[i] * (1 - 2 * u)

    def pde(self, u, ut, ux, uxx, i):
        return ut - self.eps * uxx + self.df(u, i) * ux

    def pde_rhs(self, u, ux, uxx, i):
        return self.eps * uxx - self.df(u, i) * ux

    def flux(self, u, ux, i):
        return - self.eps * ux + self.f(u, i)

    def f_param(self, u, param):
        return param * u * (1 - u)

    def df_param(self, u, param):
        return param * (1 - 2 * u)

    def pde_param(self, u, ut, ux, uxx, param):
        return ut - self.eps * uxx + self.df_param(u, param) * ux

    def pde_rhs_param(self, u, ux, uxx, param):
        return self.eps * uxx - self.df_param(u, param) * ux

    def flux_param(self, u, ux, param):
        return - self.eps * ux + self.f_param(u, param)

    def initial_cond(self, x):
        return np.zeros_like(x)

    def initial_cond_jnp(self, x):
        return jnp.zeros_like(x)


class Example0(Graph):
    def __init__(self, eps=1e-2,
                 dirichletAlpha=np.array([0.7, 0.0]),
                 dirichletBeta=np.array([0.0, 0.8]),
                 initial_cond=None,
                 initial_cond_jnp=None):

        super().__init__()
        self.id = 0
        self.title = 'One edge'
        self.A = np.array([[0, 1], [0, 0]], dtype=np.int16)

        self.pos = np.array([[0, 0], [1, 0]])

        # Set boundaries
        tmin = 0.
        tmax = 1.
        xmin = 0.
        xmax = 1.

        # Default velocity
        self.v = np.array([1.0])

        # Lower bounds
        self.lb = np.array([tmin, xmin])

        # Upper bounds
        self.ub = np.array([tmax, xmax])

        self.dirichletAlpha = dirichletAlpha
        self.dirichletBeta = dirichletBeta

        self.eps = eps

        if initial_cond is not None:
            self.initial_cond = initial_cond
            
        if initial_cond_jnp is not None:
            self.initial_cond_jnp = initial_cond_jnp

        self.buildGraph()


class Example1(Graph):
    def __init__(self, eps=1e-2):

        super().__init__()
        self.id = 1
        self.title = 'Complex graph I'
        self.A = np.array([[0, 1, 0, 0, 0, 0, 0, 0],
                           [0, 0, 1, 0, 1, 0, 0, 0],
#                           [0, 0, 0, 1, 0, 0, 0, 0],
                           [0, 0, 0, 1, 1, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0, 0, 0, 1],
                           [0, 0, 0, 0, 0, 0, 0, 0]
                           ], dtype=np.int16)

        self.pos = np.array([[2.0, 2.0],
                             [1.5, 1.5],
                             [1., 1.5],
                             [0.5, 1.5],
                             [1.5, 1.],
                             [1, 0.5],
                             [.5, 0.5],
                             [0, 0]])
        
        # Set boundaries
        tmin = 0.
        tmax = 1.
        xmin = 0.
        xmax = 1
        

        # Lower bounds
        self.lb = np.array([tmin, xmin])

        # Upper bounds
        self.ub = np.array([tmax, xmax])

        self.dirichletAlpha = np.zeros(8)
        self.dirichletBeta = np.zeros(8)
        self.dirichletAlpha[0] = .8
        self.dirichletBeta[7] = .5

        self.eps = eps

        self.buildGraph()

        # Default velocity
        self.v = 1. * np.ones(shape=(self.ne,))


class Example2(Graph):
    def __init__(self, eps=1e-2,
                 dirichletAlpha=np.array([0.9, 0.3, 0., 0., 0., 0.]),
                 dirichletBeta=np.array([0.0, 0.0, 0., 0., 0.8, 0.1]),
                 initial_cond=None,
                 initial_cond_jnp=None
                ):

        super().__init__()
        self.id = 2
        self.title = 'Training graph II'
        self.A = np.array([[0, 0, 1, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0],
                           [0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 1, 1],
                           [0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0]], dtype=np.int16)

        self.pos = np.array([[0.0, 0.0],
                             [0.0, 1.0],
                             [0.5, 0.5],
                             [0.5 + np.sqrt(2) / 2, 0.5],
                             [1.0 + np.sqrt(2) / 2, 0.0],
                             [1.0 + np.sqrt(2) / 2, 1.0]])

        # Set boundaries
        tmin = 0.
        # tmax = 10.
        tmax = 1.
        xmin = 0.
        xmax = 1.

        # Lower bounds
        self.lb = np.array([tmin, xmin])

        # Upper bounds
        self.ub = np.array([tmax, xmax])

        self.dirichletAlpha = dirichletAlpha
        self.dirichletBeta = dirichletBeta

        if initial_cond is not None:
            self.initial_cond = initial_cond
            
        if initial_cond_jnp is not None:
            self.initial_cond_jnp = initial_cond_jnp

        self.eps = eps

        self.buildGraph()
        # Default velocity
        self.v = 1. * np.ones(shape=(self.ne,))


class Example3(Graph):
    def __init__(self, eps=1e-2,
                 dirichletAlpha=np.array([0.7, 0.0, 0.0]),
                 dirichletBeta=np.array([0.0, 0.0, 0.8]),
                 initial_cond=None,
                 initial_cond_jnp=None):

        super().__init__()
        self.id = 3
        self.title = 'Two edges'
        self.A = np.array([[0, 1, 0],
                           [0, 0, 1],
                           [0, 0, 0]], dtype=np.int16)

        self.pos = np.array([[0, 0], [1, 0], [2, 0]])

        # Set boundaries
        tmin = 0.
        tmax = 1.
        xmin = 0.
        xmax = 1.

        # Lower bounds
        self.lb = np.array([tmin, xmin])

        # Upper bounds
        self.ub = np.array([tmax, xmax])

        self.dirichletAlpha = dirichletAlpha
        self.dirichletBeta = dirichletBeta

        self.eps = eps

        if initial_cond is not None:
            self.initial_cond = initial_cond
            
        if initial_cond_jnp is not None:
            self.initial_cond_jnp = initial_cond_jnp
            
        self.buildGraph()

        # Default velocity
        # self.v = 1. * np.ones(shape=(self.ne,))
        self.v = np.array([1., 1.])

    def initial_cond(self, x):
        return 0.0 * np.ones_like(x)
        # return x * (1. - x)**2

    def initial_cond_jnp(self, x):
        return 0.0 * jnp.ones_like(x)
        # return x * (1. - x)**2

class Example4(Graph):
    def __init__(self, eps=1e-2,
                 dirichletAlpha=np.array([0.7, 0.0, 0.0, 0.0]),
                 dirichletBeta=np.array([0.0, 0.0, 0.0, 0.8]),
                 initial_cond=None,
                 initial_cond_jnp=None):

        super().__init__()
        self.id = 4
        self.title = 'Training graph I'
        self.A = np.array([[0, 1, 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1],
                           [0, 0, 0, 0]], dtype=np.int16)

        self.pos = np.array([[0, 0], [1, 0], [2, 0], [3, 0]])

        # Set boundaries
        tmin = 0.
        tmax = 1.
        xmin = 0.
        xmax = 1.

        # Lower bounds
        self.lb = np.array([tmin, xmin])

        # Upper bounds
        self.ub = np.array([tmax, xmax])

        self.dirichletAlpha = dirichletAlpha
        self.dirichletBeta = dirichletBeta

        self.eps = eps

        if initial_cond is not None:
            self.initial_cond = initial_cond
            
        if initial_cond_jnp is not None:
            self.initial_cond_jnp = initial_cond_jnp
            
        self.buildGraph()

        # Default velocity
        # self.v = 1. * np.ones(shape=(self.ne,))
        self.v = np.array([1., 1., 1.])

    def initial_cond(self, x):
        return 0.0 * np.ones_like(x)
        # return x * (1. - x)**2

    def initial_cond_jnp(self, x):
        return 0.0 * jnp.ones_like(x)
        # return x * (1. - x)**2

class Example5(Graph):
    def __init__(self, eps=1e-2,
                 n_edges = 3,
                 dirichletAlpha=np.array([0.0, 0.0, 0.0, 0.0]),
                 dirichletBeta=np.array([0.0, 0.0, 0.0, 0.0]),
                 initial_cond=None,
                 initial_cond_jnp=None):

        super().__init__()
        self.id = 5
        self.n_edges = n_edges
    
        self.title = f'{n_edges} edges in a line'
        self.A = np.diag(np.ones(n_edges, dtype=np.int16),1, )

        self.pos = np.hstack([np.array(range(n_edges + 1))[:,None], np.zeros((n_edges + 1, 1))])
        # Set boundaries
        tmin = 0.
        tmax = 1.
        xmin = 0.
        xmax = 1.

        # Lower bounds
        self.lb = np.array([tmin, xmin])

        # Upper bounds
        self.ub = np.array([tmax, xmax])

        self.dirichletAlpha = dirichletAlpha
        self.dirichletBeta = dirichletBeta

        self.eps = eps

        if initial_cond is not None:
            self.initial_cond = initial_cond
            
        if initial_cond_jnp is not None:
            self.initial_cond_jnp = initial_cond_jnp
            
        self.buildGraph()

        # Default velocity
        self.v = 1. * np.ones((n_edges,))

    def initial_cond(self, x):
        return 0.0 * np.ones_like(x)
        # return x * (1. - x)**2

    def initial_cond_jnp(self, x):
        return 0.0 * jnp.ones_like(x)
        # return x * (1. - x)**2


class Example6(Graph):
    def __init__(self, eps=1e-2,
                 dirichletAlpha=np.array([0.9, 0.0, 0., 0., 0., 0.]),
                 dirichletBeta=np.array([0.0, 0.0, 0., 0., 0.0, 0.1]),
                 initial_cond=None,
                 initial_cond_jnp=None
                ):

        super().__init__()
        self.id = 6
        self.title = 'Training graph III'
        self.A = np.array([[0, 1, 0, 0, 0, 0],
                           [0, 0, 1, 1, 0, 0],
                           [0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0, 1],
                           [0, 0, 0, 0, 0, 0]], dtype=np.int16)

        self.pos = np.array([[0.0, 0.0],
                             [0.5, 0.0],
                             [1., -0.5],
                             [1., .5],
                             [1.5, 0],
                             [2, 0]])

        # Set boundaries
        tmin = 0.
        # tmax = 10.
        tmax = 1.
        xmin = 0.
        xmax = 1.

        # Lower bounds
        self.lb = np.array([tmin, xmin])

        # Upper bounds
        self.ub = np.array([tmax, xmax])

        self.dirichletAlpha = dirichletAlpha
        self.dirichletBeta = dirichletBeta

        if initial_cond is not None:
            self.initial_cond = initial_cond
            
        if initial_cond_jnp is not None:
            self.initial_cond_jnp = initial_cond_jnp

        self.eps = eps

        self.buildGraph()
        # Default velocity
        self.v = 1. * np.ones(shape=(self.ne,))



class Example7(Graph):
    def __init__(self, eps=1e-1,
                 n = 5,
                 p = 1,
                 dirichletAlpha=None,
                 dirichletBeta=None,
                 initial_cond=None,
                 initial_cond_jnp=None
                ):

        super().__init__()
        self.title = 'Binomial graph'
        self.id = 7

        G = nx.binomial_graph(n=n, p=p, seed=0, directed=True)
        
        
        A = nx.adjacency_matrix(G,weight=None)
        A = A.todense()
        A[A!=0] = 1
        A = np.triu(A)
        
        m,n = A.shape
        B = np.zeros((m+2,n+2), dtype=np.int16)
        B[1:-1,:][:,1:-1] = A
        B[0,1] = 1
        B[-2,-1] = 1
        G=nx.from_numpy_array(B, create_using = nx.MultiDiGraph())
        self.A = B
        self.pos = nx.kamada_kawai_layout(G)


        # Set boundaries
        tmin = 0.
        tmax = 1.
        xmin = 0.
        xmax = 1.

        # Lower bounds
        self.lb = np.array([tmin, xmin])

        # Upper bounds
        self.ub = np.array([tmax, xmax])

        m = self.A.shape[1]

        if dirichletAlpha is not None:
            self.dirichletAlpha = dirichletAlpha
        else:
            self.dirichletAlpha = np.zeros((m,))
            self.dirichletAlpha[0] = 1.
        
        if dirichletBeta is not None:
            self.dirichletBeta = dirichletBeta
        else:
            self.dirichletBeta = np.zeros((m,))
            self.dirichletBeta[-1] = 1.
            
        if initial_cond is not None:
            self.initial_cond = initial_cond
            
        if initial_cond_jnp is not None:
            self.initial_cond_jnp = initial_cond_jnp

        self.eps = eps

        self.buildGraph()
        
        # Default velocity
        self.v = 1. * np.ones(shape=(self.ne,))
        
class Example7Ext(Graph):
    def __init__(self, eps=1e-1,
                 n = 5,
                 p = 1,
                 n_in = 1,
                 n_out = 1,
                 dirichletAlpha=None,
                 dirichletBeta=None,
                 initial_cond=None,
                 initial_cond_jnp=None
                ):

        super().__init__()
        self.title = 'Binomial graph'
        self.id = 7

        G = nx.binomial_graph(n=n, p=p, seed=0, directed=True)
        
        
        A = nx.adjacency_matrix(G,weight=None)
        A = A.todense()
        A[A!=0] = 1
        A = np.triu(A)
        
        m,n = A.shape
        B = np.zeros((m+n_in + n_out,n+n_in+n_out), dtype=np.int16)
        B[n_in:-n_out,:][:,n_in:-n_out] = A
        B[0, n_in] = 1
        np.random.seed(0)
        n_add = np.random.choice(np.arange(1,n-1),
                                 size=n_in+n_out-2,
                                 replace=False)
        idx = 0
        for i in range(n_in-1):
            B[1+i, n_in + n_add[idx]] = 1
            idx += 1
        for j in range(n_out-1):
            B[n_in + n_add[idx], -n_out+j] = 1 
            idx += 1 
        B[-1-n_out ,-1] = 1
        G=nx.from_numpy_array(B, create_using = nx.MultiDiGraph())
        self.A = B
        self.pos = nx.kamada_kawai_layout(G)


        # Set boundaries
        tmin = 0.
        tmax = 1.
        xmin = 0.
        xmax = 1.

        # Lower bounds
        self.lb = np.array([tmin, xmin])

        # Upper bounds
        self.ub = np.array([tmax, xmax])

        m = self.A.shape[1]

        if dirichletAlpha is not None:
            self.dirichletAlpha = dirichletAlpha
        else:
            self.dirichletAlpha = np.zeros((m,))
            self.dirichletAlpha[0] = 1.
        
        if dirichletBeta is not None:
            self.dirichletBeta = dirichletBeta
        else:
            self.dirichletBeta = np.zeros((m,))
            self.dirichletBeta[-1] = 1.
            
        if initial_cond is not None:
            self.initial_cond = initial_cond
            
        if initial_cond_jnp is not None:
            self.initial_cond_jnp = initial_cond_jnp

        self.eps = eps

        self.buildGraph()
        
        # Default velocity
        self.v = 1. * np.ones(shape=(self.ne,))

        
   
class Example7Ext2(Graph):
    def __init__(self, eps=1e-1,
                 n = 5,
                 p = 1,
                 n_in = 1,
                 n_out = 1,
                 dirichletAlpha=None,
                 dirichletBeta=None,
                 initial_cond=None,
                 initial_cond_jnp=None
                ):

        super().__init__()
        self.title = 'Binomial graph'
        self.id = 7

        G = nx.binomial_graph(n=n, p=p, seed=0, directed=True)


        A = nx.adjacency_matrix(G,weight=None)
        A = A.todense()
        A[A!=0] = 1
        A = np.triu(A)
        np.random.seed(0)
        # Remove edges randomly such that node i stays connected to node i+1 and 2 other ones
        for i, ai in enumerate(A):
            na = ai.sum()
            if na > 3:
                nidx = np.random.choice(np.arange(2, na + 1), size=(na- 3,), replace=False)
                A[i, i + nidx] = 0
        
        # Add additional inflow and outflow nodes
        m,n = A.shape
        B = np.zeros((m+n_in + n_out,n+n_in+n_out), dtype=np.int16)
        B[n_in:-n_out,:][:,n_in:-n_out] = A
        B[0, n_in] = 1
        np.random.seed(0)
        n_add = np.random.choice(np.arange(1,n-1),
                                 size=n_in + n_out - 2,
                                 replace=False)
        idx = 0
        for i in range(n_in-1):
            B[1+i, n_in + n_add[idx]] = 1
            idx += 1
        for j in range(n_out-1):
            B[n_in + n_add[idx], -n_out+j] = 1 
            idx += 1 
        B[-1-n_out ,-1] = 1
        G=nx.from_numpy_array(B, create_using = nx.MultiDiGraph())
        self.A = B
        self.pos = nx.kamada_kawai_layout(G)


        # Set boundaries
        tmin = 0.
        tmax = 1.
        xmin = 0.
        xmax = 1.

        # Lower bounds
        self.lb = np.array([tmin, xmin])

        # Upper bounds
        self.ub = np.array([tmax, xmax])

        m = self.A.shape[1]

        if dirichletAlpha is not None:
            self.dirichletAlpha = dirichletAlpha
        else:
            self.dirichletAlpha = np.zeros((m,))
            self.dirichletAlpha[0] = 1.
        
        if dirichletBeta is not None:
            self.dirichletBeta = dirichletBeta
        else:
            self.dirichletBeta = np.zeros((m,))
            self.dirichletBeta[-1] = 1.
            
        if initial_cond is not None:
            self.initial_cond = initial_cond
            
        if initial_cond_jnp is not None:
            self.initial_cond_jnp = initial_cond_jnp

        self.eps = eps

        self.buildGraph()
        
        # Default velocity
        self.v = 1. * np.ones(shape=(self.ne,))

        

class Example8(Graph):
    def __init__(self, eps=1e-1,
                 dirichletAlpha=None,
                 dirichletBeta=None,
                 initial_cond=None,
                 initial_cond_jnp=None
                ):

        super().__init__()
        self.title = 'Complex graph 100'
        self.id = 8

        
        A = np.loadtxt('../src/AdjacencyComplexGraph2.csv')
        
        
        G = nx.from_numpy_array(A, create_using=nx.MultiDiGraph)

        if False:
            n_nodes = len(G.nodes)
            add_list = []
            for e in G.in_degree():
                if e[1] == 0:
                    add_list.append((n_nodes, e[0]))
                    n_nodes += 1
            for e in G.out_degree():
                if e[1] == 0:
                    add_list.append((e[0], n_nodes))
                    n_nodes += 1
            G.add_edges_from(add_list)

        s = nx.topological_sort(G)
        idx = list(s)
        self.A = nx.to_numpy_array(G)[idx,:][:, idx].astype(np.int16)
        
        #self.A = nx.to_numpy_array(G).astype(np.int16)

        m, n = self.A.shape
        #self.pos = nx.kamada_kawai_layout(G)


        # Set boundaries
        tmin = 0.
        tmax = 1.
        xmin = 0.
        xmax = 1.

        # Lower bounds
        self.lb = np.array([tmin, xmin])

        # Upper bounds
        self.ub = np.array([tmax, xmax])

        m = self.A.shape[1]

        if dirichletAlpha is not None:
            self.dirichletAlpha = dirichletAlpha
        else:
            self.dirichletAlpha = np.zeros((m,))
            self.dirichletAlpha[0] = 1.
        
        if dirichletBeta is not None:
            self.dirichletBeta = dirichletBeta
        else:
            self.dirichletBeta = np.zeros((m,))
            self.dirichletBeta[-1] = 1.
            
        if initial_cond is not None:
            self.initial_cond = initial_cond
            
        if initial_cond_jnp is not None:
            self.initial_cond_jnp = initial_cond_jnp

        self.eps = eps

        self.buildGraph()
        
        # Default velocity
        self.v = 1. * np.ones(shape=(self.ne,))

if __name__ == '__main__':

    e0 = Example0()
    e1 = Example1()
    e2 = Example2()
    e3 = Example3()
    e4 = Example4()

    # Plot example graphs
    [fig, axs] = plt.subplots(2, 3)

    e0.plotGraph(ax=axs[0][0])
    e1.plotGraph(ax=axs[0][1])
    e2.plotGraph(ax=axs[0][2])
    e3.plotGraph(ax=axs[1][0])
    e4.plotGraph(ax=axs[1][1])

    plt.show()
