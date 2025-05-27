import jax
import jax.numpy as jnp
from functools import partial
from jax import jit, vmap, grad, value_and_grad
import optax

import jax.random as random
from jax.example_libraries import optimizers
from jax.flatten_util import ravel_pytree
import itertools
from tqdm import trange

def n_res_data(res_data):
    s = res_data[2].shape
    return s[0]*s[1]

def n_init_data(init_data):
    return init_data[0].shape[0] * init_data[1].shape[0]

def n_bc_data(bc_data):
    s = bc_data[2].shape
    return s[0]*s[1]*s[2]
    
@jit
def mse(x): 
    return jnp.mean(x**2)

@partial(jit, static_argnums=(1,2))
def shuffle_idx(key, n, n_split):        
    return random.permutation(key, n).reshape((n_split,-1))

@jit
def get_res_batch_from_index(u_res, y_res, s_res, batch_idx):
    idx_sample, idx_res = jnp.divmod(batch_idx, s_res.shape[1])

    s_u = s_res[(idx_sample, idx_res, 0)]
    s_t = y_res[(idx_sample, idx_res, 0)]
    s_x = y_res[(idx_sample, idx_res, 1)]
    s_init = u_res[idx_sample,:]
    
    return s_u, s_t, s_x, s_init

@jit
def get_init_batch_from_index(u_init, x_init, batch_idx, N_DATA_BC):
    idx_sample, idx_pos = jnp.divmod(batch_idx, x_init.shape[0])
    s_x = x_init[idx_pos]
    s_u = u_init[(idx_sample, N_DATA_BC*2 + idx_pos)]
    s_t = jnp.zeros_like(s_x)
    s_init = u_init[idx_sample, :]
    
    return s_u, s_t, s_x, s_init

@jit
def get_bc_batch_from_index(u_bc, t_bc, s_bc, batch_idx):
    idx_sample, idx_ij = jnp.divmod(batch_idx, s_bc.shape[1]*s_bc.shape[2])
    idx_inout, idx_time = jnp.divmod(idx_ij, s_bc.shape[1])
    s_t = t_bc[idx_time]
    s_x = idx_inout.astype(jnp.float32)
    s_init = u_bc[idx_sample, :]
    s_u = s_bc[(idx_sample, idx_time, idx_inout)]
    
    return s_u, s_t, s_x, s_init

@jit
def get_physics_bcs_batch_from_index(u_bc, t_bc, s_bc, batch_idx, N_DATA_BC):
    idx_sample, idx_ij = jnp.divmod(batch_idx, s_bc.shape[1]*s_bc.shape[2])
    idx_inout, idx_time = jnp.divmod(idx_ij, s_bc.shape[1])
    s_t = t_bc[idx_time]
    s_x = idx_inout.astype(jnp.float32)
    s_init = u_bc[idx_sample, :]
    s_u = u_bc[(idx_sample, idx_time + idx_inout * N_DATA_BC)]

    return s_u, s_t, s_x, s_init
    
# Define MLP
def MLP(layers, activation=jax.nn.tanh):
    ''' Vanilla MLP'''
    def init(rng_key):
        
        def init_layer(key, d_in, d_out):
            k1, k2 = jax.random.split(key)
            glorot_stddev = 1. / jnp.sqrt((d_in + d_out) / 2.)
            W = glorot_stddev * jax.random.normal(k1, (d_in, d_out))
            b = jnp.zeros(d_out)
            return W, b
            
        key, *keys = jax.random.split(rng_key, len(layers))
        params = list(map(init_layer, keys, layers[:-1], layers[1:]))
        return params
        
    def apply(params, inputs):
        for W, b in params[:-1]:
            outputs = jnp.dot(inputs, W) + b
            inputs = activation(outputs)
        W, b = params[-1]
        outputs = jnp.dot(inputs, W) + b
        return outputs
        
    return init, apply

# Define modified MLP
def modified_MLP(layers, activation=jax.nn.tanh):
    def xavier_init(key, d_in, d_out):
        glorot_stddev = 1. / jnp.sqrt((d_in + d_out) / 2.)
        W = glorot_stddev * jax.random.normal(key, (d_in, d_out))
        b = jnp.zeros(d_out)
        return W, b

    def init(rng_key):
        U1, b1 =  xavier_init(jax.random.PRNGKey(12345), layers[0], layers[1])
        U2, b2 =  xavier_init(jax.random.PRNGKey(54321), layers[0], layers[1])
        def init_layer(key, d_in, d_out):
            k1, k2 = jax.random.split(key)
            W, b = xavier_init(k1, d_in, d_out)
            return W, b
        key, *keys = jax.random.split(rng_key, len(layers))
        params = list(map(init_layer, keys, layers[:-1], layers[1:]))
        return (params, U1, b1, U2, b2) 

    def apply(params, inputs):
        params, U1, b1, U2, b2 = params
        U = activation(jnp.dot(inputs, U1) + b1)
        V = activation(jnp.dot(inputs, U2) + b2)
        for W, b in params[:-1]:
            outputs = activation(jnp.dot(inputs, W) + b)
            inputs = jnp.multiply(outputs, U) + jnp.multiply(1 - outputs, V) 
        W, b = params[-1]
        outputs = jnp.dot(inputs, W) + b
        return outputs
    return init, apply

# Define Fourier feature net
def FF_MLP(layers, freqs=5, activation=jax.nn.tanh):
   # Define input encoding function
    def input_encoding(x, w):
        out = jnp.hstack([jnp.sin(jnp.dot(x, w)),
                         jnp.cos(jnp.dot(x, w))])
        return out
    # The following is used in general
    FF = freqs * jax.random.normal(jax.random.PRNGKey(0), (layers[0], layers[1]//2))
    # Values used to train the models in the paper were stored in npy files via
    # jnp.save(f'FF_{layers[0]}_{layers[1]//2}', FF)
    # and can be loaded if necessary
    
    def init(rng_key):
      def init_layer(key, d_in, d_out):
          k1, k2 = jax.random.split(key)
          glorot_stddev = 1. / jnp.sqrt((d_in + d_out) / 2.)
          W = glorot_stddev * jax.random.normal(k1, (d_in, d_out))
          b = jnp.zeros(d_out)
          return W, b
      key, *keys = jax.random.split(rng_key, len(layers))
      params = list(map(init_layer, keys, layers[1:-1], layers[2:]))
      return params
    def apply(params, inputs):
        H = input_encoding(inputs, FF)
        for W, b in params[:-1]:
            outputs = jnp.dot(H, W) + b
            H = activation(outputs)
        W, b = params[-1]
        outputs = jnp.dot(H, W) + b
        return outputs
    return init, apply


# Define Physics-informed DeepONet model
class PI_DeepONet:
    def __init__(self, graph, branch_layers, trunk_layers, branch_net=modified_MLP, trunk_net=modified_MLP, solver=None, solver_is_lbfgs=False):

        self.graph = graph
        
        # Network initialization and evaluation functions
        self.branch_init, self.branch_apply = branch_net(branch_layers, activation=jax.nn.tanh)
        self.trunk_init, self.trunk_apply = trunk_net(trunk_layers, activation=jax.nn.tanh)

        # Initialize
        branch_params = self.branch_init(rng_key = random.PRNGKey(1234))
        trunk_params = self.trunk_init(rng_key = random.PRNGKey(4321))
        self.params = (branch_params, trunk_params)

        # Use optimizers to set optimizer initialization and update functions
        self.solver_is_lbfgs = solver_is_lbfgs
        if solver is None:
            self.solver = optax.adam(optax.schedules.exponential_decay(1e-3, transition_steps=2000, decay_rate=0.9))
        else:
            self.solver = solver
        
        self.opt_state = self.solver.init(self.params)

        # Logger
        self.itercount = itertools.count()
        
        self.train_loss_log = []
        self.train_loss_ics_log = []
        self.train_loss_bcs_log = []
        self.train_loss_res_log = []
        self.train_loss_physics_log = []
        self.train_loss_bnd_physics_log = []
        
        self.val_loss_log = []
        self.val_loss_ics_log = []
        self.val_loss_bcs_log = []
        self.val_loss_res_log = []
        self.val_loss_physics_log = []
        self.val_loss_bnd_physics_log = []

    # Define DeepONet architecture
    def operator_net(self, params, u, t, x):
        branch_params, trunk_params = params
        y = jnp.stack([t,x])
        B = self.branch_apply(branch_params, u)
        T = self.trunk_apply(trunk_params, y)
        outputs = jnp.sum(B * T)
        return   outputs

    # Define PDE residual        
    def residual_net(self, params, u, t, x):
        s = self.operator_net(params, u, t, x)
        s_t = grad(self.operator_net, argnums=2)(params, u, t, x)
        s_x = grad(self.operator_net, argnums=3)(params, u, t, x)
        s_xx= grad(grad(self.operator_net, argnums=3), argnums=3)(params, u, t, x)

        res = self.graph.pde_param(s, s_t, s_x, s_xx, u[-1])

        return res

    # Define flux of pde
    def flux_net(self, params, u, t, x):
        s = self.operator_net(params, u, t, x)
        s_x = grad(self.operator_net, argnums=3)(params, u, t, x)
        res = self.graph.flux_param(s, s_x, u[-1])
        return res, s

    # Define initial loss
    @partial(jit, static_argnums=(0,))
    def loss_ics(self, params, batch):
        
        # Fetch data
        outputs, t, x, u = batch
        
        # Compute forward pass
        s_pred = vmap(self.operator_net, (None, 0, 0, 0))(params, u, t, x)

        # Compute loss
        loss = outputs - s_pred
                
        return mse(loss)

    # Define boundary physics loss
    @partial(jit, static_argnums=(0,))
    def loss_physics_bcs_single_edge(self, params, batch):
        
        # Fetch data
        outputs, t, x, u = batch
        pred, uval = vmap(self.flux_net, (None, 0, 0, 0))(params, u, t, x)
        
        loss = pred - outputs * (1- uval) * (1 - x) - outputs * uval * x 
        
        return mse(loss)

    # Define boundary physics loss
    @partial(jit, static_argnums=(0,))
    def loss_physics_bcs_inflow(self, params, batch):
        # Fetch data
        outputs, t, x, u = batch
        pred, uval = vmap(self.flux_net, (None, 0, 0, 0))(params, u, t, x)

        loss = pred - outputs * (1- uval) * (1 - x) - outputs * x
        
        return mse(loss)
    
    # Define boundary physics loss
    @partial(jit, static_argnums=(0,))
    def loss_physics_bcs_inner(self, params, batch):
        # Fetch data
        outputs, t, x, u = batch
        pred, uval = vmap(self.flux_net, (None, 0, 0, 0))(params, u, t, x)

        loss = pred - outputs * (1 - x) - outputs * x
        
        return mse(loss)
    
    # Define boundary physics loss
    @partial(jit, static_argnums=(0,))
    def loss_physics_bcs_outflow(self, params, batch):
        # Fetch data
        outputs, t, x, u = batch
        pred, uval = vmap(self.flux_net, (None, 0, 0, 0))(params, u, t, x)
        
        loss = pred  - outputs * (1 - x) - outputs * uval * x
        
        return mse(loss)

    # Define boundary loss
    @partial(jit, static_argnums=(0,))
    def loss_bcs(self, params, batch):
        # Fetch data
        outputs, t, x, u = batch

        # Compute forward pass
        s_pred = vmap(self.operator_net, (None, 0, 0, 0))(params, u, t, x)
        
        # Compute loss
        loss = outputs - s_pred
                
        return mse(loss)
        
    @partial(jit, static_argnums=(0,))
    def loss_physics(self, params, batch):
        
        # Fetch data
        outputs, t, x, u = batch
        # Compute forward pass
        pred = vmap(self.residual_net, (None, 0, 0, 0))(params, u, t, x)
        
        # Physics-informed loss
        loss = pred
        
        return mse(loss)
        
    # Define residual loss
    @partial(jit, static_argnums=(0,))
    def loss_res(self, params, batch):
        # Fetch data
        outputs, t, x, u = batch
        
        # Residual loss at known points
        val_pred = vmap(self.operator_net, (None, 0, 0, 0))(params, u, t, x)
        loss = val_pred - outputs
        # print(loss.shape)
        
        return mse(loss)

    # Define total loss
    @partial(jit, static_argnums=(0,))
    def loss(self, params, ics_batch, bcs_batch, res_batch, ph_bcs_batch, weights):
        loss =  weights['res'] *  self.loss_res(params, res_batch) 
        loss += weights['bcs'] * self.loss_bcs(params, bcs_batch)
        loss += weights['ics'] * self.loss_ics(params, ics_batch)
        loss += weights['physics'] * self.loss_physics(params, res_batch)
        loss += weights['physics_bcs_single_edge'] * self.loss_physics_bcs_single_edge(params, ph_bcs_batch)
        loss += weights['physics_bcs_inflow'] * self.loss_physics_bcs_inflow(params, ph_bcs_batch)
        loss += weights['physics_bcs_inner'] * self.loss_physics_bcs_inner(params, ph_bcs_batch)
        loss += weights['physics_bcs_outflow'] * self.loss_physics_bcs_outflow(params, ph_bcs_batch)
        return loss

        
    # Define total loss
    @partial(jit, static_argnums=(0,))
    def computeLossComplete(self, params, ics_batch, bcs_batch, res_batch, ph_bcs_batch, weights):
        loss_res = self.loss_res(params, res_batch) 
        loss_bcs = self.loss_bcs(params, bcs_batch)
        loss_ics = self.loss_ics(params, ics_batch)
        loss_physics = self.loss_physics(params, res_batch)
        loss_physics_bcs_single_edge = self.loss_physics_bcs_single_edge(params, ph_bcs_batch)
        loss_physics_bcs_inflow = self.loss_physics_bcs_inflow(params, ph_bcs_batch)
        loss_physics_bcs_inner = self.loss_physics_bcs_inner(params, ph_bcs_batch)
        loss_physics_bcs_outflow = self.loss_physics_bcs_outflow(params, ph_bcs_batch)
        
        wloss =  weights['res'] * loss_res
        wloss += weights['bcs'] * loss_bcs
        wloss += weights['ics'] * loss_ics
        wloss += weights['physics'] * loss_physics
        wloss += weights['physics_bcs_single_edge'] * loss_physics_bcs_single_edge
        wloss += weights['physics_bcs_inflow'] * loss_physics_bcs_inflow
        wloss += weights['physics_bcs_inner'] * loss_physics_bcs_inner
        wloss += weights['physics_bcs_outflow'] * loss_physics_bcs_outflow
        return wloss, loss_res, loss_bcs, loss_ics, loss_physics, loss_physics_bcs_single_edge, \
        loss_physics_bcs_inflow, loss_physics_bcs_inner, loss_physics_bcs_outflow

    # Define a compiled update step
    @partial(jit, static_argnums=(0,))
    def step(self, i, opt_state, ics_batch, bcs_batch, res_batch, ph_bcs_batch, weights, params):
        g = grad(self.loss)(params, ics_batch, bcs_batch,
                            res_batch, ph_bcs_batch, weights)
        updates, new_opt_state = self.solver.update(g, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state
        
    @partial(jit, static_argnums=(0,))
    def lbfgs_step(self, i, opt_state, ics_batch, bcs_batch, res_batch, ph_bcs_batch, weights, params):
        def fn(x): return self.loss(x, ics_batch, bcs_batch, res_batch, ph_bcs_batch, weights)
        v, g = optax.value_and_grad_from_state(fn)(params, state=opt_state)
        updates, new_opt_state = self.solver.update(g, opt_state, params, value=v, grad=g, value_fn=fn)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state

    # Optimize parameters in a loop
    def train(self, key, nEpochs, weights, RES_DATA, INIT_DATA, BC_DATA, N_SPLIT, N_DATA_BC, VAL_DATA):

        pbar = trange(nEpochs)
        n_val_init_data = n_init_data(VAL_DATA[1])
        n_val_bc_data = n_bc_data(VAL_DATA[2])
        n_val_res_data = n_res_data(VAL_DATA[0])
        self.best_model_loss = jnp.inf
        
        # Main training loop
        for it in pbar:
            
            # Once per epoch
            key, *keys = random.split(key, 4)
            res_idx = shuffle_idx(keys[0], n_res_data(RES_DATA), N_SPLIT)
            init_idx = shuffle_idx(keys[1], n_init_data(INIT_DATA), N_SPLIT)
            bc_idx = shuffle_idx(keys[2], n_bc_data(BC_DATA), N_SPLIT)
        
            for i in range(N_SPLIT):
                res_batch = get_res_batch_from_index(*RES_DATA, res_idx[i])
                ics_batch = get_init_batch_from_index(*INIT_DATA, init_idx[i], N_DATA_BC)
                bcs_batch = get_bc_batch_from_index(*BC_DATA, bc_idx[i])
                ph_bcs_batch = get_physics_bcs_batch_from_index(*BC_DATA, bc_idx[i], N_DATA_BC)

                if self.solver_is_lbfgs:
                    self.params, self.opt_state = self.lbfgs_step(next(self.itercount), self.opt_state, ics_batch, bcs_batch, res_batch, ph_bcs_batch, weights, self.params)
                else:
                    self.params, self.opt_state = self.step(next(self.itercount), self.opt_state, ics_batch, bcs_batch, res_batch, ph_bcs_batch, weights, self.params)
                
                if i % 10 == 0:
                    #params = self.get_params(self.opt_state)
                    val_loss = self.computeLossComplete(self.params,
                              get_init_batch_from_index(*VAL_DATA[1], jnp.arange(n_val_init_data), N_DATA_BC),
                              get_bc_batch_from_index(*VAL_DATA[2], jnp.arange(n_val_bc_data)),
                              get_res_batch_from_index(*VAL_DATA[0], jnp.arange(n_val_res_data)),
                              get_physics_bcs_batch_from_index(*VAL_DATA[2], jnp.arange(n_val_bc_data), N_DATA_BC),
                                                          weights)
                    # Compute losses
                    loss_value = val_loss[0]
                    loss_res_value = val_loss[1]
                    loss_bcs_value = val_loss[2]
                    loss_ics_value = val_loss[3]
                    loss_physics_value = val_loss[4]
                    
                    val_loss_bnd_physics_value = 0
                    if weights['physics_bcs_single_edge'] != 0:
                        val_loss_bnd_physics_value = weights['physics_bcs_single_edge'] * val_loss[5]
                    if weights['physics_bcs_inflow'] != 0:
                        val_loss_bnd_physics_value += weights['physics_bcs_inflow'] * val_loss[6]
                    if weights['physics_bcs_inner'] != 0:
                        val_loss_bnd_physics_value += weights['physics_bcs_inner'] * val_loss[7]
                    if weights['physics_bcs_outflow'] != 0:
                        val_loss_bnd_physics_value += weights['physics_bcs_outflow'] * val_loss[8]
    
                    # Store losses
                    self.val_loss_log.append(val_loss[0])
                    self.val_loss_ics_log.append(val_loss[3])
                    self.val_loss_bcs_log.append(val_loss[2])
                    self.val_loss_res_log.append(val_loss[1])
                    self.val_loss_physics_log.append(val_loss[4])
                    self.val_loss_bnd_physics_log.append(val_loss_bnd_physics_value)
    
    
                    # Print losses
                    pbar.set_postfix({'loss': val_loss[0], 
                                      'loss_ics' : val_loss[3],
                                      'loss_bcs' : val_loss[2], 
                                      'loss_res': val_loss[1],
                                      'loss_pde_ph': val_loss[4],
                                      'loss_bnd_ph': val_loss_bnd_physics_value})
                    

                    train_loss = self.computeLossComplete(self.params,
                                                          get_init_batch_from_index(*INIT_DATA, jnp.arange(n_val_init_data), N_DATA_BC),
                                                          get_bc_batch_from_index(*BC_DATA, jnp.arange(n_val_bc_data)),
                                                          get_res_batch_from_index(*RES_DATA, jnp.arange(n_val_res_data)),
                                                          get_physics_bcs_batch_from_index(*BC_DATA, jnp.arange(n_val_bc_data), N_DATA_BC),
                                                          weights)   
                    # Compute losses
                    train_loss_bnd_physics_value = 0
                    if weights['physics_bcs_single_edge'] != 0:
                        train_loss_bnd_physics_value = weights['physics_bcs_single_edge'] * train_loss[5]
                    if weights['physics_bcs_inflow'] != 0:
                        train_loss_bnd_physics_value += weights['physics_bcs_inflow'] * train_loss[6]
                    if weights['physics_bcs_inner'] != 0:
                        train_loss_bnd_physics_value += weights['physics_bcs_inner'] * train_loss[7]
                    if weights['physics_bcs_outflow'] != 0:
                        train_loss_bnd_physics_value += weights['physics_bcs_outflow'] * train_loss[8]
    
                    # Store losses
                    self.train_loss_log.append(train_loss[0])
                    self.train_loss_ics_log.append(train_loss[3])
                    self.train_loss_bcs_log.append(train_loss[2])
                    self.train_loss_res_log.append(train_loss[1])
                    self.train_loss_physics_log.append(train_loss[4])
                    self.train_loss_bnd_physics_log.append(train_loss_bnd_physics_value)
            if (it % 10 == 0) and it > 0:
                avg_last10_loss = jnp.mean(jnp.array(self.val_loss_log[-10:]))
                
                if avg_last10_loss < self.best_model_loss:
                    self.best_model_params = self.params
                    self.best_model_loss = avg_last10_loss
           
    # Evaluates predictions at test points  
    @partial(jit, static_argnums=(0,))
    def predict_s(self, params, U_star, Y_star):
        s_pred = vmap(self.operator_net, (None, 0, 0, 0))(params, U_star, Y_star[:,0], Y_star[:,1])
        return s_pred

    # Evaluates predictions for particular U_star at test points  
    @partial(jit, static_argnums=(0,))
    def predict_s_all(self, params, U_star, Y_star):
        s_pred = vmap(self.operator_net, (None, None, 0, 0))(params, U_star, Y_star[:,0], Y_star[:,1])
        return s_pred

    # Evaluates flux for particular U_star at test points  
    @partial(jit, static_argnums=(0,))
    def predict_flux_all(self, params, U_star, Y_star):
        flux_pred, s_pred = vmap(self.flux_net, (None, None, 0, 0))(params, U_star, Y_star[:,0], Y_star[:,1])
        return flux_pred, s_pred

        
    @partial(jit, static_argnums=(0,))
    def predict_flux(self, params, U_star, Y_star):
        flux_pred, s_pred = vmap(self.flux_net, (None, 0, 0, 0))(params, U_star, Y_star[:,0], Y_star[:,1])
        return flux_pred, s_pred

    @partial(jit, static_argnums=(0,))
    def predict_res(self, params, U_star, Y_star):
        r_pred = vmap(self.residual_net, (None, 0, 0, 0))(params, U_star, Y_star[:,0], Y_star[:,1])
        return r_pred
        
    @partial(jit, static_argnums=(0,))
    def predict_res_all(self, params, U_star, Y_star):
        r_pred = vmap(self.residual_net, (None, None, 0, 0))(params, U_star, Y_star[:,0], Y_star[:,1])
        return r_pred
