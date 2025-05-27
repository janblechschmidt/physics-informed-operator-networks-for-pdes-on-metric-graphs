import jax.numpy as jnp
from jax import config, random

# Define RBF kernel
def RBF(x1, x2, params):
    output_scale, lengthscales = params
    diffs = jnp.expand_dims(x1 / lengthscales, 1) - \
            jnp.expand_dims(x2 / lengthscales, 0)
    r2 = jnp.sum(diffs**2, axis=2)
    return output_scale * jnp.exp(-0.5 * r2)

def get_sample_fns(length_scale=0.2, N=512):
    
    # Compute Cholesky decomposition of Kernel matrix for fixed grid size
    config.update("jax_enable_x64", True)
    
    # Sample GP prior at a fine grid
    gp_params = (1.0, length_scale)
    jitter = 1e-10
    X = jnp.linspace(0, 1, N)
    K = RBF(X.reshape((N,1)), X.reshape((N,1)), gp_params)
    L = jnp.linalg.cholesky(K + jitter * jnp.eye(N))
    config.update("jax_enable_x64", False)
    

    def sample_GP_values(key):
        gp_sample = jnp.dot(L, random.normal(key, (N,)))
        return gp_sample

    def sample_u_in_fn(key, shift=0.0):
        gp_sample = sample_GP_values(key)
        gp_scale = 1. / (gp_sample.max() - gp_sample.min() + shift)
        gp_sample = gp_scale * (gp_sample - gp_sample.min() + shift)
        gp_fn = lambda t: jnp.interp(t, X, gp_sample)
        return gp_fn
    
    def sample_u_out_fn(key, shift=0.0):
        gp_sample = sample_GP_values(key)
        gp_scale = 1. / (gp_sample.max() - gp_sample.min() + shift)
        gp_sample = gp_scale * (gp_sample - gp_sample.min() + shift)
        gp_fn = lambda t: jnp.interp(t, X, gp_sample)
        return gp_fn
   
    def sample_u_init_fn(key, shift=0.0):
        gp_sample = sample_GP_values(key)
        gp_scale = 1. / (gp_sample.max() - gp_sample.min() + shift)
        gp_sample = gp_scale * (gp_sample - gp_sample.min() + shift)
        gp_fn = lambda t: jnp.interp(t, X, gp_sample)
        return gp_fn
        
    return sample_u_in_fn, sample_u_out_fn, sample_u_init_fn
