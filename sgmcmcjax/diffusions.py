import jax.numpy as jnp
from jax import lax, random
from .diffusion_util import diffusion, diffusion_sghmc, diffusion_palindrome

### diffusions

@diffusion
def sgld(dt):
    "https://www.ics.uci.edu/~welling/publications/papers/stoclangevin_v6.pdf"
    dt = make_schedule(dt)

    def init_fn(x):
        return x

    def update(i, k, g, x):
        return x + dt(i)*g + jnp.sqrt(2*dt(i))*random.normal(k, shape=jnp.shape(x))

    def get_params(x):
        return x

    return init_fn, update, get_params

@diffusion
def psgld(dt, alpha=0.99, eps=1e-5):
    "https://arxiv.org/pdf/1512.07666.pdf"
    dt = make_schedule(dt)

    def init_fn(x):
        v = jnp.zeros_like(x)
        return x, v

    def update(i, k, g, state):
        x, v = state
        v = alpha*v + (1-alpha)*jnp.square(g)
        G = 1./(jnp.sqrt(v)+eps)
        return x + dt(i)*0.5*G*g + jnp.sqrt(dt(i)*G)*random.normal(k, shape=jnp.shape(x)), v

    def get_params(state):
        x, _ = state
        return x

    return init_fn, update, get_params

@diffusion
def sgldAdam(dt, beta1=0.9, beta2=0.999, eps=1e-8):
    "https://arxiv.org/abs/2105.13059"
    dt = make_schedule(dt)

    def init_fn(x):
        m = jnp.zeros_like(x)
        v = jnp.zeros_like(x)
        return x, m, v

    def update(i, k, g, state):
        x,m,v = state
        m = beta1*m + (1-beta1)*g
        v = beta2*v + (1-beta2)*jnp.square(g)
        m_hat = m/(1-beta1**(i+1))
        v_hat = v/(1-beta2**(i+1))
        adapt_dt = dt(i)/(jnp.sqrt(v_hat) + eps)

        return x + adapt_dt*0.5*m_hat + jnp.sqrt(adapt_dt)*random.normal(key=k, shape=jnp.shape(x)), m, v

    def get_params(state):
        x, _, _ = state
        return x

    return init_fn, update, get_params

@diffusion_sghmc
def sghmc(dt, alpha=0.01, beta=0):
    "https://arxiv.org/abs/1402.4102"
    dt = make_schedule(dt)

    def init_fn(x):
        v = jnp.zeros_like(x)
        return x, v

    def update(i, k, g, state):
        x, v = state
        x = x + v
        v = v + dt(i)*g - alpha*v + jnp.sqrt(2*(alpha - beta)*dt(i))*random.normal(k, shape=jnp.shape(x))
        return x, v

    def get_params(state):
        x, _ = state
        return x

    def resample_momentum(i, k, x):
        v = jnp.sqrt(dt(i))*random.normal(k, shape=jnp.shape(x))
        return x, v

    return init_fn, update, get_params, resample_momentum

@diffusion_palindrome
def baoab(dt, gamma, tau=1):
    dt = make_schedule(dt)

    def init_fn(x):
        v = jnp.zeros_like(x)
        return x, v

    def update1(i, k, g, state):
        x, v = state

        v = v + dt(i)*0.5*g
        x = x + v*dt(i)*0.5

        c1 = jnp.exp(-gamma*dt(i))
        c2 = jnp.sqrt(1 - c1**2)
        v = c1*v + tau*c2*random.normal(k, shape=jnp.shape(v))

        x = x + v*dt(i)*0.5

        return x, v

    def update2(i, k, g, state):
        x, v = state
        v = v + dt(i)*0.5*g
        return x, v

    def get_params(state):
        x, _ = state
        return x

    return init_fn, (update1, update2), get_params

@diffusion
def sgnht(dt, a=0.01):
    "http://people.ee.duke.edu/~lcarin/sgnht-4.pdf: Algorithm 2"
    dt = make_schedule(dt)

    def init_fn(x):
        v = jnp.zeros_like(x)
        alpha = a
        return x, v, alpha

    def initial_momentum(kv):
        "sample momentum at the first iteration"
        k, v = kv
        key, subkey = random.split(k)
        v = jnp.sqrt(dt(0))*random.normal(subkey, shape=v.shape)
        return key, v

    def update(i, k, g, state):
        x, v, alpha = state
        k,v = lax.cond(i==0,
                initial_momentum,
                lambda kv: (k,v),
                (k,v)
            )
        v = v - alpha*v + dt(i)*g + jnp.sqrt(2*a*dt(i))*random.normal(k, shape=jnp.shape(x))
        x = x + v
        alpha = alpha + (jnp.linalg.norm(v)**2)/v.size - dt(i)
        return x, v, alpha

    def get_params(state):
        x, _, _ = state
        return x

    return init_fn, update, get_params

@diffusion_palindrome
def badodab(dt, a=0.01):
    "https://arxiv.org/abs/1505.06889"
    dt = make_schedule(dt)

    def init_fn(x):
        v = jnp.zeros_like(x)
        alpha = a
        return x, v, alpha

    def update(i, k, g, state):
        x, v, alpha = state

        dt2 = dt(i)/2
        mu = 1.
        sigma = 1.

        v = v + dt2*g
        x = x + dt2*v
        alpha = alpha + (dt2/mu)*(jnp.linalg.norm(v) - v.size)

        c1 = jnp.exp(-alpha*dt(i))
        c2 = jnp.where(alpha==0, jnp.sqrt(dt(i)), jnp.sqrt(jnp.abs((1-c1**2)/(2*alpha))))
        v = c1*v + c2*sigma*random.normal(k, shape=jnp.shape(v))

        alpha = alpha + (dt2/mu)*(jnp.linalg.norm(v) - v.size)
        x = x + dt2*v
        return x, v, alpha

    def update2(i, k, g, state):
        x, v, alpha = state
        v = v + dt(i)*0.5*g
        return x, v, alpha

    def get_params(state):
        x, _, _ = state
        return x

    return init_fn, (update, update2), get_params


### step size schedules
def constant(step_size):
    def schedule(i):
        return step_size
    return schedule

def welling_teh_schedule(a,b, gamma=0.55):
    "Polynomial schedule from https://www.ics.uci.edu/~welling/publications/papers/stoclangevin_v6.pdf"
    def schedule(i):
        return a*(b+i)**(-gamma)
    return schedule

def cyclical_schedule(alpha_0, M, K):
    "https://arxiv.org/abs/1902.03932"
    def schedule(i):
        mod_term = (i-1) % jnp.ceil(K/M)
        return alpha_0*0.5*(jnp.cos( jnp.pi*mod_term /jnp.ceil(K/M) ) + 1)
    return schedule

def make_schedule(scalar_or_schedule):
    if callable(scalar_or_schedule):
        return scalar_or_schedule
    elif jnp.ndim(scalar_or_schedule) == 0:
        return constant(scalar_or_schedule)
    else:
        raise TypeError(type(scalar_or_schedule))
