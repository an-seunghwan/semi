#%%
import tensorflow as tf
import numpy as np
#%%
def weight_decay_decoupled(model, buffer_model, decay_rate):
    # weight decay
    for var, buffer_var in zip(model.variables, buffer_model.variables):
        var.assign(var - decay_rate * buffer_var)
    # update buffer model
    for var, buffer_var in zip(model.variables, buffer_model.variables):
        buffer_var.assign(var)
#%%
def linear_rampup(current, lampup_length):
    if current >= lampup_length:
        return 1.
    else:
        return current / lampup_length
#%%
def cosine_rampdown(current, rampdown_length):
    return float(0.5 * (np.cos(np.pi * current / rampdown_length) + 1))
#%%