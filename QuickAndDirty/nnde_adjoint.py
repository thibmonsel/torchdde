import torch
import numpy as np
import scipy
import scipy.integrate as sciinteg
import torch.nn as nn
from . import interpolators
#
# Implements a version of the NeuralODE adjoint optimisation algorithm, with the Adaptive Checkpoint Adjoint method
#
# Original NODE : https://arxiv.org/abs/1806.07366
# ACA version : https://arxiv.org/abs/2006.02493
#
# The forward integration is done through the scipy IVP solver.
# The backpropagation is based on the timesteps chosen by the scipy solver and only requires a step function for 
# the relevant scheme to be added in the code. 
# Any explicit scheme available with the scipy solver can be easily added here. 

#
# NB : This code is heavily based on the torch_ACA package (https://github.com/juntang-zhuang/torch_ACA)


# Used for float comparisons
tiny = 1e-8

class nddeint_ACA(torch.autograd.Function):

    @staticmethod
    def forward(ctx, history, func, options, *params):

        

        # Saving parameters for backward()
        ctx.func = func
        #ctx.flat_params = flat_params
        with torch.no_grad():
            ctx.save_for_backward(*params)
        ctx.options = options
        ctx.nSteps = options['nSteps']
        ctx.dt = options['dt']
        t = options['t0']
        val = history(t)
        delays = func.delays
        # Device used
        argdev = val.device
        
        argshape = val.shape
        argdtype = val.dtype

        # Simulation
        with torch.no_grad():
            values = [val]
            alltimes = [t]
            for i in range(ctx.nSteps):
                val = val + ctx.dt * ctx.func(t,val,*[history(t-tau) for tau in delays])
                t += ctx.dt
                history.add_point(t,val)
                values.append(val)
                alltimes.append(t)
        # Retrieving the time stamps selected by the solver
        ctx.alltimes = alltimes
        ctx.history = history
        
        ctx.allstates = torch.stack(values)
        evaluations = ctx.allstates[options['eval_idx']]
        
        return evaluations

    @staticmethod
    def backward(ctx, *grad_y):
        # This function implements the adjoint gradient estimation method for NODEs

        # grad_output holds the gradient of the loss w.r.t. each evaluation step
        grad_output = grad_y[0]
        
        # h is the value of the forward time step
        h = ctx.dt

        # Retrieving the time mesh and the corresponding states created in forward()
        allstates = ctx.allstates
        time_mesh = ctx.alltimes

        # f_params holds the NDDE parameters for which a gradient will be computed
        params = ctx.saved_tensors
        

        # The last step of the time mesh is an evaluation step, thus the adjoint state which corresponds to the
        # gradient of the loss w.r.t. the evaluation states is initialised with the gradient corresponding to
        # the last evaluation time.

        adjoint_state = grad_output[-1]
        i_ev=len(time_mesh)-2

        adjoint_history = interpolators.TorchLinearInterpolator(
                            adjoint_state.reshape(adjoint_state.shape[0],1,*adjoint_state.shape[1:]),
                            torch.tensor(time_mesh[-1]).reshape(1).to(adjoint_state.device))
        
        out2 = None

        # The adjoint state as well as the parameters' gradient are integrated backwards in time.
        # Following the Adaptive Checkpoint Adjoint method, the time steps and corresponding states of the forward
        # integration are re-used by going backwards in the time mesh.
        for i in range(len(time_mesh),0,-1):
            t = time_mesh[i-1]
            if i!=len(time_mesh):
                adjoint_history.add_point(t, adjoint_state)
            # Backward Integrating the adjoint state and the parameters' gradient between time i and i-1

            
            with torch.enable_grad():
                # Taking a step with the NODE function to build a graph which will be differentiated
                # so as to integrate the adjoint state and the parameters' gradient
                derivative = 0.

                
                
                z_var = torch.autograd.Variable(ctx.history(t),requires_grad=True)

                for tau in ctx.func.delays+[0]:
                    if (t+tau - time_mesh[-1]) < tiny:
                        temp_adjoint = adjoint_history(t+tau)
                        delayed_values = []
                        for tau2 in ctx.func.delays:
                            if abs(tau-tau2) < tiny:
                                delayed_values.append(z_var)
                            else:
                                delayed_values.append(ctx.history(t+tau-tau2))
                                
                        if tau == 0:
                            out = ctx.func(t,z_var,*delayed_values)
                        else:
                            out = ctx.func(t+tau,ctx.history(t+tau),*delayed_values)
                        if tau==0:
                            param_derivative_inc =  torch.autograd.grad(
                                            out, params,
                                            adjoint_state, retain_graph=True)

                        derivative_inc = torch.autograd.grad(
                                            out,  z_var,
                                            -temp_adjoint)[0]


                        derivative = derivative + derivative_inc

                adjoint_state = adjoint_state + ctx.dt * derivative

            # incrementing the parameters' grad
            if out2 is None:
                out2 = tuple([ctx.dt * p for p in param_derivative_inc])
            else:
                for _1, _2 in zip([*out2], [*param_derivative_inc]):
                    _1 = _1 - ctx.dt * _2
            # When reaching an evaluation step, the adjoint state is incremented with the gradient of the corresponding
            # evaluation step 
            next_i=i-1
            if next_i in ctx.options['eval_idx'] and i!=len(time_mesh):
                adjoint_state += grad_output[i_ev]
                i_ev = i_ev - 1




        # Returning the gradient value for each forward() input
        out = tuple([None] + [None,  None])+out2
    

        return out


        
def nddesolve_adjoint(history, func, options):
    # Main function to be called to integrate the NODE

    # z0 : (tensor) Initial state of the NODE
    # func : (torch Module) Derivative of the NODE
    # options : (dict) Dictionary of solver options, should at least have a 

    # The parameters for which a gradient should be computed are passed as a flat list of tensors to the forward function
    # The gradient returned by backward() will take the same shape.
    #flat_params = flatten_grad_params(func.parameters())
    params = find_parameters(func)

    # Forward integrating the NODE and returning the state at each evaluation step
    zs = nddeint_ACA.apply(history, func, options, *params)
    return zs


def flatten_grad_params(params):
    # Parameters for which a grad is required are flattened and returned as a list
    flat_params = []
    for p in params:
        if p.requires_grad:
            flat_params.append(p.contiguous().view(-1))

    return torch.cat(flat_params) if len(flat_params) > 0 else torch.tensor([])

def flatten_params(params):
    # values in the params tuple are flattened and returned as a list
    flat_params = [p.contiguous().view(-1) for p in params]
    return torch.cat(flat_params) if len(flat_params) > 0 else torch.tensor([])

def get_integration_options(n0,n1,dt,substeps):

    sub_dt = float(dt / substeps)
    nSteps = substeps*(n1-n0)
    
    integration_options = {
                            't0':n0*dt,
                            'dt':sub_dt,
                            'nSteps':nSteps,
                            'eval_idx':np.arange(0,nSteps+1,substeps)
                        }

    return integration_options

def find_parameters(module):

    assert isinstance(module, nn.Module)

    # If called within DataParallel, parameters won't appear in module.parameters().
    if getattr(module, '_is_replica', False):

        def find_tensor_attributes(module):
            tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v) and v.requires_grad]
            return tuples

        gen = module._named_members(get_members_fn=find_tensor_attributes)
        return [param for _, param in gen]
    else:
        
        return [r for r in module.parameters() if r.requires_grad]
        