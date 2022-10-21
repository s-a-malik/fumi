"""
Based on https://github.com/ikostrikov/pytorch-a2c-ppo-acktr
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def init(module, weight_init, bias_init, gain=1.0, gain_for_bias_also=False):
    weight_init(module.weight.data, gain=gain)
    if module.bias is not None:
        if gain_for_bias_also:
            bias_init(module.bias.data, gain=gain)
        else:
            bias_init(module.bias.data)
    return module


# https://github.com/openai/baselines/blob/master/baselines/common/tf_util.py#L87
def init_normc_(weight, gain=1):
    weight.normal_(0, 1)
    weight *= gain / torch.sqrt(weight.pow(2).sum(1, keepdim=True))


def total_num_params(layer_sizes):
    num_params = 0
    for dim_in, dim_out in layer_sizes:
        num_params += dim_in*dim_out + dim_out # num params in linear layer
    return num_params


def linear_batched_weights(x, weight, bias):
    # a generalization of F.linear that allows for a different weight and bias for each item in the batch
    # Otherwise, F.linear fails on transposing the 3D weight tensor
    # E.g. This should pass the assert:
    if len(weight.shape)==2:
        return F.linear(x, weight, bias)
    assert len(weight.shape)==3, weight.shape
    assert len(x) == len(weight), (x.shape, weight.shape)

    w_T = weight.transpose(-1,-2)
    v = (x.unsqueeze(-1).expand_as(w_T)*w_T).sum(-2) + bias
    
    # You can check with this slow assert (comment out for speed):
    # batch_sz = len(x)
    # slow_results = []
    # for b in range(batch_sz):
    #     slow_results.append(F.linear(x[b], weight[b], bias[b]))
    # slow_results = torch.stack(slow_results)
    # assert v.shape == slow_results.shape, (v.shape, slow_results.shape)
    # assert torch.allclose(v, slow_results, atol=1e-05, rtol=0), (v, slow_results)

    return v

# kaiming uniform from https://pytorch.org/docs/stable/_modules/torch/nn/init.html#kaiming_uniform_
# but with gain argument instead of 'nonlinearity'
def kaiming_uniform_with_gain(tensor, mode='fan_in', gain=1):
    fan = nn.init._calculate_correct_fan(tensor, mode)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)

def kaiming_uniform_ignore_gain(tensor, gain=None):
    return nn.init.kaiming_uniform_(tensor)

# Hyperfan-In uniform for bias from (Chang et al., 2020), assuming ReLU and producing weights and bias
def HFI_bias_layer_init(final_hyper_hidden_sz, gain=1):
    variance = 1/(2*final_hyper_hidden_sz) # = 1/(2 * dl), since the rest of the terms cancel
    bound = gain*np.sqrt(3*variance) # gain is sqrt(2) for RelU, which cancels with denominator
    return lambda m: init(m, lambda w, gain: nn.init.uniform_(w, -bound, bound), 
                             lambda b: nn.init.constant_(b, 0), 
                             gain=None, gain_for_bias_also=False)

# Hyperfan-In uniform for weight from (Chang et al., 2020), assuming ReLU and producing weights and bias
def HFI_weight_layer_init(final_hyper_hidden_sz, base_curr_input_dim, gain=1):
    variance = 1/(2*final_hyper_hidden_sz*base_curr_input_dim) # = 1/(2 * dj * dk), since the rest of the terms cancel
    bound = gain*np.sqrt(3*variance) # gain is sqrt(2) for RelU, which cancels with denominator
    return lambda m: init(m, lambda w, gain: nn.init.uniform_(w, -bound, bound),
                             lambda b: nn.init.constant_(b, 0), 
                             gain=None, gain_for_bias_also=False)

# Defines the initialization of the weight or biases of the hyper-network so that each set of produced parameters is reasonable. 
# (Used in heads of the hypernetwork that output a weight matrix)
def init_hyper_match(param, is_weight, policy_initialisation_str, hyper_layer_dim, input_dim, output_dim, fix_init_b_gain, gain=None, scale=None):
    if policy_initialisation_str == 'normc':
        policy_initialisation = init_normc_ 
    elif policy_initialisation_str == 'orthogonal':
        policy_initialisation = nn.init.orthogonal_
    else:
        assert policy_initialisation_str == 'kaiming', policy_initialisation_str
        policy_initialisation = kaiming_uniform_with_gain if fix_init_b_gain else kaiming_uniform_ignore_gain
    # Assuming that the final layer (of size hyper_layer_dim) is one-hot, we want each weight matrix produced
    # of shape (input_dim, output_dim) to produce the original policy_initialisation
    if is_weight:
        assert param.shape == (input_dim*output_dim, hyper_layer_dim)
    else:
        assert param.shape == (input_dim*output_dim,), (param.shape, input_dim*output_dim)
    # each column is a weight matrix for the policy, so init each column
    # original_data = param.data.clone()
    if is_weight:
        for col_indx in range(hyper_layer_dim):
            col = param[:, col_indx]
            policy_weight = col.reshape((output_dim, input_dim)) # This should be the shape expected for the init tensor inputs
            policy_initialisation(policy_weight, gain=gain)
    else:
        policy_weight = param.reshape((output_dim, input_dim))
        policy_initialisation(policy_weight, gain=gain)
    # original_data = param.data.clone()
    if scale:
        with torch.no_grad():
            param.data.copy_(scale * param.data)
    # print(original_data-param.data)
    # print(param.data/original_data)

# How to initialize the hypernetwork head (weight and bias) that produces biases for the base network
def hyper_bias_layer_init(weight_for_uniform=None): # weight_for_uniform needed to use default initialization from Linear class
    # TODO: this assumes that the bias in the base net is created by bias in the hypernet. Normally this is 0, so it doesn't matter,
    # but the final layer of the critic by default has bias initialized with uniform init. (When weight_for_uniform is not None.)
    # In this case, we may want to allow for the option to generate the bias init using weights. (But I doubt this will affect anything.)
    if weight_for_uniform is not None:
        # default behavior from Linear class
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight_for_uniform)
        bound = 1 / math.sqrt(fan_in)
        init_for_hyper_bias = lambda b: nn.init.uniform_(b, -bound, bound)
    else:
        init_for_hyper_bias = lambda b: nn.init.constant_(b, 0)
    # get an init function that sets weight and bias to 0 so that bias produced is 0
    return lambda m: init(m, lambda w, gain: nn.init.constant_(w, 0), 
                             init_for_hyper_bias, 
                             None)

# How to initialize the hypernetwork head (weight and bias) that produces weights for the base network
def hyper_weight_layer_init(activation_function, policy_initialisation_str, hyper_layer_dim, input_dim, output_dim, fix_init_b_gain, override_gain=None,
                            adjust_weights=True, adjust_bias=False, use_film=False):
    if override_gain is None:
        override_gain = nn.init.calculate_gain(activation_function)
    # get an init function that sets each weight so it is similar to original policy weight init
    scale = .5 if adjust_weights and adjust_bias else None
    # define weight init
    if adjust_weights:
        if use_film:
            weight_init = lambda w, gain: nn.init.constant_(w, 1 if scale is None else scale) # In FiLM, init all weights that produce scaling to 1 so that one-hot input produces scaling = 1. (.5 is shared with bias)
        else:
            weight_init = lambda w, gain: init_hyper_match(w, True, policy_initialisation_str, hyper_layer_dim, input_dim, output_dim, fix_init_b_gain, gain=gain, scale=scale)
    else:
        weight_init = lambda w, gain: nn.init.constant_(w, 0)
    # define bias init
    if adjust_bias:
        if use_film:
            bias_init = lambda b: nn.init.constant_(b, 1 if scale is None else scale) # In FiLM, init all bias that produce scaling to 1 so that it produces scaling = 1. (.5 is shared with weights)
        else:
            gain_for_bias_also = True
            bias_init = lambda b, gain: init_hyper_match(b, False, policy_initialisation_str, hyper_layer_dim, input_dim, output_dim, fix_init_b_gain, gain=gain, scale=scale)
    else:
        gain_for_bias_also = False
        bias_init = lambda b: nn.init.constant_(b, 0)
    # define layer initializer
    if use_film:
        override_gain = 1 # all gain on weights should be taken care of through weight init in base net
        gain_for_bias_also = False
    return lambda m: init(m, weight_init, 
                             bias_init, 
                             override_gain, gain_for_bias_also=gain_for_bias_also)

class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs, gain, create_params=True):
        super(Categorical, self).__init__()

        self.create_params = create_params

        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0),
                               gain=gain)

        self.linear = init_(nn.Linear(num_inputs, num_outputs)) if self.create_params else None

    def forward(self, x, weight=None, bias=None, scaling=None, return_dist_params=False):
        if (weight is None) and (bias is None): # Normal
            assert self.create_params
            x = self.linear(x)
        elif scaling is None: # HyperNet
            assert (weight is not None) and (bias is not None), "Must have weight and bias for HyperNet" 
            x = linear_batched_weights(x, weight, bias)
        else: # FiLM
            assert (weight is None) and (bias is not None)
            self.linear.bias = None
            x = self.linear(x)
            x = x*scaling + bias
        if return_dist_params:
            return x
        return FixedCategorical(logits=x)


class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs, init_std, min_std,
                 action_low, action_high, norm_actions_of_policy, gain, create_params=True):
        # create_params=False allows forward to take in a linear layer parameters. Necessary when using hypernetwork.
        # Note: create_params=False WILL still create logstd parameter, shared across all networks
        super(DiagGaussian, self).__init__()

        self.create_params = create_params

        init_ = lambda m: init(m,
                               init_normc_,
                               lambda x: nn.init.constant_(x, 0),
                               gain=gain)

        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs)) if self.create_params else None
        self.logstd = nn.Parameter(np.log(torch.zeros(num_outputs) + init_std)) # Always created (easier to implement)
        self.min_std = torch.tensor([min_std]).to(device)

        # whether or not to conform to the action space given by the env
        # (scale / squash actions that the network outpus)
        self.norm_actions_of_policy = norm_actions_of_policy
        if len(np.unique(action_low)) == 1 and len(np.unique(action_high)) == 1:
            self.unique_action_limits = True
        else:
            self.unique_action_limits = False

        self.action_low = torch.from_numpy(action_low).to(device)
        self.action_high = torch.from_numpy(action_high).to(device)

    def forward(self, x, weight=None, bias=None, scaling=None, return_dist_params=False):
        if (weight is None) and (bias is None): # Normal
            assert self.create_params
            action_mean = self.fc_mean(x)
        elif scaling is None: # HyperNet
            assert (weight is not None) and (bias is not None), "Must have weight and bias for HyperNet" 
            action_mean = linear_batched_weights(x, weight, bias)
        else: # FiLM
            assert (weight is None) and (bias is not None)
            self.fc_mean.bias = None
            action_mean = self.fc_mean(x)
            action_mean = action_mean*scaling + bias

        if self.norm_actions_of_policy:
            if self.unique_action_limits and \
                    torch.unique(self.action_low) == -1 and \
                    torch.unique(self.action_high) == 1:
                action_mean = torch.tanh(action_mean)
            else:
                # TODO: this isn't tested
                action_mean = torch.sigmoid(action_mean) * (self.action_high - self.action_low) + self.action_low
        std = torch.max(self.min_std, self.logstd.exp())
        if return_dist_params:
            return action_mean, std
        return FixedNormal(action_mean, std)


class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().reshape(1, -1)
        else:
            bias = self._bias.t().reshape(1, -1, 1, 1)

        return x + bias
