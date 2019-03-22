import torch.nn as nn
import math
from torch.nn import init
import torch
import pdb
import numpy as np
from torch.autograd import Function

'''def init_weight(module):
    if isinstance(module, nn.Linear):
        init.kaiming_normal_(module.weight)
        #bais cannot be handle by kaiming
        if module.bias is not None:
            init.kaiming_normal_(module.bias)'''

def init_weight(linear, pad_idx=None):
    if isinstance(linear, nn.Conv2d):
        init.xavier_normal_(linear.weight)
        '''n = linear.kernel_size[0] * linear.kernel_size[1] * linear.out_channels
        linear.weight.data.normal_(0, math.sqrt(2. / n))'''
    if isinstance(linear, nn.Linear):
        init.xavier_normal_(linear.weight)
    if isinstance(linear, nn.Embedding):
        bias = np.sqrt(3.0 / linear.weight.size(1))
        nn.init.uniform_(linear.weight, -bias, bias)
        if pad_idx is not None:
            linear.weight.data[pad_idx] = 0

def l2norm(input, p=2.0, dim=1, eps=1e-12):
    """
    Compute L2 norm, row-wise
    """
    return input / input.norm(p, dim).clamp(min=eps).expand_as(input)

def init_gru_cell(input):

    weight = eval('input.weight_ih')
    bias = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
    nn.init.uniform_(weight, -bias, bias)
    weight = eval('input.weight_hh')
    bias = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
    nn.init.uniform_(weight, -bias, bias)

    if input.bias:
        weight = eval('input.bias_ih' )
        weight.data.zero_()
        weight.data[input.hidden_size: 2 * input.hidden_size] = 1
        weight = eval('input.bias_hh')
        weight.data.zero_()
        weight.data[input.hidden_size: 2 * input.hidden_size] = 1

def init_lstm(input_lstm):

    for ind in range(0, input_lstm.num_layers):
        weight = eval('input_lstm.weight_ih_l'+str(ind))
        nn.init.orthogonal(weight)
        weight = eval('input_lstm.weight_hh_l'+str(ind))
        nn.init.orthogonal(weight)

    if input_lstm.bias:
        for ind in range(0, input_lstm.num_layers):
            weight = eval('input_lstm.bias_ih_l'+str(ind))
            weight.data.zero_()
            weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1
            weight = eval('input_lstm.bias_hh_l'+str(ind))
            weight.data.zero_()
            weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1


def build_mlp(dim_list, activation='relu', batch_norm='none',
              dropout=0, final_non_linearity=True):
    layers = []
    for i in range(len(dim_list) - 1):
        dim_in, dim_out = dim_list[i], dim_list[i + 1]
        layers.append(nn.Linear(dim_in, dim_out))
        final_layer = (i == len(dim_list) - 2)
        if not final_layer or final_non_linearity:
            if batch_norm == 'batch':
                layers.append(nn.BatchNorm1d(dim_out))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'leakyrelu':
                layers.append(nn.LeakyReLU())
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
    return nn.Sequential(*layers)

def format_dict(d, s, p):
    rv = ""
    for (k,v) in d.items():
        if len(rv) > 0: rv += " , "
        rv+=p+str(k) + ":" + s.format(v*100)
    return rv

def cross_entropy_loss(pred, target, ignore_index=None):
    #-x[class] + log (sum_j exp(x[j]))
    if ignore_index is not None and target == ignore_index:
        #print('no loss')
        return 0
    _x_class = - pred[target]
    pred = torch.unsqueeze(pred,1)
    #print('pred size',pred.size() )
    #max_score, max_i = torch.max(pred)
    score,_ = torch.sort(pred, 0, True)
    max_score = torch.unsqueeze(score[0],1)

    max_score_broadcast = max_score.view(-1,1).expand(pred.size())
    denom = max_score + torch.log(torch.sum(torch.exp(pred - max_score_broadcast)))

    #denom = torch.log(torch.sum(torch.exp(pred)))

    #print('target :', target, _x_class, denom)

    loss = _x_class + denom
    #print('single loss ', loss)
    return loss

def binary_cross_entropy(pred, target, p_w=10, n_w=1):
    #loss = -p_w*target*torch.log(pred) - n_w*(1-target)*torch.log(1-pred)
    loss = torch.clamp(pred, min=0) - pred * target + (1 + 9*target)*torch.log(1 + torch.exp(-torch.abs(pred)))

    #print(loss.size())

    tot_loss = torch.sum(loss)
    #print(tot_loss)

    return tot_loss

def likelihood(pred, target, ignore_index=None):
    if target == ignore_index:
        #print('no loss')
        return 0
    likelihood = pred[target]

    return likelihood

def group_features(net_):

    cnn_verb_features = list(net_.conv_verb.parameters())
    cnn_verb_feature_len = len(list(net_.conv_verb.parameters()))
    cnn_noun_features = list(net_.conv_noun.parameters())
    cnn_noun_feature_len = len(list(net_.conv_noun.parameters()))
    verb_features = list(net_.verb.parameters())
    verb_feature_len = len(list(net_.verb.parameters()))
    role_features = list(net_.parameters())[(cnn_verb_feature_len + cnn_noun_feature_len + verb_feature_len):]

    print('Network details :')
    print('\tcnn verb features :', cnn_verb_feature_len)
    print('\tcnn noun features :', cnn_noun_feature_len)
    print('\tverb features :', verb_feature_len)
    print('\trole features :', len(role_features))


    return cnn_verb_features, cnn_noun_features, verb_features, role_features

def group_features_single(net_):

    cnn_features = list(net_.conv.parameters())
    cnn_feature_len = len(list(net_.conv.parameters()))
    verb_features = list(net_.verb.parameters())
    verb_feature_len = len(list(net_.verb.parameters()))
    role_features = list(net_.parameters())[(cnn_feature_len + verb_feature_len):]

    print('Network details :')
    print('\tcnn features :', cnn_feature_len)
    print('\tverb features :', verb_feature_len)
    print('\trole features :', len(role_features))


    return cnn_features, verb_features, role_features

def group_features_vqa(net_):

    cnn_features = list(net_.conv.parameters())
    cnn_feature_len = len(list(net_.conv.parameters()))
    mac_features = list(net_.parameters())[cnn_feature_len:]

    print('Network details :')
    print('\tcnn features :', cnn_feature_len)
    print('\tmac features :', len(mac_features))

    return cnn_features, mac_features

def group_features_noun(net_):

    cnn_features = list(net_.conv.parameters())
    cnn_feature_len = len(list(net_.conv.parameters()))
    role_features = list(net_.parameters())[(cnn_feature_len):]

    print('Network details :')
    print('\tcnn features :', cnn_feature_len)
    print('\trole features :', len(role_features))


    return cnn_features, role_features

def group_features_verbqiter(net_):

    cnn_features = list(net_.conv.parameters())
    cnn_feature_len = len(list(net_.conv.parameters()))
    role_features = list(net_.role_module.parameters())
    role_feature_len = len(list(net_.role_module.parameters()))

    other_features = list(net_.parameters())[(role_feature_len + cnn_feature_len):]

    print('Network details :')
    print('\tcnn features :', cnn_feature_len)
    print('\trole features :', role_feature_len)
    print('\tvqa features :', len(other_features))


    return cnn_features, role_features, other_features

def group_features_joint_reverb(net_):
    verb_features = list(net_.verb_module.parameters())
    verb_feature_len = len(list(net_.verb_module.parameters()))
    role_ptrtrain_features = list(net_.role_module.parameters())
    role_ptrtrain_features_len = len(list(net_.role_module.parameters()))
    cnn_features = list(net_.conv.parameters())
    cnn_feature_len = len(list(net_.conv.parameters()))
    new_verb_features = list(net_.parameters())[(verb_feature_len + role_ptrtrain_features_len + cnn_feature_len):]

    return cnn_features, new_verb_features

def group_features_agent2verb(net_):

    cnn_agent_features = list(net_.conv_agent.parameters())
    cnn_agent_feature_len = len(list(net_.conv_agent.parameters()))
    cnn_verb_features = list(net_.conv_verb.parameters())
    cnn_verb_feature_len = len(list(net_.conv_verb.parameters()))
    agent_features = list(net_.agent.parameters())
    agent_feature_len = len(list(net_.agent.parameters()))
    verb_features = list(net_.parameters())[(cnn_agent_feature_len + cnn_verb_feature_len + agent_feature_len):]


    print('Network details :')
    print('\tcnn features :', cnn_agent_feature_len, cnn_verb_feature_len)
    print('\tagent features :', agent_feature_len)
    print('\tverb features :', len(verb_features))


    return cnn_agent_features, cnn_verb_features, agent_features, verb_features

def group_features_agent2verb_directcnn(net_):

    cnn_agent_features = list(net_.conv_agent.parameters())
    cnn_agent_feature_len = len(list(net_.conv_agent.parameters()))
    cnn_verb_features = list(net_.conv_verb.parameters())
    cnn_verb_feature_len = len(list(net_.conv_verb.parameters()))
    verb_features = list(net_.parameters())[(cnn_agent_feature_len + cnn_verb_feature_len ):]


    print('Network details :')
    print('\tcnn features :', cnn_agent_feature_len, cnn_verb_feature_len)
    print('\tverb features :', len(verb_features))


    return cnn_agent_features, cnn_verb_features, verb_features

def group_features_agent2verb_small(net_):

    cnn_agent_features = list(net_.conv_agent.parameters())
    cnn_agent_feature_len = len(list(net_.conv_agent.parameters()))
    cnn_verb_features = list(net_.conv_verb.parameters())
    cnn_verb_feature_len = len(list(net_.conv_verb.parameters()))
    verb_features = list(net_.parameters())[(cnn_agent_feature_len + cnn_verb_feature_len):]


    print('Network details :')
    print('\tcnn features :', cnn_agent_feature_len, cnn_verb_feature_len)
    print('\tverb features :', len(verb_features))


    return cnn_agent_features, cnn_verb_features, verb_features

def set_trainable(model, requires_grad):
    set_trainable_param(model.parameters(), requires_grad)

def set_trainable_param(parameters, requires_grad):
    for param in parameters:
        param.requires_grad = requires_grad

def get_optimizer(lr, decay, mode, cnn_features,cnn_noun_features,  verb_features, role_features):
    """ To get the optimizer
    mode 0: training from scratch
    mode 1: cnn fix, verb fix, role training
    mode 2: cnn fix, verb fine tune, role training
    mode 3: cnn finetune, verb finetune, role training
    mode 4: cnn fix, verb finetune, role finetune"""
    if mode == 0:
        set_trainable_param(cnn_features, True)
        set_trainable_param(cnn_noun_features, True)
        set_trainable_param(verb_features, True)
        set_trainable_param(role_features, True)
        optimizer = torch.optim.Adam([
            {'params': cnn_features},
            {'params': cnn_noun_features},
            {'params': verb_features},
            {'params': role_features}
        ], lr=lr, weight_decay=decay)

    elif mode == 1:
        set_trainable_param(role_features, True)
        optimizer = torch.optim.Adam([
            {'params': role_features}
        ], lr=lr, weight_decay=decay)

    elif mode == 2:
        set_trainable_param(verb_features, True)
        set_trainable_param(role_features, True)
        optimizer = torch.optim.Adam([
            {'params': verb_features, 'lr': 5e-5},
            {'params': role_features}],
            lr=1e-3)

    elif mode == 3:
        set_trainable_param(cnn_features, True)
        set_trainable_param(cnn_noun_features, True)
        set_trainable_param(verb_features, True)
        set_trainable_param(role_features, True)
        optimizer = torch.optim.Adam([
            {'params': cnn_features},
            {'params': cnn_noun_features},
            {'params': verb_features},
            {'params': role_features}
        ], lr=lr, weight_decay=decay)

    elif mode == 4:
        set_trainable_param(verb_features, True)
        set_trainable_param(role_features, True)
        optimizer = torch.optim.Adam([
            {'params': verb_features, 'lr': 5e-5},
            {'params': role_features, 'lr': 5e-5}])

    return optimizer

def get_optimizer_single(lr, decay, mode, cnn_features,  verb_features, role_features):
    """ To get the optimizer
    mode 0: training from scratch
    mode 1: cnn fix, verb fix, role training
    mode 2: cnn fix, verb fine tune, role training
    mode 3: cnn finetune, verb finetune, role training"""
    if mode == 0:
        set_trainable_param(cnn_features, True)
        set_trainable_param(verb_features, True)
        set_trainable_param(role_features, True)
        optimizer = torch.optim.Adam([
            {'params': cnn_features, 'lr': 5e-5},
            {'params': verb_features},
            {'params': role_features}
        ], lr=1e-3)

    elif mode == 1:
        set_trainable_param(role_features, True)
        optimizer = torch.optim.Adam([
            {'params': role_features}
        ], lr=lr, weight_decay=decay)

    elif mode == 2:
        set_trainable_param(verb_features, True)
        set_trainable_param(role_features, True)
        optimizer = torch.optim.Adam([
            {'params': verb_features, 'lr': 5e-5},
            {'params': role_features}],
            lr=1e-3)

    elif mode == 3:
        set_trainable_param(cnn_features, True)
        set_trainable_param(verb_features, True)
        set_trainable_param(role_features, True)
        optimizer = torch.optim.Adam([
            {'params': cnn_features},
            {'params': verb_features},
            {'params': role_features}
        ], lr=lr, weight_decay=decay)

    return optimizer

def get_optimizer_vqa(mode, cnn_features,  mac_features):
    """ To get the optimizer
    mode 0: training from scratch
    mode 1: cnn fix, v training
    mode 2: cnn fix, verb fine tune, role training
    mode 3: cnn finetune, verb finetune, role training"""
    if mode == 0:
        set_trainable_param(cnn_features, True)
        set_trainable_param(mac_features, True)
        optimizer = torch.optim.Adam([
            {'params': cnn_features, 'lr': 5e-5},
            {'params': mac_features},
        ], lr=1e-3)

    return optimizer

def get_optimizer_verb(mode, cnn_features,  verb_features):
    """ To get the optimizer
    mode 0: training from scratch
    mode 1: cnn fix, v training
    mode 2: cnn fix, verb fine tune, role training
    mode 3: cnn finetune, verb finetune, role training"""
    if mode == 0:
        set_trainable_param(cnn_features, True)
        set_trainable_param(verb_features, True)
        optimizer = torch.optim.Adam([
            {'params': cnn_features, 'lr': 5e-5},
            {'params': verb_features},
        ], lr=1e-3)

    return optimizer

def get_optimizer_agent2verb(lr, decay, mode, cnn_agent_features,  cnn_verb_features, agent_features, verb_features):
    """ To get the optimizer
    mode 0: training from scratch
    mode 1: cnn fix, v training
    mode 2: cnn fix, verb fine tune, role training
    mode 3: cnn finetune, verb finetune, role training"""
    if mode == 0:
        set_trainable_param(cnn_agent_features, True)
        set_trainable_param(agent_features, True)
        set_trainable_param(verb_features, True)
        set_trainable_param(cnn_verb_features, True)
        optimizer = torch.optim.Adam([
            {'params': cnn_verb_features, 'lr': 5e-5},
            {'params': verb_features},
        ], lr=1e-3)
    if mode == 1:
        set_trainable_param(verb_features, True)
        set_trainable_param(cnn_verb_features, True)
        optimizer = torch.optim.Adam([
            {'params': cnn_verb_features, 'lr': 5e-5},
            {'params': verb_features},
        ], lr=1e-3)
    if mode == 2:
        set_trainable_param(agent_features, True)
        set_trainable_param(verb_features, True)
        optimizer = torch.optim.Adam([
            {'params': agent_features, 'lr': 5e-5},
            {'params': verb_features},
        ], lr=1e-3)

    return optimizer

def get_optimizer_agent2verb_directagentcnn(lr, decay, mode, cnn_agent_features,  cnn_verb_features, verb_features):
    """ To get the optimizer
    mode 0: training from scratch
    mode 1: cnn fix, v training
    mode 2: cnn fix, verb fine tune, role training
    mode 3: cnn finetune, verb finetune, role training"""
    if mode == 0:
        set_trainable_param(cnn_agent_features, True)
        set_trainable_param(verb_features, True)
        set_trainable_param(cnn_verb_features, True)
        optimizer = torch.optim.Adam([
            {'params': cnn_verb_features, 'lr': 5e-5},
            {'params': verb_features},
        ], lr=1e-3)
    if mode == 1:
        set_trainable_param(verb_features, True)
        set_trainable_param(cnn_verb_features, True)
        optimizer = torch.optim.Adam([
            {'params': cnn_verb_features, 'lr': 5e-5},
            {'params': verb_features},
        ], lr=1e-3)
    if mode == 2:
        set_trainable_param(verb_features, True)
        optimizer = torch.optim.Adam([
            {'params': verb_features},
        ], lr=1e-3)

    return optimizer

def get_optimizer_agent2verbsmall(lr, decay, mode, cnn_agent_features,  cnn_verb_features, verb_features):
    """ To get the optimizer
    mode 0: training from scratch
    mode 1: cnn fix, v training
    mode 2: cnn fix, verb fine tune, role training
    mode 3: cnn finetune, verb finetune, role training"""
    if mode == 1:
        set_trainable_param(verb_features, True)
        set_trainable_param(cnn_verb_features, True)
        optimizer = torch.optim.Adam([
            {'params': cnn_verb_features, 'lr': 5e-5},
            {'params': verb_features},
        ], lr=1e-3)
    if mode == 2:
        #set_trainable_param(agent_features, True)
        set_trainable_param(verb_features, True)
        optimizer = torch.optim.Adam([
            #{'params': agent_features, 'lr': 5e-5},
            {'params': verb_features},
        ], lr=1e-3)

    return optimizer

def get_optimizer_noun(lr, decay, mode, cnn_features, role_features):
    """ To get the optimizer
    mode 0: training from scratch
    mode 1: cnn fix, verb fix, role training
    mode 2: cnn fix, verb fine tune, role training
    mode 3: cnn finetune, verb finetune, role training"""
    if mode == 0:
        set_trainable_param(cnn_features, True)
        set_trainable_param(role_features, True)
        optimizer = torch.optim.Adam([
            {'params': cnn_features},
            {'params': role_features}
        ], lr=lr, weight_decay=decay)

    elif mode == 1:
        set_trainable_param(role_features, True)
        optimizer = torch.optim.Adam([
            {'params': role_features}
        ], lr=lr, weight_decay=decay)

    elif mode == 2:
        set_trainable_param(role_features, True)
        optimizer = torch.optim.Adam([
            {'params': role_features}],
            lr=1e-3)

    elif mode == 3:
        set_trainable_param(cnn_features, True)
        set_trainable_param(role_features, True)
        optimizer = torch.optim.Adam([
            {'params': cnn_features},
            {'params': role_features}
        ], lr=lr, weight_decay=decay)

    return optimizer

def get_optimizer_frcnn_noun(lr, decay, mode, role_features):
    """ To get the optimizer
    mode 0: training from scratch
    mode 1: cnn fix, verb fix, role training
    mode 2: cnn fix, verb fine tune, role training
    mode 3: cnn finetune, verb finetune, role training"""
    if mode == 0:
        #set_trainable_param(cnn_features, True)
        set_trainable_param(role_features, True)
        optimizer = torch.optim.Adam([
            {'params': role_features}
        ], lr=1e-3)

    elif mode == 1:
        set_trainable_param(role_features, True)
        optimizer = torch.optim.Adam([
            {'params': role_features}
        ], lr=lr, weight_decay=decay)

    elif mode == 2:
        set_trainable_param(role_features, True)
        optimizer = torch.optim.Adam([
            {'params': role_features}],
            lr=1e-3)

    elif mode == 3:
        set_trainable_param(role_features, True)
        optimizer = torch.optim.Adam([
            {'params': role_features}
        ], lr=lr, weight_decay=decay)

    return optimizer

def load_net(fname, net_list, prefix_list = None):
    need_modification = False
    if prefix_list is not None and len(prefix_list) > 0:
        need_modification = True
    for i in range(0, len(net_list)):
        dict = torch.load(fname)
        try:
            for k, v in net_list[i].state_dict().items():
                #print('trying to copy :', k, v.size(), v.type())
                if need_modification:
                    k = prefix_list[i] + '.' + k
                if k in dict:
                    #print('came here')
                    param = torch.from_numpy(np.asarray(dict[k].cpu()))
                    #print('param size :', param.size(), dict[k].type())
                    v.copy_(param)
                    #print('[Copied]: {}'.format(k))
                else:
                    print('[Missed]: {}'.format(k))
        except Exception as e:
            print(e)
            pdb.set_trace()
            print ('[Loaded net not complete] Parameter[{}] Size Mismatch...'.format(k))

def load_net_deproleqa0td(fname, net_list, prefix_list = None):
    need_modification = False
    if prefix_list is not None and len(prefix_list) > 0:
        need_modification = True
    for i in range(0, len(net_list)):
        dict = torch.load(fname)
        try:
            for k, v in net_list[i].state_dict().items():
                #print('trying to copy :', k, v.size(), v.type())
                if need_modification:
                    k = prefix_list[i] + '.' + k
                if k in dict:
                    #print('came here')
                    param = torch.from_numpy(np.asarray(dict[k].cpu()))
                    #print('param size :', param.size(), dict[k].type())
                    v.copy_(param)
                    #print('[Copied]: {}'.format(k))
                elif 'last_class' in k:
                    #print('loadinggg', k)
                    k = 'verb_module' + '.' + k
                    #print('new', k)
                    param = torch.from_numpy(np.asarray(dict[k].cpu()))
                    v.copy_(param)

                else:
                    print('[Missed]: {}'.format(k))
        except Exception as e:
            print(e)
            pdb.set_trace()
            print ('[Loaded net not complete] Parameter[{}] Size Mismatch...'.format(k))

def load_net_nlpq_fuse_ft(fname, net_list, prefix_list = None):
    need_modification = False
    if prefix_list is not None and len(prefix_list) > 0:
        need_modification = True
    for i in range(0, len(net_list)):
        dict = torch.load(fname)
        try:
            for k, v in net_list[i].state_dict().items():
                #print('trying to copy :', k, v.size(), v.type())
                if need_modification:
                    k = prefix_list[i] + '.' + k

                if k == 'verb_vqa.classifier':
                    print('came here ', k)
                    k = 'verb_module' + '.' + k
                    #print('new', k)
                    param = torch.from_numpy(np.asarray(dict[k].cpu()))
                    v.copy_(param)

                elif k in dict:
                    #print('came here')
                    param = torch.from_numpy(np.asarray(dict[k].cpu()))
                    #print('param size :', param.size(), dict[k].type())
                    v.copy_(param)
                    #print('[Copied]: {}'.format(k))
                elif 'last_class' in k:
                    #print('loadinggg', k)
                    k = 'verb_module' + '.' + k
                    #print('new', k)
                    param = torch.from_numpy(np.asarray(dict[k].cpu()))
                    v.copy_(param)

                else:
                    print('[Missed]: {}'.format(k))
        except Exception as e:
            print(e)
            pdb.set_trace()
            print ('[Loaded net not complete] Parameter[{}] Size Mismatch...'.format(k))

#optimizer from transformer
class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        rate = self.factor * \
               (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))
        return rate
class CosineAnnealingWR:
    "Optim wrapper that implements rate."
    def __init__(self, alpha_0, T, M, optimizer, min_lr):
        self.optimizer = optimizer
        self.t = 0
        self.alpha_0 = alpha_0
        self.T = T
        self.M = M
        self.min_lr = min_lr

    def step(self):
        "Update parameters and rate"
        self.t += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self.alpha_0 = rate
        self.optimizer.step()
    #original
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self.t
        #print('alpha 0 :', self.alpha_0)
        rate = (self.alpha_0/2) * (math.cos((math.pi * ((step-1)%math.ceil(self.T/self.M)))/math.ceil(self.T/self.M)) + 1)
        #print('rate')
        #if step % 400 == 0:
        #print('current rate :', rate)
        if rate < self.min_lr:
            rate = self.min_lr

        return rate

class negative_expoWR:
    "Optim wrapper that implements rate."
    def __init__(self, alpha_0, T, M, optimizer, min_lr):
        self.optimizer = optimizer
        self.t = 0
        self.alpha_0 = alpha_0
        self.T = T
        self.M = M
        self.min_lr = min_lr

    def step(self):
        "Update parameters and rate"
        self.t += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self.alpha_0 = rate
        self.optimizer.step()
    #original
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self.t
        #print('alpha 0 :', self.alpha_0)
        rate = (self.alpha_0/2) * (math.exp(-(math.pi * ((step-1)%math.ceil(self.T/self.M)))/math.ceil(self.T/self.M)) + 1)

        if rate < self.min_lr:
            rate = self.min_lr
        #print('rate')
        #if step % 400 == 0:
        #print('current rate :', rate)

        return rate


