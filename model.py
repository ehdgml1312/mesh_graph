import torch
from torch_geometric.nn import GCNConv, TopKPooling, ASAPooling, LEConv, EdgeConv
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import spspmm
from torch_geometric.utils import (add_self_loops, sort_edge_index,
                                   remove_self_loops)
from torch_geometric.utils.repeat import repeat
from torch.distributions import Normal, Independent, kl
from torch_scatter import scatter_mean, scatter_max
from point_trans import PointTransformerConv


def readout(x, batch):
    x_mean = scatter_mean(x, batch, dim=0)
    x_max, _ = scatter_max(x, batch, dim=0)
    return torch.cat((x_mean, x_max), dim=-1)

class autoencoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, batch):
        super(autoencoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x).relu()
        return self.conv2(x)

class MLPEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, batch):
        super(MLPEncoder, self).__init__()
        self.conv1 = nn.Linear(in_channels, 2 * out_channels)
        self.conv_mu = nn.Linear(2 * out_channels, out_channels)
        self.conv_logstd = nn.Linear(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x).relu()
        return self.conv_mu(x), self.conv_logstd(x)

class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv_mu = GCNConv(2 * out_channels, out_channels)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

class asap(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(asap, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, 2 * out_channels)

        self.pool1 = ASAPooling(2 * out_channels, GNN=LEConv)
        self.pool2 = ASAPooling(2 * out_channels, GNN=LEConv)

        self.lin1 = nn.Linear(2 * out_channels * 2, 2 * out_channels)
        self.lin2 = nn.Linear(2 * out_channels, out_channels)
        self.lin3 = nn.Linear(2 * out_channels, out_channels)


    def forward(self, x, edge_index, edge_weight = None, batch = None):
        num = len(x)
        x = self.conv1(x, edge_index)
        x, edge_index, _, batch, _ = self.pool1(x, edge_index, edge_weight, batch)
        x1 = readout(x, batch)

        x = self.conv2(x, edge_index)
        x, edge_index, _, batch, _ = self.pool2(x, edge_index, edge_weight, batch)
        x2 = readout(x, batch)

        # x = self.conv3(x, edge_index)
        # x, edge_index, _, batch, _ = self.pool3(x, edge_index, edge_weight, batch)
        # x3 = readout(x, batch)

        x = x1 + x2
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        mu = self.lin2(x)
        logstd = self.lin3(x)
        return mu.repeat(num,1), logstd.repeat(num,1)

class avg(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(avg, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv_mu = GCNConv(2 * out_channels, out_channels)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        mu = torch.mean(self.conv_mu(x, edge_index),0)
        logstd = torch.mean(self.conv_logstd(x, edge_index),0)
        return mu.repeat(len(x),1), logstd.repeat(len(x),1)

class GraphUnet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_classes,
                 pool_ratios=0.5, sum_res=True, act=F.relu):
        super(GraphUnet, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.depth = len(hidden_channels)
        self.num_classes = num_classes
        self.pool_ratios = repeat(pool_ratios, self.depth)
        self.act = act
        self.sum_res = sum_res

        channels = hidden_channels

        self.down_convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        self.down_convs.append(GCNConv(in_channels, channels[0], improved=True))
        for i in range(self.depth):
            self.pools.append(TopKPooling(channels[i], self.pool_ratios[i]))
            if i == self.depth - 1:  # bottom layer
                self.down_convs.append(GCNConv(channels[i], channels[i], improved=True))
            else:
                self.down_convs.append(GCNConv(channels[i], channels[i + 1], improved=True))

        self.up_convs = torch.nn.ModuleList()
        for i in range(self.depth - 1):
            self.up_convs.append(GCNConv(2 * channels[self.depth - i - 1], channels[self.depth - i - 2], improved=True))
        self.up_convs.append(GCNConv(2 * channels[0], out_channels, improved=True))

        self.reset_parameters()

        self.decode = nn.Linear(out_channels , num_classes)

    def reset_parameters(self):
        for conv in self.down_convs:
            conv.reset_parameters()
        for pool in self.pools:
            pool.reset_parameters()
        for conv in self.up_convs:
            conv.reset_parameters()

    def forward(self, data, batch=None):
        """"""
        x, edge_index = data.x[:,:self.in_channels], data.edge_index

        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        edge_weight = x.new_ones(edge_index.size(1))

        x = self.down_convs[0](x, edge_index, edge_weight)
        x = self.act(x)

        xs = [x]
        edge_indices = [edge_index]
        edge_weights = [edge_weight]
        perms = []

        for i in range(1, self.depth + 1):
            edge_index, edge_weight = self.augment_adj(edge_index, edge_weight,
                                                       x.size(0))
            x, edge_index, edge_weight, batch, perm, _ = self.pools[i - 1](
                x, edge_index, edge_weight, batch)

            x = self.down_convs[i](x, edge_index, edge_weight)
            x = self.act(x)

            if i < self.depth:
                xs += [x]
                edge_indices += [edge_index]
                edge_weights += [edge_weight]
            perms += [perm]

        for i in range(self.depth):
            j = self.depth - 1 - i

            res = xs[j]
            edge_index = edge_indices[j]
            edge_weight = edge_weights[j]
            perm = perms[j]

            up = torch.zeros_like(res)
            up[perm] = x
            x = res + up if self.sum_res else torch.cat((res, up), dim=-1)

            x = self.up_convs[i](x, edge_index, edge_weight)
            x = self.act(x) if i < self.depth - 1 else x

        x = self.decode(x)

        return x


    def augment_adj(self, edge_index, edge_weight, num_nodes):
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                                 num_nodes=num_nodes)
        edge_index, edge_weight = sort_edge_index(edge_index, edge_weight,
                                                  num_nodes)
        edge_index, edge_weight = spspmm(edge_index, edge_weight, edge_index,
                                         edge_weight, num_nodes, num_nodes,
                                         num_nodes)
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        return edge_index, edge_weight


    def __repr__(self):
        return '{}({}, {}, {}, depth={}, pool_ratios={})'.format(
            self.__class__.__name__, self.in_channels, self.hidden_channels,
            self.out_channels, self.depth, self.pool_ratios)

class TransUnet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_classes,
                 pool_ratios, sum_res=True, act=F.relu):
        super(TransUnet, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.depth = len(hidden_channels)
        self.num_classes = num_classes
        self.pool_ratios = repeat(pool_ratios, self.depth)
        self.act = act
        self.sum_res = sum_res

        channels = hidden_channels

        self.down_convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        self.down_convs.append(PointTransformerConv(in_channels, channels[0]))
        for i in range(self.depth):
            self.pools.append(TopKPooling(channels[i], self.pool_ratios[i]))
            if i == self.depth - 1:  # bottom layer
                self.down_convs.append(PointTransformerConv(channels[i], channels[i]))
            else:
                self.down_convs.append(PointTransformerConv(channels[i], channels[i + 1]))

        self.up_convs = torch.nn.ModuleList()
        for i in range(self.depth - 1):
            self.up_convs.append(PointTransformerConv(2 * channels[self.depth - i - 1], channels[self.depth - i - 2]))
        self.up_convs.append(PointTransformerConv(2 * channels[0], out_channels))

        self.reset_parameters()

        self.decode = nn.Linear(out_channels , num_classes)

    def reset_parameters(self):
        for conv in self.down_convs:
            conv.reset_parameters()
        for pool in self.pools:
            pool.reset_parameters()
        for conv in self.up_convs:
            conv.reset_parameters()

    def forward(self, data, batch=None):
        """"""
        x, edge_index = data.x[:,:self.in_channels], data.edge_index

        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        edge_weight = x.new_ones(edge_index.size(1))

        x = self.down_convs[0](x, edge_index)
        x = self.act(x)

        xs = [x]
        edge_indices = [edge_index]
        edge_weights = [edge_weight]
        perms = []

        for i in range(1, self.depth + 1):
            edge_index, edge_weight = self.augment_adj(edge_index, edge_weight,
                                                       x.size(0))
            x, edge_index, edge_weight, batch, perm, _ = self.pools[i - 1](
                x, edge_index, edge_weight, batch)

            x = self.down_convs[i](x, edge_index)
            x = self.act(x)

            if i < self.depth:
                xs += [x]
                edge_indices += [edge_index]
                edge_weights += [edge_weight]
            perms += [perm]

        for i in range(self.depth):
            j = self.depth - 1 - i

            res = xs[j]
            edge_index = edge_indices[j]
            edge_weight = edge_weights[j]
            perm = perms[j]

            up = torch.zeros_like(res)
            up[perm] = x
            x = res + up if self.sum_res else torch.cat((res, up), dim=-1)

            x = self.up_convs[i](x, edge_index)
            x = self.act(x) if i < self.depth - 1 else x

        x = self.decode(x)

        return x


    def augment_adj(self, edge_index, edge_weight, num_nodes):
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                                 num_nodes=num_nodes)
        edge_index, edge_weight = sort_edge_index(edge_index, edge_weight,
                                                  num_nodes)
        edge_index, edge_weight = spspmm(edge_index, edge_weight, edge_index,
                                         edge_weight, num_nodes, num_nodes,
                                         num_nodes)
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        return edge_index, edge_weight


    def __repr__(self):
        return '{}({}, {}, {}, depth={}, pool_ratios={})'.format(
            self.__class__.__name__, self.in_channels, self.hidden_channels,
            self.out_channels, self.depth, self.pool_ratios)

class EdgeUnet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_classes,
                 pool_ratios=0.5, sum_res=True, act=F.relu):
        super(EdgeUnet, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.depth = len(hidden_channels)
        self.num_classes = num_classes
        self.pool_ratios = repeat(pool_ratios, self.depth)
        self.act = act
        self.sum_res = sum_res

        channels = hidden_channels

        self.down_convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        self.down_convs.append(EdgeConv(nn.Linear(2*in_channels, channels[0]), aggr='max'))
        for i in range(self.depth):
            self.pools.append(TopKPooling(channels[i], self.pool_ratios[i]))
            if i == self.depth - 1:  # bottom layer
                self.down_convs.append(EdgeConv(nn.Linear(2*channels[i], channels[i]), aggr='max'))
            else:
                self.down_convs.append(EdgeConv(nn.Linear(2*channels[i], channels[i + 1]), aggr='max'))

        self.up_convs = torch.nn.ModuleList()
        for i in range(self.depth - 1):
            self.up_convs.append(EdgeConv(nn.Linear(2* 2 * channels[self.depth - i - 1], channels[self.depth - i - 2]), aggr='max'))
        self.up_convs.append(EdgeConv(nn.Linear(2* 2 * channels[0], out_channels), aggr='max'))

        self.reset_parameters()

        self.decode = nn.Linear(out_channels , num_classes)

    def reset_parameters(self):
        for conv in self.down_convs:
            conv.reset_parameters()
        for pool in self.pools:
            pool.reset_parameters()
        for conv in self.up_convs:
            conv.reset_parameters()

    def forward(self, data, batch=None):
        """"""
        x, edge_index = data.x[:,:self.in_channels], data.edge_index

        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        edge_weight = x.new_ones(edge_index.size(1))

        x = self.down_convs[0](x, edge_index)
        x = self.act(x)

        xs = [x]
        edge_indices = [edge_index]
        edge_weights = [edge_weight]
        perms = []

        for i in range(1, self.depth + 1):
            edge_index, edge_weight = self.augment_adj(edge_index, edge_weight,
                                                       x.size(0))
            x, edge_index, edge_weight, batch, perm, _ = self.pools[i - 1](
                x, edge_index, edge_weight, batch)

            x = self.down_convs[i](x, edge_index)
            x = self.act(x)

            if i < self.depth:
                xs += [x]
                edge_indices += [edge_index]
                edge_weights += [edge_weight]
            perms += [perm]

        for i in range(self.depth):
            j = self.depth - 1 - i

            res = xs[j]
            edge_index = edge_indices[j]
            edge_weight = edge_weights[j]
            perm = perms[j]

            up = torch.zeros_like(res)
            up[perm] = x
            x = res + up if self.sum_res else torch.cat((res, up), dim=-1)

            x = self.up_convs[i](x, edge_index)
            x = self.act(x) if i < self.depth - 1 else x

        x = self.decode(x)

        return x


    def augment_adj(self, edge_index, edge_weight, num_nodes):
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                                 num_nodes=num_nodes)
        edge_index, edge_weight = sort_edge_index(edge_index, edge_weight,
                                                  num_nodes)
        edge_index, edge_weight = spspmm(edge_index, edge_weight, edge_index,
                                         edge_weight, num_nodes, num_nodes,
                                         num_nodes)
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        return edge_index, edge_weight


    def __repr__(self):
        return '{}({}, {}, {}, depth={}, pool_ratios={})'.format(
            self.__class__.__name__, self.in_channels, self.hidden_channels,
            self.out_channels, self.depth, self.pool_ratios)

class ProbGraphUnet(torch.nn.Module):
    def __init__(self, config):
        super(ProbGraphUnet, self).__init__()
        assert len(config.hidden_channels) >= 1
        self.in_channels = config.in_channels
        self.hidden_channels = config.hidden_channels
        self.out_channels = config.out_channels
        self.latent_channels = config.latent_channels
        self.depth = len(config.hidden_channels)
        self.num_classes = config.num_classes
        self.pool_ratios = config.pool_ratios
        self.sum_res = config.sum_res
        self.act = F.relu
        self.num_samples = config.num_samples

        if config.encoder == 'asap':
            self.encoder = asap
        else:
            self.encoder = avg

        self.prior = self.encoder(config.in_channels, config.latent_channels)
        self.posterior = self.encoder(config.in_channels + config.num_classes, config.latent_channels)

        channels = self.hidden_channels


        self.down_convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        self.down_convs.append(GCNConv(self.in_channels, channels[0], improved=True))
        for i in range(self.depth):
            self.pools.append(TopKPooling(channels[i], self.pool_ratios[i]))
            if i == self.depth - 1:  # bottom layer
                self.down_convs.append(GCNConv(channels[i], channels[i], improved=True))
            else:
                self.down_convs.append(GCNConv(channels[i], channels[i + 1], improved=True))

        self.up_convs = torch.nn.ModuleList()
        for i in range(self.depth - 1):
            self.up_convs.append(GCNConv(2 * channels[self.depth - i - 1], channels[self.depth - i - 2], improved=True))
        self.up_convs.append(GCNConv(2 * channels[0], self.out_channels, improved=True))

        self.reset_parameters()

        self.decode = nn.Linear(self.out_channels + self.latent_channels, self.num_classes)

    def reset_parameters(self):
        for conv in self.down_convs:
            conv.reset_parameters()
        for pool in self.pools:
            pool.reset_parameters()
        for conv in self.up_convs:
            conv.reset_parameters()

    def forward(self, data, batch = None):
        """"""
        x, edge_index = data.x, data.edge_index

        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        edge_weight = x.new_ones(edge_index.size(1))

        x_vae = x

        x = self.down_convs[0](x, edge_index, edge_weight)
        x = self.act(x)

        xs = [x]
        edge_indices = [edge_index]
        edge_weights = [edge_weight]
        perms = []

        for i in range(1, self.depth + 1):
            edge_index, edge_weight = self.augment_adj(edge_index, edge_weight,
                                                       x.size(0))
            x, edge_index, edge_weight, batch, perm, _ = self.pools[i - 1](
                x, edge_index, edge_weight, batch)

            x = self.down_convs[i](x, edge_index, edge_weight)
            x = self.act(x)

            if i < self.depth:
                xs += [x]
                edge_indices += [edge_index]
                edge_weights += [edge_weight]
            perms += [perm]

        for i in range(self.depth):
            j = self.depth - 1 - i

            res = xs[j]
            edge_index = edge_indices[j]
            edge_weight = edge_weights[j]
            perm = perms[j]

            up = torch.zeros_like(res)
            up[perm] = x
            x = res + up if self.sum_res else torch.cat((res, up), dim=-1)

            x = self.up_convs[i](x, edge_index, edge_weight)
            x = self.act(x) if i < self.depth - 1 else x
            x_gu = x # Graph Unet feature map

        seg = []
        mu_pri, logstd_pri = self.prior(x_vae, edge_index)
        self.prior_space = Independent(Normal(loc=mu_pri, scale=torch.exp(logstd_pri)), 1)
        if self.training:
            y = data.y
            y = F.one_hot(y, self.num_classes)
            x_pos = torch.cat((x_vae, y), 1)

            mu_pos, logstd_pos = self.posterior(x_pos, edge_index)
            self.posterior_space = Independent(Normal(loc=mu_pos, scale=torch.exp(logstd_pos)), 1)
            z_pos = self.posterior_space.rsample()

            x = torch.cat((x_gu, z_pos), 1)

            x = self.decode(x)
            seg.append(x)

        else:
            for i in range(self.num_samples):
                z_pri = self.prior_space.rsample()

                x = torch.cat((x_gu, z_pri), 1)

                x = self.decode(x)
                seg.append(x)

        return seg

    def augment_adj(self, edge_index, edge_weight, num_nodes):
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                                 num_nodes=num_nodes)
        edge_index, edge_weight = sort_edge_index(edge_index, edge_weight,
                                                  num_nodes)
        edge_index, edge_weight = spspmm(edge_index, edge_weight, edge_index,
                                         edge_weight, num_nodes, num_nodes,
                                         num_nodes)
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        return edge_index, edge_weight

    def kl_div(self):
        return kl.kl_divergence(self.posterior_space, self.prior_space)

    def __repr__(self):
        return '{}({}, {}, {}, depth={}, pool_ratios={})'.format(
            self.__class__.__name__, self.in_channels, self.hidden_channels,
            self.out_channels, self.depth, self.pool_ratios)
