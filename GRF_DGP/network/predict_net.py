import torch
import torch.optim as optim

from torch import nn as nn
from torch.nn import functional as F
from torch.distributions import Categorical, Distribution, Normal
from torch.autograd import Variable

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6


# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class LayerNorm(nn.Module):
    """
    Simple 1D LayerNorm.
    """

    def __init__(self, features, center=True, scale=False, eps=1e-6):
        super().__init__()
        self.center = center
        self.scale = scale
        self.eps = eps
        if self.scale:
            self.scale_param = nn.Parameter(torch.ones(features))
        else:
            self.scale_param = None
        if self.center:
            self.center_param = nn.Parameter(torch.zeros(features))
        else:
            self.center_param = None

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        output = (x - mean) / (std + self.eps)
        if self.scale:
            output = output * self.scale_param
        if self.center:
            output = output + self.center_param
        return output


class Predict_action(nn.Module):  ###########

    def __init__(self, num_inputs, hidden_dim, num_outputs, layer_norm=True, lr=1e-3):
        super(Predict_action, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.last_fc = nn.Linear(hidden_dim, num_outputs)

        self.layer_norm = layer_norm
        if layer_norm:
            self.ln1 = LayerNorm(hidden_dim)

        self.apply(weights_init_)
        self.lr = lr

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, input):
        if self.layer_norm:
            h = F.relu(self.ln1(self.linear1(input)))
        else:
            h = F.relu(self.linear1(input))

        h = F.relu(self.linear2(h))
        h = self.last_fc(h)
        # x = torch.softmax(self.last_fc(h), dim=-1)

        return h

    def crossEntropy(self, predict_variable, target):
        log_prob = F.log_softmax(predict_variable, dim=-1)
        log_prob = -log_prob * target
        return log_prob

    def get_log_pi(self, own_variable, other_variable):
        predict_variable = self.forward(own_variable)
        log_prob = -1 * self.crossEntropy(predict_variable, other_variable)
        # log_prob = -1 * F.mse_loss(predict_variable,
        #                            other_variable, reduction='none')
        log_prob = torch.sum(log_prob, -1, keepdim=True)
        return log_prob

    def update(self, own_variable, other_variable, mask):
        if mask.sum() > 0:
            predict_variable = self.forward(own_variable)
            # loss = F.cross_entropy(predict_variable,
            #                        other_variable, reduction='none')
            loss = self.crossEntropy(predict_variable,
                                     other_variable)
            loss = loss.sum(dim=-1, keepdim=True)
            loss = (loss * mask).sum() / mask.sum()

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.)
            self.optimizer.step()


class Predict_ID(nn.Module):  ###########

    def __init__(self, num_inputs, hidden_dim, num_outputs, layer_norm=True, lr=1e-3):
        super(Predict_ID, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.last_fc = nn.Linear(hidden_dim, num_outputs)

        self.layer_norm = layer_norm
        if layer_norm:
            self.ln1 = LayerNorm(hidden_dim)

        self.apply(weights_init_)
        self.lr = lr

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, input):
        if self.layer_norm:
            h = F.relu(self.ln1(self.linear1(input)))
        else:
            h = F.relu(self.linear1(input))

        h = F.relu(self.linear2(h))
        h = self.last_fc(h)
        # x = torch.softmax(self.last_fc(h), dim=-1)

        return h

    def crossEntropy(self, predict_variable, target):
        log_prob = F.log_softmax(predict_variable, dim=-1)
        log_prob = -log_prob * target
        return log_prob

    def get_log_pi(self, own_variable, other_variable):
        predict_variable = self.forward(own_variable)
        log_prob = -1 * self.crossEntropy(predict_variable, other_variable)

        # log_prob = -1 * F.cross_entropy(predict_variable,
        #                                 other_variable, reduction='none')
        log_prob = torch.sum(log_prob, -1, keepdim=True)
        return log_prob

    def update(self, own_variable, other_variable, mask):
        if mask.sum() > 0:
            predict_variable = self.forward(own_variable)
            # loss = F.cross_entropy(predict_variable,
            #                        other_variable, reduction='none')
            loss = self.crossEntropy(predict_variable,
                                     other_variable)
            loss = loss.sum(dim=-1, keepdim=True)
            loss = (loss * mask).sum() / mask.sum()

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.)
            self.optimizer.step()


class IVF(nn.Module):

    def __init__(self, num_inputs, hidden_dim, layer_num=3, layer_norm=False):
        super(IVF, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        if layer_num == 3:
            self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.last_fc = nn.Linear(hidden_dim, 1)

        self.layer_norm = layer_norm
        self.layer_num = layer_num
        if layer_norm:
            self.ln1 = LayerNorm(hidden_dim)

        self.apply(weights_init_)

    def forward(self, input):
        if self.layer_norm:
            h = F.relu(self.ln1(self.linear1(input)))
        else:
            h = F.relu(self.linear1(input))

        if self.layer_num == 3:
            h = F.relu(self.linear2(h))
        x = self.last_fc(h)
        return x


# class Predict_ID(nn.Module):
#
#     def __init__(self, num_inputs, hidden_dim, n_agents, add_loss_item, lr=1e-3):
#         super(Predict_ID, self).__init__()
#
#         self.linear1 = nn.Linear(num_inputs, hidden_dim)
#         self.linear2 = nn.Linear(hidden_dim, hidden_dim)
#         self.last_fc = nn.Linear(hidden_dim, n_agents)
#
#         self.apply(weights_init_)
#         self.lr = lr
#         self.add_loss_item = add_loss_item
#         self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
#
#         self.CE = nn.CrossEntropyLoss()
#         self.CEP = nn.CrossEntropyLoss(reduction='none')
#
#     def forward(self, input):
#         h = F.relu(self.linear1(input))
#         h = F.relu(self.linear2(h))
#         x = torch.softmax(self.last_fc(h), dim=-1)
#         return x
#
#     def get_q_id_o(self, obs, id):
#         with torch.no_grad():
#             predict_ = self.forward(obs)
#             log_prob = -1. * \
#                 self.CEP(predict_, id *
#                          torch.ones([obs.shape[0]]).type_as(predict_).long())
#             return log_prob.detach()
#
#     def update(self, obs, id):
#         predict_ = self.forward(obs)
#         loss = self.CE(
#             predict_, id * torch.ones([obs.shape[0]]).type_as(predict_).long())
#         obs_c = obs.clone()
#         obs_c[1:] = obs[:-1]
#
#         loss += self.add_loss_item * \
#             F.mse_loss(predict_, self.forward(obs_c).detach())
#
#         self.optimizer.zero_grad()
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(self.parameters(), 1.)
#         self.optimizer.step()
class FC_VAE(nn.Module):
    '''
    Thanks to https://vxlabs.com/2017/12/08/variational-autoencoder-in-pytorch-commented-and-annotated/
    '''

    def __init__(self, input_size, NUM_Z, HIDDEN_1, batch_size, learning_rate=1e-3, CUDA=True):
        super(FC_VAE, self).__init__()
        self.input_size = input_size
        self.batch_size = batch_size
        # ENCODER
        self.fc1 = nn.Linear(input_size, HIDDEN_1)
        # rectified linear unit layer from 400 to 400
        # max(0, x)
        self.relu = nn.ReLU()
        self.fc21 = nn.Linear(HIDDEN_1, NUM_Z)  # mu layer
        self.fc22 = nn.Linear(HIDDEN_1, NUM_Z)  # logvariance layer
        # this last layer bottlenecks through ZDIMS connections

        # DECODER
        # from bottleneck to hidden 400
        self.fc3 = nn.Linear(NUM_Z, HIDDEN_1)

        self.fc4 = nn.Linear(HIDDEN_1, input_size)
        self.sigmoid = nn.Sigmoid()

        self.optim = optim.Adam(self.parameters(), lr=learning_rate)
        self.CUDA = CUDA

    def encode(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        z1 = self.fc21(out)
        z2 = self.fc22(out)
        return z1, z2

    def decode(self, z):
        out = self.fc3(z)
        out = self.relu(out)
        out = self.fc4(out)
        out = self.sigmoid(out)
        return out

    def reparameterize(self, mu, var):
        if self.training:
            std = var.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            z = eps.mul(std).add_(mu)
            return z
        else:
            return mu

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_size))
        z = self.reparameterize(mu, logvar)
        out = self.decode(z)
        return out, mu, logvar

    def predict(self, s):
        self.eval()
        data = torch.from_numpy(s).float().to(device)
        # data = Variable(s, volatile=True)
        mu, logvar = self.encode(data.view(-1, self.input_size))
        z = self.reparameterize(mu, logvar)
        return z.data.cpu().numpy()

    def loss_function(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, self.input_size))
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        KLD /= self.batch_size * self.input_size

        # BCE tries to make our reconstruction as accurate as possible
        # KLD tries to push the distributions as close as possible to unit Gaussian
        return BCE + KLD

    def train_model(self, epoch, train_loader, LOG_INTERVAL):
        self.train()
        train_loss = 0

        for batch_idx, (data, _) in enumerate(train_loader):
            data = Variable(data)
            if self.CUDA:
                data = data.cuda()
            self.optim.zero_grad()

            # push whole batch of data through VAE.forward() to get recon_loss
            recon_batch, mu, logvar = self.forward(data)
            # calculate scalar loss
            loss = self.loss_function(recon_batch, data, mu, logvar)
            # calculate the gradient of the loss w.r.t. the graph leaves
            # i.e. input variables -- by the power of pytorch!
            loss.backward()
            train_loss += loss.data[0]
            self.optim.step()
            if batch_idx % LOG_INTERVAL == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader),
                           loss.data[0] / len(data)))

        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss / len(train_loader.dataset)))
        return train_loss

    def test_model(self, epoch, test_loader):
        self.eval()
        test_loss = 0

        # each data is of BATCH_SIZE (default 128) samples
        for i, (data, _) in enumerate(test_loader):
            if self.CUDA:
                # make sure this lives on the GPU
                data = data.cuda()

            # we're only going to infer, so no autograd at all required: volatile=True
            data = Variable(data, volatile=True)
            recon_batch, mu, logvar = self.forward(data)
            test_loss += self.loss_function(recon_batch, data, mu, logvar).data[0]
            if i == 0:
                n = min(data.size(0), 8)
            # for the first 128 batch of the epoch, show the first 8 input digits
            # with right below them the reconstructed output digits

            comparison = torch.cat([data[:n],
                                    recon_batch.view(recon_batch.size()[0], 3, 64, 64)[:n]])
            save_image(comparison.data.cpu(),
                       'vae_results/reconstruction_' + str(epoch) + '.png', nrow=n)

        test_loss /= len(test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss))
        return test_loss


class VAE(nn.Module):

    def __init__(self, args, input_dim=784, h_dim=400, z_dim=20):
        # 调用父类方法初始化模块的state
        super(VAE, self).__init__()

        self.input_dim = input_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.shape0 = args.batch_size
        self.shape1 = args.n_agents
        # 编码器 ： [b, input_dim] => [b, z_dim]
        self.fc1 = nn.Linear(input_dim, h_dim)  # 第一个全连接层
        self.fc2 = nn.Linear(h_dim, z_dim)  # mu
        self.fc3 = nn.Linear(h_dim, z_dim)  # log_var

        # 解码器 ： [b, z_dim] => [b, input_dim]
        self.fc4 = nn.Linear(z_dim, h_dim)
        self.fc5 = nn.Linear(h_dim, input_dim)
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, x):
        """
        向前传播部分, 在model_name(inputs)时自动调用
        :param x: the input of our training model [b, batch_size, 1, 28, 28]
        :return: the result of our training model
        """
        batch_size = x.shape[0]  # 每一批含有的样本的个数
        # flatten  [b, batch_size, 1, 28, 28] => [b, batch_size, 784]
        # tensor.view()方法可以调整tensor的形状，但必须保证调整前后元素总数一致。view不会修改自身的数据，
        # 返回的新tensor与原tensor共享内存，即更改一个，另一个也随之改变。
        # x = x.view(batch_size, self.input_dim)  # 一行代表一个样本

        # encoder
        mu, log_var = self.encode(x)
        # reparameterization trick
        sampled_z = self.reparameterization(mu, log_var)
        # decoder
        x_hat = self.decode(sampled_z)
        # reshape
        # x_hat = x_hat.view(batch_size, 1, 28, 28)
        return x_hat, mu, log_var

    def loss_function(self, x_hat, player_obs, mu, log_var):
        """
        Calculate the loss. Note that the loss includes two parts.
        :param x_hat:
        :param player_obs:
        :param mu:
        :param log_var:
        :return: total loss, BCE and KLD of our model
        """
        # 1. the reconstruction loss.
        # We regard the MNIST as binary classification
        BCE = F.mse_loss(x_hat, player_obs, reduction='none').sum(dim=-1, keepdim=True)

        # 2. KL-divergence
        # D_KL(Q(z|X) || P(z)); calculate in closed form as both dist. are Gaussian
        # here we assume that \Sigma is a diagonal matrix, so as to simplify the computation
        KLD = 0.5 * torch.sum(torch.exp(log_var) + torch.pow(mu, 2) - 1. - log_var, axis=-1, keepdim=True)

        # 3. total loss
        loss = BCE + KLD
        return loss, BCE, KLD

    def get_log_pi(self, player_obs):
        # encoder
        mu, log_var = self.encode(player_obs)
        # reparameterization trick
        log_var = torch.exp(log_var * 0.5)

        # sampled_z = self.reparameterization(mu, log_var).permute(0, 2, 1, 3)
        # dst = sampled_z.repeat(1, 1, 1, self.shape1).view(sampled_z.shape[0], sampled_z.shape[1], -1,
        #                                                   self.z_dim) - sampled_z.repeat(1, 1, self.shape1, 1)
        # dst = torch.norm(dst, dim=-1).reshape(sampled_z.shape[0], sampled_z.shape[1], self.shape1, self.shape1).sum(
        #     dim=-1, keepdim=True).permute(0, 2, 1, 3)

        mu = mu.permute(0, 2, 1, 3)
        log_var = log_var.permute(0, 2, 1, 3)
        dst_1 = mu.repeat(1, 1, 1, self.shape1).view(mu.shape[0], mu.shape[1], -1,
                                                     self.z_dim) - mu.repeat(1, 1, self.shape1, 1)
        dst_1 = torch.sum(dst_1 ** 2, dim=-1)
        dst_2 = log_var.repeat(1, 1, 1, self.shape1).view(log_var.shape[0], log_var.shape[1], -1,
                                                          self.z_dim) - log_var.repeat(1, 1, self.shape1, 1)
        dst_2 = torch.sum(dst_2 ** 2, dim=-1)
        dst = (dst_1 + dst_2).reshape(dst_1.shape[0], dst_1.shape[1], self.shape1, self.shape1).sum(
            dim=-1, keepdim=True).permute(0, 2, 1, 3)

        return dst

    def update(self, player_obs, mask):
        if mask.sum() > 0:
            x_hat, mu, log_var = self.forward(player_obs)  # 模型的输出，在这里会自动调用model中的forward函数
            loss, BCE, KLD = self.loss_function(x_hat, player_obs, mu, log_var)  # 计算损失值，即目标函数

            loss = (loss * mask).sum() / mask.sum()
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.)
            self.optimizer.step()

    def encode(self, x):
        """
        encoding part
        :param x: input image
        :return: mu and log_var
        """
        h = F.relu(self.fc1(x))
        mu = self.fc2(h)
        log_var = self.fc3(h)

        return mu, log_var

    def reparameterization(self, mu, log_var):
        """
        Given a standard gaussian distribution epsilon ~ N(0,1),
        we can sample the random variable z as per z = mu + sigma * epsilon
        :param mu:
        :param log_var:
        :return: sampled z
        """
        sigma = torch.exp(log_var * 0.5)
        eps = torch.randn_like(sigma)
        return mu + sigma * eps  # 这里的“*”是点乘的意思

    def decode(self, z):
        """
        Given a sampled z, decode it back to image
        :param z:
        :return:
        """
        h = F.relu(self.fc4(z))
        x_hat = torch.sigmoid(self.fc5(h))  # 图片数值取值为[0,1]，不宜用ReLU
        return x_hat



class Predict_Network1(nn.Module):

    def __init__(self, num_inputs, hidden_dim, num_outputs, layer_norm=True, lr=1e-3):
        super(Predict_Network1, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.last_fc = nn.Linear(hidden_dim, num_outputs)

        self.layer_norm = layer_norm
        if layer_norm:
            self.ln1 = LayerNorm(hidden_dim)

        self.apply(weights_init_)
        self.lr = lr

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, input):
        if self.layer_norm:
            h = F.relu(self.ln1(self.linear1(input)))
        else:
            h = F.relu(self.linear1(input))

        h = F.relu(self.linear2(h))
        x = self.last_fc(h)
        return x

    def get_log_pi(self, own_variable, other_variable):
        predict_variable = self.forward(own_variable)
        log_prob = -1 * F.mse_loss(predict_variable,
                                   other_variable, reduction='none')
        log_prob = torch.sum(log_prob, -1, keepdim=True)
        return log_prob

    def update(self, own_variable, other_variable, mask):
        if mask.sum() > 0:
            predict_variable = self.forward(own_variable)
            loss = F.mse_loss(predict_variable,
                              other_variable, reduction='none')
            loss = loss.sum(dim=-1, keepdim=True)
            loss = (loss * mask).sum() / mask.sum()

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.)
            self.optimizer.step()


class Predict_Network1_combine(nn.Module):

    def __init__(self, num_inputs, hidden_dim, num_outputs, n_agents, layer_norm=True, lr=1e-3):
        super(Predict_Network1_combine, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim + n_agents, hidden_dim)
        self.last_fc = nn.Linear(hidden_dim, num_outputs)

        self.layer_norm = layer_norm
        if layer_norm:
            self.ln1 = LayerNorm(hidden_dim)

        self.apply(weights_init_)
        self.lr = lr
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, input, add_id):
        if self.layer_norm:
            h = F.relu(self.ln1(self.linear1(input)))
        else:
            h = F.relu(self.linear1(input))

        h = torch.cat([h, add_id], dim=-1)
        h = F.relu(self.linear2(h))
        x = self.last_fc(h)
        return x

    def get_log_pi(self, own_variable, other_variable, add_id):
        predict_variable = self.forward(own_variable, add_id)
        log_prob = -1 * F.mse_loss(predict_variable,
                                   other_variable, reduction='none')
        log_prob = torch.sum(log_prob, -1, keepdim=True)
        return log_prob

    def update(self, own_variable, other_variable, add_id, mask):
        if mask.sum() > 0:
            predict_variable = self.forward(own_variable, add_id)
            loss = F.mse_loss(predict_variable,
                              other_variable, reduction='none')
            loss = loss.sum(dim=-1, keepdim=True)
            loss = (loss * mask).sum() / mask.sum()

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.)
            self.optimizer.step()

class Predict_mse(nn.Module):  ###########

    def __init__(self, num_inputs, hidden_dim, num_outputs, layer_norm=True, lr=1e-3):
        super(Predict_mse, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.last_fc = nn.Linear(hidden_dim, num_outputs)

        self.layer_norm = layer_norm
        if layer_norm:
            self.ln1 = LayerNorm(hidden_dim)

        self.apply(weights_init_)
        self.lr = lr

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, input):
        if self.layer_norm:
            h = F.relu(self.ln1(self.linear1(input)))
        else:
            h = F.relu(self.linear1(input))

        h = F.relu(self.linear2(h))
        x = self.last_fc(h)
        #x = torch.softmax(self.last_fc(h), dim=-1)

        return x

    def get_log_pi(self, own_variable, other_variable):
        predict_variable = self.forward(own_variable)
        log_prob = -1 * F.mse_loss(predict_variable,
                                   other_variable, reduction='none')
        log_prob = torch.sum(log_prob, -1, keepdim=True)
        return log_prob

    def update(self, own_variable, other_variable, mask):
        if mask.sum() > 0:
            predict_variable = self.forward(own_variable)
            loss = F.mse_loss(predict_variable,
                              other_variable, reduction='none')
            loss = loss.sum(dim=-1, keepdim=True)
            loss = (loss * mask).sum() / mask.sum()

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.)
            self.optimizer.step()


class Predict_combine_mse(nn.Module):  ##########

    def __init__(self, num_inputs, hidden_dim, num_outputs, n_agents, layer_norm=True, lr=1e-3):
        super(Predict_combine_mse, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim + n_agents, hidden_dim)
        self.last_fc = nn.Linear(hidden_dim, num_outputs)

        self.layer_norm = layer_norm
        if layer_norm:
            self.ln1 = LayerNorm(hidden_dim)

        self.apply(weights_init_)
        self.lr = lr
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, input, add_id):
        if self.layer_norm:
            h = F.relu(self.ln1(self.linear1(input)))
        else:
            h = F.relu(self.linear1(input))

        h = torch.cat([h, add_id], dim=-1)
        h = F.relu(self.linear2(h))
        x = self.last_fc(h)
        #x = torch.softmax(self.last_fc(h), dim=-1)

        return x

    def get_log_pi(self, own_variable, other_variable, add_id):
        predict_variable = self.forward(own_variable, add_id)
        log_prob = -1 * F.mse_loss(predict_variable,
                                   other_variable, reduction='none')
        log_prob = torch.sum(log_prob, -1, keepdim=True)
        return log_prob

    def get_log_pi_plus(self, own_variable, other_variable, add_id):
        predict_variable = self.forward(own_variable, add_id)
        log_prob = -1 * F.mse_loss(predict_variable,
                                   other_variable, reduction='none')
        log_prob = torch.sum(log_prob, -1, keepdim=True)
        return log_prob, predict_variable

    def update(self, own_variable, other_variable, add_id, mask):
        if mask.sum() > 0:
            predict_variable = self.forward(own_variable, add_id)
            loss = F.mse_loss(predict_variable,
                              other_variable, reduction='none')
            loss = loss.sum(dim=-1, keepdim=True)
            loss = (loss * mask).sum() / mask.sum()

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.)
            self.optimizer.step()


class Predict_Network2(nn.Module):

    def __init__(self, num_inputs, hidden_dim, num_components=4, layer_norm=True, lr=1e-3):
        super(Predict_Network2, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.layer_norm = layer_norm
        if layer_norm:
            self.ln1 = LayerNorm(hidden_dim)

        self.mean_list = []
        for _ in range(num_components):
            self.mean_list.append(nn.Linear(hidden_dim, num_inputs))

        self.mean_list = nn.ModuleList(self.mean_list)
        self.num_components = num_components
        self.com_last_fc = nn.Linear(hidden_dim, num_components)

        self.apply(weights_init_)
        self.lr = lr

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, input):

        if self.layer_norm:
            x1 = F.relu(self.ln1(self.linear1(input)))
        else:
            x1 = F.relu(self.linear1(input))

        x2 = F.relu(self.linear2(x1))
        com_h = torch.softmax(self.com_last_fc(x2), dim=-1)

        means, stds = [], []
        for i in range(self.num_components):
            mean = self.mean_list[i](x2)
            means.append(mean)
            stds.append(torch.ones_like(mean))

        return com_h, means, stds

    def get_log_pi(self, own_variable, other_variable):
        com_h, means, stds = self.forward(own_variable)
        mix = Categorical(logits=com_h)
        means = torch.stack(means, 1)
        stds = torch.stack(stds, 1)

        comp = torch.distributions.independent.Independent(
            Normal(means, stds), 1)
        gmm = torch.distributions.mixture_same_family.MixtureSameFamily(
            mix, comp)

        return gmm.log_prob(other_variable)


class Predict_Network3(nn.Module):

    def __init__(self, num_inputs, hidden_dim, num_components=4, layer_norm=True, lr=1e-3):
        super(Predict_Network3, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.layer_norm = layer_norm
        if layer_norm:
            self.ln1 = LayerNorm(hidden_dim)

        self.mean_list = []
        for _ in range(num_components):
            self.mean_list.append(nn.Linear(hidden_dim, num_inputs))

        self.log_std_list = []
        for _ in range(num_components):
            self.log_std_list.append(nn.Linear(hidden_dim, num_inputs))

        self.mean_list = nn.ModuleList(self.mean_list)
        self.log_std_list = nn.ModuleList(self.log_std_list)
        self.num_components = num_components
        self.com_last_fc = nn.Linear(hidden_dim, num_components)

        self.apply(weights_init_)
        self.lr = lr

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, input):

        if self.layer_norm:
            x1 = F.relu(self.ln1(self.linear1(input)))
        else:
            x1 = F.relu(self.linear1(input))

        x2 = F.relu(self.linear2(x1))
        com_h = torch.softmax(self.com_last_fc(x2), dim=-1)

        means, stds = [], []
        for i in range(self.num_components):
            mean = self.mean_list[i](x2)
            log_std = self.log_std_list[i](x2)

            means.append(mean)
            stds.append(log_std.exp())

        return com_h, means, stds

    def get_log_pi(self, own_variable, other_variable):
        com_h, means, stds = self.forward(own_variable)
        mix = Categorical(logits=com_h)
        means = torch.stack(means, 1)
        stds = torch.stack(stds, 1)

        comp = torch.distributions.independent.Independent(
            Normal(means, stds), 1)
        gmm = torch.distributions.mixture_same_family.MixtureSameFamily(
            mix, comp)

        return gmm.log_prob(other_variable)


def get_predict_model(num_inputs, hidden_dim, model_id, layer_norm=True):
    if model_id == 1:
        return Predict_Network1(num_inputs, hidden_dim, layer_norm=layer_norm)
    elif model_id == 2:
        return Predict_Network2(num_inputs, hidden_dim, layer_norm=layer_norm)
    elif model_id == 3:
        return Predict_Network3(num_inputs, hidden_dim, layer_norm=layer_norm)
    else:
        raise (print('error predict model'))
