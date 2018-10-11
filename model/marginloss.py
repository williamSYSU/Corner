import torch
import torch.nn as nn


class MaxMargin(nn.Module):
    def __init__(self, opt):
        super(MaxMargin, self).__init__()
        self.opt = opt

    def forward(self, z_s, z_n, r_s, eps=1e-06):

        # z_s = z_s / K.cast(K.epsilon() + K.sqrt(K.sum(K.square(z_s), axis=-1, keepdims=True)), K.floatx())
        # z_n = z_n / K.cast(K.epsilon() + K.sqrt(K.sum(K.square(z_n), axis=-1, keepdims=True)), K.floatx())
        # r_s = r_s / K.cast(K.epsilon() + K.sqrt(K.sum(K.square(r_s), axis=-1, keepdims=True)), K.floatx())

        div = eps + torch.norm(z_s, 2, -1)
        # if 0 in div:
        #     print("zs there is 0")
        div = div.view(-1, 1)
        z_s = torch.div(z_s, div)

        div = eps + torch.norm(z_n, 2, -1)
        # if 0 in div.view(-1):
        #     print("zn there is 0")
        div = div.view(z_n.size(0), z_n.size(1), 1)
        z_n = torch.div(z_n, div)

        # div = eps + torch.sqrt(torch.sum(r_s ** 2, dim=-1))
        div = eps + torch.norm(r_s, 2, -1)
        # if 0 in div:
        #     print("rs there is 0")
        div = div.view(-1, 1)
        r_s = torch.div(r_s, div)

        z_s = z_s.float()
        z_n = z_n.float()
        r_s = r_s.float()

        steps = self.opt.neg_size

        pos = torch.sum(z_s * r_s, dim=-1, keepdim=False)
        pos = torch.unsqueeze(pos, dim=1).expand(-1, steps)
        # pos = K.repeat_elements(pos, steps, axis=-1)
        r_s = torch.unsqueeze(r_s, dim=1).expand(-1, steps, -1)
        # r_s = K.expand_dims(r_s, dim=-2)
        # r_s = K.repeat_elements(r_s, steps, axis=1)
        neg = torch.sum(z_n * r_s, dim=-1)

        loss = torch.sum(torch.max(
            torch.zeros(self.opt.batch_size, 1).to(self.opt.device),
            (torch.ones(self.opt.batch_size, 1).to(self.opt.device) - pos + neg)),
            dim=-1)
        loss = torch.sum(loss, dim=0)
        return loss
