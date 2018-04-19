# Implemented by Dingquan Li
# Email: dingquanli@pku.edu.cn
# Date: 2018/4/19
#
# TODO: hyper-parameters to config
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNIQAnet(nn.Module):
    def __init__(self, train=True, ker_size=7, n_kers=50, n1_nodes=800, n2_nodes=800):
        super(CNNIQAnet, self).__init__()
        self.conv1  = nn.Conv2d(1, n_kers, ker_size)
        self.fc1    = nn.Linear(2 * n_kers, n1_nodes)
        self.fc2    = nn.Linear(n1_nodes, n2_nodes)
        self.fc3  = nn.Linear(n2_nodes, 1)

    def forward(self, x, train=True):

        h = self.conv1(x)

        h1 = F.adaptive_max_pool2d(h, 1)
        h2 = -F.adaptive_max_pool2d(-h, 1)
        h = torch.cat((h1,h2),1) #
        h = torch.squeeze(torch.squeeze(h,3),2)

        h = F.relu(self.fc1(h))
        h = F.dropout(h, p=0.5, training=train)
        h = F.relu(self.fc2(h))

        q = self.fc3(h)
        return q