import torch
import torch.nn as nn
import torch.nn.functional as F


def l2normalize(x):
    return F.normalize(x, p=2, dim=1)

class GermanEncoder(nn.Module):

    def __init__(self, size_feature, size, size_embed=64, depth=1, dropout_p=0.0):
        super(GermanEncoderBottom, self).__init__()
        self.h0   = torch.autograd.Variable(torch.zeros(self.depth, 1, self.size))
        self.Embed  = nn.Embedding(self.size_feature, self.size_embed)
        self.Dropout = nn.Dropout(p=self.dropout_p)
        self.RNN  = nn.GRU(self.size_embed, self.size, self.depth, batch_first=True)


    def forward(self, text):
        h0 = self.h0.expand(self.depth, text.size(0), self.size).cuda()
        out, last = self.RNN(self.Dropout(self.Embed(text)), h0)
        return out


class EnglishEncoder(nn.Module):

    def __init__(self, size_feature, size, size_embed=64, depth=1, dropout_p=0.0):
        super(GermanEncoderBottom, self).__init__()
        util.autoassign(locals())
        self.h0   = torch.autograd.Variable(torch.zeros(self.depth, 1, self.size))
        self.Embed  = nn.Embedding(self.size_feature, self.size_embed)
        self.Dropout = nn.Dropout(p=self.dropout_p)
        self.RNN  = nn.GRU(self.size_embed, self.size, self.depth, batch_first=True)


    def forward(self, text):
        h0 = self.h0.expand(self.depth, text.size(0), self.size).cuda()
        out, last = self.RNN(self.Dropout(self.Embed(text)), h0)
        return out



class ImageEncoder(nn.Module):

