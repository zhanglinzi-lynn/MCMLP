import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, num_features, expansion_factor):
        super().__init__()
        num_hidden = num_features * expansion_factor
        self.fc1 = nn.Linear(num_features, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_features)

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x


class TokenMixer(nn.Module):
    def __init__(self, num_patches, expansion_factor):
        super().__init__()
        self.mlp = MLP(num_patches, expansion_factor)

    def forward(self, x):
        residual = x
        x = x.transpose(1, 2)
        x = self.mlp(x)
        x = x.transpose(1, 2)
        out = x + residual
        return out


class ChannelMixer(nn.Module):
    def __init__(self, num_features, expansion_factor):
        super().__init__()
        self.mlp = MLP(num_features, expansion_factor)

    def forward(self, x):
        residual = x
        x = self.mlp(x)
        out = x + residual
        return out


class MixerLayer(nn.Module):
    def __init__(self, num_features, num_patches, expansion_factor):
        super().__init__()
        self.token_mixer = TokenMixer(
           num_features, expansion_factor
        )
        self.channel_mixer = ChannelMixer(
            num_patches, expansion_factor
        )

    def forward(self, x):
        x = self.token_mixer(x)
        x = self.channel_mixer(x)
        return x





class Model(nn.Module):
    def __init__(self, args, data):
        super(Model, self).__init__()
        self.num_features = data.m
        self.num_patches = args.window
        self.expansion_factor = args.expansion
        self.num_layers = args.layers

        self.mixers = nn.Sequential(
            *[
                MixerLayer(self.num_patches, self.num_features, self.expansion_factor)
                for _ in range(self.num_layers)
            ]
        )
        self.use_cuda = args.cuda
        self.P = args.window;
        self.m = data.m
        self.hidR = args.hidRNN;
        self.hidC = args.hidCNN;
        self.hidS = args.hidSkip;
        self.Ck = args.CNN_kernel;
        self.skip = args.skip;
        self.pt = int((self.P - self.Ck) / self.skip)
        self.hw = args.highway_window
        # self.conv1 = nn.Conv2d(1, self.hidC, kernel_size=(self.Ck, self.m));
        self.GRU1 = nn.GRU(self.hidC, self.hidR);
        self.dropout = nn.Dropout(p=args.dropout);
        if (self.skip > 0):
            self.GRUskip = nn.GRU(self.hidC, self.hidS);
            self.linear1 = nn.Linear(self.hidR + self.skip * self.hidS, self.m);
        else:
            self.linear1 = nn.Linear(self.hidR, self.m);
        if (self.hw > 0):
            self.highway = nn.Linear(self.hw, 1);
        self.output = None;
        if (args.output_fun == 'sigmoid'):
            self.output = F.sigmoid;
        if (args.output_fun == 'tanh'):
            self.output = F.tanh;

    def forward(self, x):
        batch_size = x.size(0);
        # batch_size = 100;


        #MLP-Mixer
        c = self.mixers(x)
        # print("shape after MLP:",c.shape)


        # RNN
        r = c.permute(2, 0, 1).contiguous();
        # print("shape after RNN:", r.shape)

        _, r = self.GRU1(r);


        r = self.dropout(torch.squeeze(r, 0));

        # skip-rnn

        if (self.skip > 0):
            s = c[:, :, int(-self.pt * self.skip):].contiguous();
            s = s.view(batch_size, self.hidC, self.pt, self.skip);
            s = s.permute(2, 0, 3, 1).contiguous();
            s = s.view(self.pt, batch_size * self.skip, self.hidC);
            _, s = self.GRUskip(s);
            s = s.view(batch_size, self.skip * self.hidS);
            s = self.dropout(s);
            r = torch.cat((r, s), 1);

        res = self.linear1(r);

        # highway
        if (self.hw > 0):
            z = x[:, -self.hw:, :];
            z = z.permute(0, 2, 1).contiguous().view(-1, self.hw);
            z = self.highway(z);
            z = z.view(-1, self.m);
            res = res + z;

        if (self.output):
            res = self.output(res);
        return res;




