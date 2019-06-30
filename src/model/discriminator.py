from torch import nn


class Discriminator(nn.Module):

    DIS_ATTR = ['input_dim', 'dis_layers', 'dis_hidden_dim', 'dis_dropout']

    def __init__(self, params):
        """
        Discriminator initialization.
        """
        super(Discriminator, self).__init__()

        self.is_parallel = 2
        self.input_dim = params.emb_dim * 2 
        self.dis_layers = params.dis_layers
        self.dis_hidden_dim = params.dis_hidden_dim
        self.dis_dropout = params.dis_dropout

        layers = []
        for i in range(self.dis_layers + 1):
            if i == 0:
                input_dim = self.input_dim
            else:
                input_dim = self.dis_hidden_dim
            output_dim = self.dis_hidden_dim if i < self.dis_layers else self.is_parallel
            layers.append(nn.Linear(input_dim, output_dim))
            if i < self.dis_layers:
                layers.append(nn.LeakyReLU(0.2))
                layers.append(nn.Dropout(self.dis_dropout))
        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        return self.layers(input)