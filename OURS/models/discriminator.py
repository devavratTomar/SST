import torch.nn as nn

class DisLatentCode(nn.Module):
    def __init__(self, code_dim, n_layers=5, h_dim=None):
        super(DisLatentCode, self).__init__()
        
        prev_nf = code_dim
        if h_dim is None:
            h_dim = code_dim

        layers = []
        for i in range(n_layers):
            layers += [nn.Linear(prev_nf, h_dim),
                       nn.LeakyReLU(0.2, True)]
            prev_nf = h_dim

        layers += [nn.Linear(h_dim, 1)]

        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        # first flatten the tensor to have size batch x features
        x = x.view(x.shape[0], -1)
        return self.layers(x)


