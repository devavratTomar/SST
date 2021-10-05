import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, ndf):
        super(Discriminator, self).__init__()
        modules = []

        modules += [nn.Conv3d(1, ndf, kernel_size=4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, True)]

        for i in range(1, 4):
            fl = (2**i) * ndf
            modules += [nn.Conv3d(fl//2, fl, kernel_size=4, stride=2, padding=1),
                        nn.InstanceNorm3d(fl),
                        nn.LeakyReLU(0.2, True)]

        modules += [nn.Conv3d(fl, 1, kernel_size=4, stride=1, padding=1)]

        self.main = nn.Sequential(*modules)

    def forward(self, img):
        return self.main(img)
