from torch import nn
from torch.autograd import grad
import torch
DIM=64
OUTPUT_DIM=64*64*1

class MyConvo2d(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, he_init = True,  stride = 1, bias = True):
        super(MyConvo2d, self).__init__()
        self.he_init = he_init
        self.padding = int((kernel_size - 1)/2)
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride=1, padding=self.padding, bias = bias)

    def forward(self, input):
        output = self.conv(input)
        return output

class ConvMeanPool(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, he_init = True):
        super(ConvMeanPool, self).__init__()
        self.he_init = he_init
        self.conv = MyConvo2d(input_dim, output_dim, kernel_size, he_init = self.he_init)

    def forward(self, input):
        output = self.conv(input)
        output = (output[:,:,::2,::2] + output[:,:,1::2,::2] + output[:,:,::2,1::2] + output[:,:,1::2,1::2]) / 4
        return output

class MeanPoolConv(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, he_init = True):
        super(MeanPoolConv, self).__init__()
        self.he_init = he_init
        self.conv = MyConvo2d(input_dim, output_dim, kernel_size, he_init = self.he_init)

    def forward(self, input):
        output = input
        output = (output[:,:,::2,::2] + output[:,:,1::2,::2] + output[:,:,::2,1::2] + output[:,:,1::2,1::2]) / 4
        output = self.conv(output)
        return output

class DepthToSpace(nn.Module):
    def __init__(self, block_size):
        super(DepthToSpace, self).__init__()
        self.block_size = block_size
        self.block_size_sq = block_size*block_size

    def forward(self, input):
        output = input.permute(0, 2, 3, 1)
        (batch_size, input_height, input_width, input_depth) = output.size()
        output_depth = int(input_depth / self.block_size_sq)
        output_width = int(input_width * self.block_size)
        output_height = int(input_height * self.block_size)
        t_1 = output.reshape(batch_size, input_height, input_width, self.block_size_sq, output_depth)
        spl = t_1.split(self.block_size, 3)
        stacks = [t_t.reshape(batch_size,input_height,output_width,output_depth) for t_t in spl]
        output = torch.stack(stacks,0).transpose(0,1).permute(0,2,1,3,4).reshape(batch_size,output_height,output_width,output_depth)
        output = output.permute(0, 3, 1, 2)
        return output


class UpSampleConv(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, he_init = True, bias=True):
        super(UpSampleConv, self).__init__()
        self.he_init = he_init
        self.conv = MyConvo2d(input_dim, output_dim, kernel_size, he_init = self.he_init, bias=bias)
        self.depth_to_space = DepthToSpace(2)

    def forward(self, input):
        output = input
        output = torch.cat((output, output, output, output), 1)
        output = self.depth_to_space(output)
        output = self.conv(output)
        return output


class ResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, resample=None, hw=DIM):
        super(ResidualBlock, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.resample = resample
        self.bn1 = None
        self.bn2 = None
        self.relu1 = None
        self.relu2 = None
        if resample == 'down':
            self.relu1 = nn.LeakyReLU(negative_slope=0.2)
            self.relu2 = nn.LeakyReLU(negative_slope=0.2)
            self.bn1 = nn.LayerNorm([input_dim, hw, hw])
            self.bn2 = nn.LayerNorm([input_dim, hw, hw])
        elif resample == 'up':
            self.relu1 = nn.ReLU()
            self.relu2 = nn.ReLU()
            self.bn1 = nn.BatchNorm2d(input_dim)
            self.bn2 = nn.BatchNorm2d(output_dim)
        elif resample == None:
            #TODO: ????
            self.bn1 = nn.BatchNorm2d(output_dim)
            self.bn2 = nn.LayerNorm([input_dim, hw, hw])
        else:
            raise Exception('invalid resample value')

        if resample == 'down':
            self.conv_shortcut = MeanPoolConv(input_dim, output_dim, kernel_size = 1, he_init = False)
            self.conv_1 = MyConvo2d(input_dim, input_dim, kernel_size = kernel_size, bias = False)
            self.conv_2 = ConvMeanPool(input_dim, output_dim, kernel_size = kernel_size)
        elif resample == 'up':
            self.conv_shortcut = UpSampleConv(input_dim, output_dim, kernel_size = 1, he_init = False)
            self.conv_1 = UpSampleConv(input_dim, output_dim, kernel_size = kernel_size, bias = False)
            self.conv_2 = MyConvo2d(output_dim, output_dim, kernel_size = kernel_size)
        elif resample == None:
            self.conv_shortcut = MyConvo2d(input_dim, output_dim, kernel_size = 1, he_init = False)
            self.conv_1 = MyConvo2d(input_dim, input_dim, kernel_size = kernel_size, bias = False)
            self.conv_2 = MyConvo2d(input_dim, output_dim, kernel_size = kernel_size)
        else:
            raise Exception('invalid resample value')

    def forward(self, input):
        if self.input_dim == self.output_dim and self.resample == None:
            shortcut = input
        else:
            shortcut = self.conv_shortcut(input)

        output = input
        output = self.bn1(output)
        output = self.relu1(output)
        output = self.conv_1(output)
        output = self.bn2(output)
        output = self.relu2(output)
        output = self.conv_2(output)

        return shortcut + output

class ReLULayer(nn.Module):
    def __init__(self, n_in, n_out):
        super(ReLULayer, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.linear = nn.Linear(n_in, n_out)
        self.relu = nn.ReLU()

    def forward(self, input):
        output = self.linear(input)
        output = self.relu(output)
        return output

class FCGenerator(nn.Module):
    def __init__(self, FC_DIM=512):
        super(FCGenerator, self).__init__()
        self.relulayer1 = ReLULayer(128, FC_DIM)
        self.relulayer2 = ReLULayer(FC_DIM, FC_DIM)
        self.relulayer3 = ReLULayer(FC_DIM, FC_DIM)
        self.relulayer4 = ReLULayer(FC_DIM, FC_DIM)
        self.linear = nn.Linear(FC_DIM, OUTPUT_DIM)
        self.tanh = nn.Tanh()

    def forward(self, input):
        output = self.relulayer1(input)
        output = self.relulayer2(output)
        output = self.relulayer3(output)
        output = self.relulayer4(output)
        output = self.linear(output)
        output = self.tanh(output)
        return output

class GoodGenerator(nn.Module):
    def __init__(self, dim=DIM, output_dim=OUTPUT_DIM):
        super(GoodGenerator, self).__init__()

        self.dim = dim

        self.ln1 = nn.Linear(128, 4*4*8*self.dim) # (8*64, 4, 4)
        self.rb1 = ResidualBlock(8*self.dim, 8*self.dim, 3, resample = 'up') # (8*64, 8, 8)
        self.rb2 = ResidualBlock(8*self.dim, 4*self.dim, 3, resample = 'up') # (4*64, 16, 16)
        self.rb3 = ResidualBlock(4*self.dim, 2*self.dim, 3, resample = 'up') # (2*64, 32, 32)
        self.rb4 = ResidualBlock(2*self.dim, 1*self.dim, 3, resample = 'up') # (64, 64, 64)
        self.rb5 = ResidualBlock(1*self.dim, int(.5*self.dim), 3, resample = 'up') # (32, 128, 128)
        self.bn  = nn.BatchNorm2d(int(.5*self.dim))

        self.conv1 = MyConvo2d(int(.5*self.dim), 1, 3) # (1, 64, 64)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, input):
        output = self.ln1(input.contiguous())
        output = output.view(-1, 8*self.dim, 4, 4)
        output = self.rb1(output)
        output = self.rb2(output)
        output = self.rb3(output)
        output = self.rb4(output)
        output = self.rb5(output)

        output = self.bn(output)
        output = self.relu(output)
        output = self.conv1(output)
        output = self.tanh(output)
        output = output.view(-1, OUTPUT_DIM)
        # print(output.shape)
        return output

class GoodDiscriminator(nn.Module):
    def __init__(self, dim=DIM, num_class=2):
        super(GoodDiscriminator, self).__init__()

        self.dim = dim
        self.num_class = num_class
        self.conv1 = MyConvo2d(1, int(.5*self.dim), 3, he_init = False)
        self.rb0 = ResidualBlock(int(.5*self.dim), self.dim, 3, resample = 'down', hw=DIM*2)
        self.rb1 = ResidualBlock(self.dim, 2*self.dim, 3, resample = 'down', hw=DIM)
        self.rb2 = ResidualBlock(2*self.dim, 4*self.dim, 3, resample = 'down', hw=int(DIM/2))
        self.rb3 = ResidualBlock(4*self.dim, 8*self.dim, 3, resample = 'down', hw=int(DIM/4))
        self.rb4 = ResidualBlock(8*self.dim, 8*self.dim, 3, resample = 'down', hw=int(DIM/8))
        self.ln1 = nn.Linear(4*4*8*self.dim, 1)

        self.ln2 = nn.Linear(4*4*8*self.dim, self.num_class)

    def forward(self, input):
        output = input.contiguous()
        output = output.view(-1, 1, DIM*2, DIM*2)  # (1, 128, 128)
        output = self.conv1(output) # (32, 128, 128)
        output = self.rb0(output)   # (1*64, 64, 64)
        output = self.rb1(output)   # (2*64, 32, 32)
        output = self.rb2(output)   # (4*64, 16, 16)
        output = self.rb3(output)   # (8*64, 8, 8)
        output = self.rb4(output)   # (8*64, 4, 4)
        print(output.shape)
        output = output.view(-1, 4*4*8*self.dim)
        output_wgan = self.ln1(output)
        output_wgan = output_wgan.view(-1)
        output_congan = self.ln2(output)
        return output_wgan, output_congan

class Generator(torch.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # self.output_dim = output # should be (1, 128, 128)
        self.main_module = nn.Sequential(
            # Z latent vector 128
            nn.ConvTranspose2d(in_channels=128, out_channels=1024, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=1024),
            nn.ReLU(True),
            # State (1024x4x4)
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(True),
            # State (512x8x8)
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(True),
            # State (256x16x16)
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(True),
            # State (128x32x32)
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(True),
            # State (64x64x64)
            nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=4, stride=2, padding=1, bias=False),
            # output of main module --> Image (Cx32x32)            
            )
        self.output = nn.Tanh()

    def forward(self, x):
        x = x.view(-1, 128, 1, 1)
        x = self.main_module(x)
        # print(x.shape)
        x = self.output(x)
        # x = x.view(-1, self.output_dim)
        return x


class Discriminator(torch.nn.Module):
    def __init__(self, channels=1, num_class=2):
        super(Discriminator, self).__init__()
        # Input_dim = channels (1x128x128)
        # Output_dim = 1
        self.num_class = num_class
        self.main_module = nn.Sequential(
            # Omitting batch normalization in critic because our new penalized training objective (WGAN with gradient penalty) is no longer valid
            # in this setting, since we penalize the norm of the critic's gradient with respect to each input independently and not the enitre batch.
            # There is not good & fast implementation of layer normalization --> using per instance normalization nn.InstanceNorm2d()
            # Image (Cx128x128)
            nn.Conv2d(in_channels=channels, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(64, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state (64x64x64)
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(128, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state (128x32x32)
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # State (256x16x16)
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(512, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # State (512x8x8)
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(1024, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            )
            # output of main module --> State (1024x4x4)

        self.output = nn.Sequential(
            # The output of D is no longer a probability, we do not apply sigmoid at the output of D.
            nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False)
            )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.output_c = nn.Sequential(
            nn.Linear(1024*1*1, self.num_class)
            )


    def forward(self, x):
        x = self.main_module(x)
        output_wgan = self.output(x)
        # print(x.shape)
        output_wgan = output_wgan.view(-1)
        x = self.avgpool(x)
        # x = x.view(-1, 4*4*1024)
        x = torch.flatten(x, 1)
        output_congan = self.output_c(x)
        return output_wgan, output_congan

    def feature_extraction(self, x):
        # Use discriminator for feature extraction then flatten to vector of 16384
        x = self.main_module(x)
        return x.view(-1, 1024*4*4)