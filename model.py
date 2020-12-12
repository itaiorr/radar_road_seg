# All rights reserved. 

# Copyright (c) 2020

# Source and binary forms are subject non-exclusive, revocable, non-transferable, and limited right to use the code for the exclusive purpose of undertaking academic or not-for-profit research.

# Redistributions must retain the above copyright notice, this license and the following disclaimer.

# Use of the code or any part thereof for commercial purposes is strictly prohibited.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from imports import *

class add_coord(nn.Module):
    
    def __init__(self,image_shape,gpus_list,batch_size):
        super().__init__()
        image_height = image_shape[0]
        image_width  = image_shape[1]
               
        y_coords = (2.0 * torch.arange(image_height).unsqueeze(1).expand(image_height, image_width) / (image_height - 1.0) - 1.0)
        x_coords = torch.arange(image_width).unsqueeze(0).expand(image_height, image_width).float() / image_width
        self.coords = torch.unsqueeze(torch.stack((y_coords, x_coords), dim=0), dim=0).repeat(batch_size, 1, 1, 1)
        self.coords = self.coords.cuda(gpus_list[0])
        
    def forward(self,x):
        return torch.cat((x,self.coords), dim=1)

class CAM_Module(nn.Module):
    def __init__(self, in_dim,batch_size):
        super().__init__()
        self.chanel_in = in_dim
        self.gamma     = nn.Parameter(torch.ones(batch_size,in_dim,1,1))
        self.softmax   = nn.Softmax(dim=-1)
        
    def forward(self,x):
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key   = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy     = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention  = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)
        return self.gamma*out + x

class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch,batch_size):
        super().__init__()
        
        self.conv1  = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1  = nn.InstanceNorm2d(out_ch,affine=True)
        self.atten1 = CAM_Module(in_ch,batch_size)
        self.act1   = nn.LeakyReLU()
        
        self.conv2  = nn.Conv2d(in_ch+out_ch, out_ch, 3, padding=1)
        self.norm2  = nn.InstanceNorm2d(out_ch,affine=True)
        self.atten2 = CAM_Module(in_ch+out_ch,batch_size)
        self.act2   = nn.LeakyReLU()

    def forward(self, x):
        res = x
        
        x = self.atten1(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
                
        x = self.atten2(torch.cat([res,x],dim=1))
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act2(x)
        
        return x

class inconv(nn.Module):
    def __init__(self, in_ch, out_ch,batch_size):
        super().__init__()
        self.conv = double_conv(in_ch, out_ch,batch_size)

    def forward(self, x):
        x = self.conv(x)
        return x

class down(nn.Module):
    def __init__(self, in_ch, out_ch,batch_size):
        super().__init__()
        self.mpconv = nn.Sequential(
            nn.AvgPool2d(2),
            double_conv(in_ch, out_ch,batch_size))

    def forward(self, x):
        x = self.mpconv(x)
        return x

class up(nn.Module):
    def __init__(self, in_ch, out_ch,batch_size, bilinear=False):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch,batch_size)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        if diffX<0:
            diffX = abs(diffX)
            if diffY<0:
                diffY = abs(diffY)
            x1 = F.pad(x1, (diffY // 2, diffY - diffY//2, diffX // 2, diffX + diffX//2))
        elif diffX>0:
            x2 = F.pad(x2, (diffY // 2, diffY - diffY//2, diffX // 2, diffX - diffX//2))
        elif diffX==0:
            x2 = F.pad(x2, (diffY // 2, diffY - diffY//2, diffX // 2, diffX - diffX//2))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

class net(nn.Module):
    def __init__(self,gpus_list,batch_size,output_shape):
        super().__init__()
        scaleFactor = 2
        self.inc    = inconv(48*2+2,               int(64/scaleFactor),batch_size)
        self.down1  = down(int(64/scaleFactor),    int(128/scaleFactor),batch_size)
        self.down2  = down(int(128/scaleFactor),   int(256/scaleFactor),batch_size)
        self.down3  = down(int(256/scaleFactor),   int(512/scaleFactor),batch_size)
        self.down4  = down(int(512/scaleFactor),   int(512/scaleFactor),batch_size)
        self.up1    = up(int(1024/scaleFactor),    int(256/scaleFactor),batch_size)
        self.up2    = up(int(512/scaleFactor),     int(128/scaleFactor),batch_size)
        self.up3    = up(int(256/scaleFactor),     int(64/scaleFactor),batch_size)
        self.up4    = up(int(128/scaleFactor),     int(64/scaleFactor),batch_size)
        self.outc   = outconv(int(64/scaleFactor), 1)
        
        # Position encoding
        self.coord = add_coord(output_shape,gpus_list,batch_size)

    def forward(self, x):
                
        x1 = self.inc(self.coord(x))
 
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x  = self.up1(x5, x4)
        x  = self.up2(x, x3)
        x  = self.up3(x, x2)
        x  = self.up4(x, x1)
        x  = self.outc(x)
        
        return x
