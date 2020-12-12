# All rights reserved. 

# Copyright (c) 2020

# Source and binary forms are subject non-exclusive, revocable, non-transferable, and limited right to use the code for the exclusive purpose of undertaking academic or not-for-profit research.

# Redistributions must retain the above copyright notice, this license and the following disclaimer.

# Use of the code or any part thereof for commercial purposes is strictly prohibited.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from imports import *

def DiceLoss(pred, label):
    smooth = 1.0

    pred  = pred.view(pred.shape[0],-1)
    label = label.view(label.shape[0],-1)

    return 1-((2.0 * (pred * label).sum() + smooth) / (iflat.sum()**2 + tflat.sum()**2 + smooth))

def rceLoss(pred, label, gpus_list):
    label = torch.clamp(label, min=1e-4, max=1.0)
    pred  = torch.clamp(pred, min=1e-4, max=1.0)

    # RCE loss
    rce = (-1*torch.sum(pred * torch.log(label), dim=1)).mean()
    
    return rce

class customLoss(nn.Module):  
    def __init__(self,gpus_list,dice_coeff,bce_coeff,rce_coeff):
        super().__init__()
        
        self.gpus_list = gpus_list
        
        self.BCE = nn.BCELoss().cuda(gpus_list[0])
        self.dice_coeff = dice_coeff
        self.bce_coeff  = bce_coeff
        self.rce_coeff  = rce_coeff

    def forward(self,logits,label_seg):
        
        pred_probs = torch.sigmoid(logits)
        dice_loss  = self.dice_coeff * DiceLoss(pred_probs,label_seg)
        bce_loss   = self.bce_coeff * self.BCE(pred_probs,label_seg)
        rce_loss   = self.rce_coeff * rceLoss(pred_probs,label_seg,self.gpus_list)
        
        criterion  = dice_loss + bce_loss + rce_loss

        # Build logger
        logger = {'criterion':criterion,
                  'dice'     : dice_loss,
                  'bce'      : bce_loss,
                  'rce'      : rce_loss}
                
        return logger