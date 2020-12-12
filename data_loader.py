# All rights reserved. 

# Copyright (c) 2020

# Source and binary forms are subject non-exclusive, revocable, non-transferable, and limited right to use the code for the exclusive purpose of undertaking academic or not-for-profit research.

# Redistributions must retain the above copyright notice, this license and the following disclaimer.

# Use of the code or any part thereof for commercial purposes is strictly prohibited.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from imports import *

class DatasetFromFolder(data.Dataset):  
    def __init__(self, df, output_shape, transform=None):
        
        self.filenames    = df 
        self.output_shape = output_shape
       
    def __len__(self):
        return len(self.filenames)
   
    def __getitem__(self, idx):
                    
        # Grab sample           
        sample_dataset   = str(self.filenames.loc[idx,'Dataset'])
        sample_disc      = self.filenames.loc[idx,'Disc']
        sample_wf        = self.filenames.loc[idx,'WF']
        timestamp_camera = str(self.filenames.loc[idx,'timestampCamera'])
        timestamp_radar  = str(self.filenames.loc[idx,'timestampRadar'])

        # Grab radar frame
        path_radar = '/workspace/'+sample_disc+'/'+sample_wf+'/'+sample_dataset+'/range_doppler/range_doppler_'+timestamp_radar+'.npy'
        radar_rd   = np.load(path_radar)
        radar_rd   = np.transpose(radar_rd,(1,2,0))

        # Torch and Split into real and imaginary
        S_re = torch.from_numpy(np.real(radar_rd))
        S_im = torch.from_numpy(np.imag(radar_rd))

        # Build input range_doppler 
        input_rd = torch.zeros(2*radar_rd.shape[0],radar_rd.shape[1],radar_rd.shape[2])  
        for i in range(len(self.input_ch)):
            input_rd[2*i,:,:]   = F.interpolate(S_re.unsqueeze(0),(self.output_shape[0],self.output_shape[1])).squeeze(0)
            input_rd[2*i+1,:,:] = F.interpolate(S_im.unsqueeze(0),(self.output_shape[0],self.output_shape[1])).squeeze(0)
            
        # Grab label seg
        label = np.load('/workspace/'+sample_disc+'/'+sample_wf+'/'+ sample_dataset+'/camera_road_seg/camera_road_seg_' + timestamp_camera + '.npz')['arr_0']

        # Resize label to output shape
        label = cv2.resize(label,(self.output_shape[1],self.output_shape[0]))

        # Torch label
        label = torch.unsqueeze(torch.from_numpy(label),0)

        # Build sample
        sample = {'input_rd':input_rd,
                  'path_radar':path_radar,
                  'label':label,
                  'dataset':sample_dataset,
                  'wf':sample_wf,
                  'disc':sample_disc,
                  'timestamp_camera':timestamp_camera,
                  'timestamp_radar':timestamp_radar}

        return sample