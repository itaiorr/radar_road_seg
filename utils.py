# All rights reserved. 

# Copyright (c) 2020

# Source and binary forms are subject non-exclusive, revocable, non-transferable, and limited right to use the code for the exclusive purpose of undertaking academic or not-for-profit research.

# Redistributions must retain the above copyright notice, this license and the following disclaimer.

# Use of the code or any part thereof for commercial purposes is strictly prohibited.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from imports import *
from data_loader import *
from loss import *
from model import net

def run_configuration(config):
    
    # Load model
    model = net(config.gpus_list,config.batch_size,config.output_shape)     
    printNetwork(model)
    
    if config.pretrained:
        model.load_state_dict(torch.load(config.pretrained_path, map_location=lambda storage, loc: storage))
        print('Pre-trained model loded')
        print('---'*30)

    model = model.cuda(config.gpus_list[0])
    
    trainDatasets = []
    trainDisc     = []
    trainWf       = []

    valDatasets   = [] 
    valDisc       = []
    valWf         = []
            
    os.makedirs(config.save_path,exist_ok=True)

    if config.pretrained:
        dfLoss     = pd.read_csv('')
        prev_epoch = len(dfLoss)
    else:
        dfLoss     = pd.DataFrame()
        prev_epoch = 0

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.lrMax, betas=(0.9, 0.999), eps=1e-8)

    # Define loss
    loss = customLoss(config.gpus_list,config.dice_coeff,config.bce_coeff,config.rce_coeff)

    # Create datasets
    dfTrain = create_df(trainDatasets,trainDisc,trainWf)
    dfVal   = create_df(valDatasets,valDisc,valWf)
    print('Training size:',len(dfTrain))
    print('Validation size:',len(dfVal))

    # Create data loaders
    train_set       = DatasetFromFolder(dfTrain,config.output_shape)
    trainDataLoader = DataLoader(dataset=train_set, num_workers=config.n_threads, batch_size=config.batch_size,shuffle=True,worker_init_fn=lambda x: np.random.seed(),pin_memory=True,drop_last=True)
    
    val_set         = DatasetFromFolder(dfVal,config.output_shape)
    valDataLoader   = DataLoader(dataset=val_set, num_workers=config.n_threads, batch_size=config.batch_size, shuffle=True,worker_init_fn=lambda x: np.random.seed(),drop_last=True)

    run_train_val(config, trainDataLoader, valDataLoader, optimizer, loss, epoch, model, dfLoss, prev_epoch)

        
def run_train_val(config, trainDataLoader, valDataLoader, optimizer, loss, epoch, model, dfLoss, prev_epoch):
    
    for epoch in range(1, config.n_epochs + 1):

        # Train for 1 epoch
        pbar = tqdm(iter(trainDataLoader),leave=False, total=len(trainDataLoader))
        pbar.set_description('Epoch ' + str(epoch) + ' Train')
        epochTrainLogger = run_epoch(config,optimizer,loss,epoch,pbar,model,configuration='train')

        # Evalute for 1 epoch
        pbar2 = tqdm(iter(valDataLoader),leave=False, total=len(valDataLoader))
        pbar2.set_description('Epoch ' + str(epoch) + ' Val')
        epochValLogger = run_epoch(config,optimizer,loss,epoch,pbar2,model,configuration='val')

        # Learning rate scheduler: cosine annealing with warm restart (decay if lrCycles=1)
        for param_group in optimizer.param_groups:
            param_group['lr'] = config.lr_min + 0.5*(config.lrmax-config.lr_min)*(1+ np.cos(epoch*np.pi/config.n_epochs*config.lr_cycles))
        print('Learning rate decay rd: lr={}'.format(optimizer.param_groups[0]['lr']))

        # Save weights
        model_out_path = config.save_path + config.model_type + "_epoch_{}.pth".format(epoch)
        torch.save(model.state_dict(), model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))

        # Save loss
        print_logger_stats(config, dfLoss, epoch, prev_epoch, epochTrainLogger, epochValLogger)

def run_epoch(config,optimizer,loss,epoch,pbar,model,configuration):
    
    if configuration=='train':
        model.train()
    elif configuration=='val':
        model.eval()  
        
    epoch_logger = {'criterion':torch.zeros(1).cuda(gpus_list[0]),'dice':torch.zeros(1).cuda(gpus_list[0]),'bce':torch.zeros(1).cuda(gpus_list[0]),'rce':torch.zeros(1).cuda(gpus_list[0])}
    
    for iteration,batch in enumerate(pbar, 1):
        
        if configuration=='train':
            optimizer.zero_grad()
            logits = model(batch['input_rd'].cuda(gpus_list[0]))
            
        elif configuration=='val':
            with torch.no_grad():
                logits = model(batch['input_rd'].cuda(gpus_list[0]))
                
        logger = loss(logits,batch['label'].cuda(gpus_list[0]))
            
        if configuration=='train':
            logger['criterion'].backward()  
            optimizer.step()
       
        epoch_logger['criterion'] += logger['criterion'].detach()
        epoch_logger['dice']      += logger['dice'].detach()
        epoch_logger['bce']       += logger['bce'].detach()
        epoch_logger['rce']       += logger['rce'].detach()

        stats = {'criterion':epoch_logger['criterion'].detach().item()/(iteration*batch_size),'dice':epoch_logger['dice'].detach().item()/(iteration*batch_size),
                 'bce':epoch_logger['bce'].detach().item()/(iteration*batch_size),'rce':epoch_logger['rce'].detach().item()/(iteration*batch_size)}
        pbar.set_postfix(ordered_dict=stats, refresh=True)
        
        # Debug
        if debug_mode and iteration==4: break
            
    epoch_logger['criterion'] /= (iteration*batch_size)
    epoch_logger['dice']      /= (iteration*batch_size)
    epoch_logger['bce']       /= (iteration*batch_size)
    epoch_logger['rce']       /= (iteration*batch_size)

    return epoch_logger

def create_df(datasets,disc,wf):
    df = pd.DataFrame()

    for i,experiment in enumerate(datasets):
        dfExperiment            = pd.read_csv('/workspace/' + disc[i] + '/' + wf[i] +'/' + experiment + '/synced_data.csv')
        dfExperiment['Dataset'] = experiment
        dfExperiment['Disc']    = disc[i]
        dfExperiment['WF']      = wf[i]
        df = df.append(dfExperiment)
    df = df.reset_index(drop=True)

    return df

def print_logger_stats(config, dfLoss, epoch, prev_epoch, epochTrainLogger, epochValLogger):
    
    # Save loss
    dfLoss.loc[epoch+prev_epoch,'epoch']      = epoch+prev_epoch
    dfLoss.loc[epoch+prev_epoch,'train_loss'] = epochTrainLogger['criterion'].item()
    dfLoss.loc[epoch+prev_epoch,'train_dice'] = epochTrainLogger['dice'].item()
    dfLoss.loc[epoch+prev_epoch,'train_bce']  = epochTrainLogger['bce'].item()
    dfLoss.loc[epoch+prev_epoch,'train_rce']  = epochTrainLogger['rce'].item()
    dfLoss.loc[epoch+prev_epoch,'val_loss']   = epochValLogger['criterion'].item()
    dfLoss.loc[epoch+prev_epoch,'val_dice']   = epochValLogger['dice'].item()
    dfLoss.loc[epoch+prev_epoch,'val_bce']    = epochValLogger['bce'].item()
    dfLoss.loc[epoch+prev_epoch,'val_rce']    = epochValLogger['rce'].item()
    
    dfLoss.to_csv(config.savePath + '/loss_logger.csv', sep=',',index=False), sep=',',index=False)
    
    # Print loss figure
    plt.figure('results',figsize=(20,10),dpi=300)
        
    plt.subplot(2,3,1)
    plt.title('Total Loss')
    plt.plot(dfLoss['epoch'],dfLoss['train_loss'],'b',label='Train')
    plt.plot(dfLoss['epoch'],dfLoss['val_loss'],'r',label='Val')
    plt.legend()
    if prev_epoch>0:
        plt.vlines(prev_epoch,
                   np.min([dfLoss['train_loss'],dfLoss['val_loss']]),
                   np.max([dfLoss['train_loss'],dfLoss['val_loss']]),'r',linestyles='dashed')

    plt.subplot(2,3,(2,3))
    plt.plot(dfLoss.loc[prev_epoch:,'train_loss'],label='train')
    plt.plot(dfLoss.loc[prev_epoch:,'val_loss'],label='val')
    plt.legend()
    plt.title('Total Loss | Current Train')
    
    plt.subplot(2,3,4)
    plt.title('Dice')
    plt.plot(dfLoss['epoch'],dfLoss['train_dice'],'b',label='Train')
    plt.plot(dfLoss['epoch'],dfLoss['val_dice'],'r',label='Val')
    plt.legend()
    if prev_epoch>0:
        plt.vlines(prev_epoch,
                   np.min([dfLoss['train_dice'],dfLoss['val_dice']]),
                   np.max([dfLoss['train_dice'],dfLoss['val_dice']]),'r',linestyles='dashed')
    
    plt.subplot(2,3,5)
    plt.title('BCE')
    plt.plot(dfLoss['epoch'],dfLoss['train_bce'],'b',label='Train')
    plt.plot(dfLoss['epoch'],dfLoss['val_bce'],'r',label='Val')
    plt.legend()
    if prev_epoch>0:
        plt.vlines(prev_epoch,
                   np.min([dfLoss['train_bce'],dfLoss['val_bce']]),
                   np.max([dfLoss['train_bce'],dfLoss['val_bce']]),'r',linestyles='dashed')
 
    plt.subplot(2,3,6)
    plt.title('RCE')
    plt.plot(dfLoss['epoch'],dfLoss['train_rce'],'b',label='Train')
    plt.plot(dfLoss['epoch'],dfLoss['val_rce'],'r',label='Val')
    plt.legend()
    if prev_epoch>0:
        plt.vlines(prev_epoch,
                   np.min([dfLoss['train_rce'],dfLoss['val_rce']]),
                   np.max([dfLoss['train_rce'],dfLoss['val_rce']]),'r',linestyles='dashed')
        
    # Save figure
    plt.savefig(config.savePath + 'loss_logger')
    plt.close()