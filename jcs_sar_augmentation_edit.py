import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torchvision
from PIL import Image
import torchvision.transforms as TRF
import torchvision.transforms.functional as TRF_F
import torchvision.utils as vutils
import math
import matplotlib.pyplot as plt
import re
import random
import cv2
import matplotlib.animation as animation

# Making File Info Data Frame 
## Example: dirpath/deg_loc/xxx/xxx/obj_loc/file_name.jpg -> (0/1/2/3/4/5) deg_loc: 1, obj_loc: 4
## img_ext: ['JPG','PNG']
def make_file_info(dirpath, deg_loc, obj_loc, img_ext=['JPG','PNG'], sep='/'):
    file_list = list()
    for (dirpath, dirnames, filenames) in os.walk(dirpath):
        file_list += [os.path.join(dirpath, file) for file in filenames]
    file_image = [f for f in file_list if f[-3:] in img_ext]
    file_label = [str(np.apply_along_axis(sep.join, 0, np.array(re.split(sep,fname))[[deg_loc,obj_loc]])) for fname in file_image]
    deg_label = [re.split(sep,fname)[deg_loc] for fname in file_image]
    obj_label = [re.split(sep,fname)[obj_loc] for fname in file_image]
    file_info = pd.DataFrame({'degree':deg_label, 'object':obj_label,'file_label':file_label,'file_loc':file_image})
    
    return file_info


# Plot Figure
def plot_images(file_info, nrow=5, ncol=5, fig_size = (20,20),cmap='gray', label_col=2, file_col=3):
    plt.figure(figsize=fig_size)
    for i in range(0,nrow*ncol):
        img = Image.open(file_info.iloc[i,file_col])
        img = np.array(img)
        plt.subplot(nrow, ncol, i+1)
        plt.title(file_info.iloc[i,label_col])
        plt.imshow(img,cmap=cmap)
        plt.axis('off')

def data_loader_target(file_info, size_out = (64,64), scale='max',dtype=torch.float32, device='cpu',
                       deg_col=0, label_col=1, file_col=3, normalize=[0.5,0.5],
                       class_label=['SLICY', 'ZSU_23_4', 'T62', '2S1', 'BRDM_2']):
    n = len(file_info)
    n_cl = len(class_label)
    
    resize_ftn = TRF.Resize(size_out)
    
    # image tensor
    img_list = []
    for i in range(n):
        img_pil = Image.open(file_info.iloc[i,file_col])
        img_pil_rsz = resize_ftn(img_pil)
        img = np.array(img_pil_rsz)
        if scale=='std':
            img_max = img.max()
            img = img/img_max
            img -= normalize[0]
            img /= normalize[1]
            
        if scale=='max':
            img_max = img.max()
            img = (img/img_max)
            
        if scale=='minmax':
            img_min = img.min()
            img_max = img.max()
            img = ((img-img_min)/(img_max-img_min))
            img = img
        if scale=='8bit':
            img = (img/255.0)
        if scale=='none':
            img = img
        
        if img.ndim==2:
            h, w = img.shape
            img = img.reshape((1,h,w))
        elif img.ndim==3:
            img = np.transpose(img, (1,2,0))
            
        img_list.append(img)
    
    img_tensor = torch.tensor(np.array(img_list), dtype=dtype, device=device)
    
    # label tensor: one-hot encoding
    lbl_np = np.zeros((n,n_cl))
    obj_np = file_info.iloc[:,label_col].values
    for i in range(len(class_label)):
        indx = np.where(obj_np==class_label[i])[0]
        lbl_np[indx,i] = 1
    
    lbl_tensor = torch.tensor(lbl_np.argmax(1), dtype=torch.int64, device=device)
    
    # deg tensor
    deg = [f for f in file_info.iloc[:,deg_col]]
    #deg_np = np.array(deg)
    #deg_tensor = torch.tensor(deg_np, device=device)
    
    return img_tensor, lbl_tensor, deg


def train_test_split(data, label, train_portion = 0.75, seed=1234):
    np.random.seed(seed)
    n = len(data)
    shuffle_index = np.random.permutation(n)
    tr_n = int(np.ceil(train_portion*n))
    ts_n = n - tr_n
    tr_indx = shuffle_index[:tr_n]
    ts_indx = shuffle_index[tr_n:]
    train_data, train_label = data[tr_indx], label[tr_indx]
    test_data, test_label = data[ts_indx], label[ts_indx]
    
    return train_data, train_label, test_data, test_label



class cnn_clf_net(nn.Module):
    
    def __init__(self, img_size=64, ncol=1, num_class = 5):
        super(cnn_clf_net, self).__init__()
        
        self.img_sz = img_size
        self.ncol = ncol
        self.num_class = num_class
        self.conv1 = nn.Conv2d(self.ncol,32,(3,3),padding=1)
        self.conv2 = nn.Conv2d(32,32,(3,3),padding=1)
        self.conv3 = nn.Conv2d(32,64,(3,3),padding=1)
        self.conv4 = nn.Conv2d(64,64,(3,3),padding=1)
        self.conv5 = nn.Conv2d(64,128,(3,3),padding=1)
        self.conv6 = nn.Conv2d(128,128,(3,3),padding=1)
        self.conv7 = nn.Conv2d(128,256,(6,6),padding=0)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
         
        self.dropout = nn.Dropout(p=0.5)
        
        self.fsz = math.floor(self.img_sz/8.0 -6 + 1)
        self.fc1 = nn.Linear((self.fsz**2)*256, 512)
        self.fc2 = nn.Linear(512, self.num_class)
        
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.bn1(self.conv2(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        x = F.relu(self.conv3(x))
        x = F.relu(self.bn2(self.conv4(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        x = F.relu(self.conv5(x))
        x = F.relu(self.bn3(self.conv6(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
                
        x = F.relu(self.conv7(x))
        
        x = x.view(x.size(0),-1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        #x = self.softmax(x)
        
        return x
    

class Data_loader:
    def __init__(self, data, label, batch_size=50, shuffle=False, device='cuda'):
        self.data = data
        self.n = len(data)
        self.label = label
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device
        self.index = np.arange(self.n, dtype=np.int32)
        if self.shuffle==True:
            self.index = np.random.permutation(self.n).astype(np.int32)
    
    def shuffle_ftn(self):
        self.index = np.random.permutation(self.n).astype(np.int32)
    
    def __len__(self):
        return np.ceil(self.n/self.batch_size).astype(np.int32)

    
    def __iter__(self):
        st_indx = np.arange(0,self.n,self.batch_size)
        if self.n % self.batch_size == 0:
            ed_indx = np.arange(self.batch_size-1,self.n,self.batch_size)
        else:                        
            ed_indx = np.append(np.arange(self.batch_size-1,self.n,self.batch_size),self.n-1)

        n_bc = len(st_indx)
        index = self.index
        for st, ed in zip(st_indx, ed_indx):
            yield self.data[index[st:(ed+1)]].to(device=self.device), self.label[index[st:(ed+1)]].to(device=self.device)


def train_model(tr_data, tr_label, batch_size=48, n_epoch=5, lr_rate=0.001, momentum=0.9, shuffle=True, device='cuda'):
    
    tr_data_load = Data_loader(tr_data, tr_label, batch_size=batch_size, shuffle=False)
    ncol, img_sz = tr_data.shape[1:3]
    model = cnn_clf_net(img_sz, ncol, num_class = 5).to(device)
    opt = optim.SGD(model.parameters(), lr=lr_rate, momentum=momentum)

    n = tr_data_load.n
    ep_loss = []
    ep_loss_hist = []
    ep_acc = []
    ep_prd_list = []
    ep_y_list = []
    for k in range(n_epoch):
        print(f"Epoch {k+1}\n-------------------------------")
        pred_list = []
        y_list = []
        loss_list = []
        model.train()
        tr_data_load.shuffle_ftn()

        total_loss, correct = 0,0
        for i, (X, y) in enumerate(tr_data_load):
            
            n_sub = len(X)
            opt.zero_grad()
            pred = model(X)
            loss = F.cross_entropy(pred, y)
            loss.backward()
            opt.step()

            pred_list.append(pred.argmax(1))
            y_list.append(y)
            loss_list.append(loss.item())

            total_loss += loss.item()*n_sub
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

            if i % 50 == 0:
                loss_val, current = loss.item(), i * len(X)
                print(f"loss: {loss_val:>7f}  [{current:>5d}/{n:>5d}]")

        total_loss /= n
        correct /= n

        print(f" Training Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {total_loss:>8f} \n")

        ep_loss.append(total_loss)
        ep_acc.append(correct)
        ep_prd_list.append(torch.cat(pred_list))
        ep_y_list.append(torch.cat(y_list))
        ep_loss_hist.append(loss_list)
    print("Done!")
    
    return model, ep_loss, ep_acc, ep_prd_list, ep_y_list, ep_loss_hist


def test_model(model, ts_data, ts_label, batch_size=48):
 
    ts_data_load = Data_loader(ts_data, ts_label, batch_size=batch_size, shuffle=False)
    n = ts_data_load.n
    pred_list = []
    y_list = []
    model.eval()
    total_loss, correct = 0,0
    for i, (X, y) in enumerate(ts_data_load):
        n_sub = len(X)
        with torch.no_grad():
            pred = model(X)
            loss = F.cross_entropy(pred, y)

        pred_list.append(pred.argmax(1))
        y_list.append(y)
        total_loss += loss.item()*n_sub
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    total_loss /= n
    correct /= n

    print(f" Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {total_loss:>8f} \n")
    
    return torch.cat(pred_list), torch.cat(y_list), total_loss, correct





# 'vflip': True or False
# 'hflip': True or False
# 'rotate': angle value or vector (-180,180)
# 'resize_center_crop': crop_size
# 'resize': output_size
# 'add_noise': dictionary 
#             {'dist':'unif' or 'norm', 'param': [min,max] or [mean, sd] }
# 'affine': dictionary 
#             {'angle': angle_value, 'translate': [h, v],
#              'scale': scale_value, 'shear': value, 'fill', value}


def geom_augmentation(tensor_data, tensor_label, **kwargs):
    arg_list = np.array(['vflip','hflip','rotate','resize','resize_center_crop',
               'add_noise', 'affine'])
    arg_name =  np.array(list(kwargs.keys()))
    #print(arg_name)
    if np.all(np.isin(arg_name,arg_list))==False:
        print('Given arguments have non-valid arguments')
        return None
    
    aug_data_list = []
    aug_label_list = []
    
    if 'vflip' in arg_name:
        if kwargs['vflip']==True:
            out = TRF_F.vflip(tensor_data)
            aug_data_list.append(out)
            aug_label_list.append(tensor_label)
    
    if 'hflip' in arg_name:
        if kwargs['hflip']==True:
            out = TRF_F.hflip(tensor_data)
            aug_data_list.append(out)
            aug_label_list.append(tensor_label)
    
    if 'rotate' in arg_name:
        angle = kwargs['rotate']
        for ang in angle:
            out = TRF_F.rotate(tensor_data, float(ang))
            aug_data_list.append(out)
            aug_label_list.append(tensor_label)
    
    if 'resize' in arg_name: #[..., H, W]
        out = TRF_F.resize(tensor_data, size=kwargs['resize'])
        aug_data_list.append(out)
        aug_label_list.append(tensor_label)
    
    if 'resize_center_crop' in arg_name:
        org_hw = list(tensor_data.shape[-2:])
        out_size = kwargs['resize_center_crop']
        out_crop = TRF_F.center_crop(tensor_data, output_size=out_size)
        out = TRF_F.resize(out_crop, size=org_hw)
        aug_data_list.append(out)
        aug_label_list.append(tensor_label)
        
    if 'add_noise' in arg_name:
        org_shape = tensor_data.shape
        if kwargs['add_noise']['dist']=='unif':
            min_u, max_u = kwargs['add_noise']['param']
            noise_mat = np.random.uniform(min_u, max_u, size=org_shape)
        
        if kwargs['add_noise']['dist']=='norm':
            mean, sd = kwargs['add_noise']['param']
            noise_mat = np.random.uniform(mean, sd, size=org_shape)
        
        out = tensor_data + torch.tensor(noise_mat)
        aug_data_list.append(out)
        aug_label_list.append(tensor_label)
    
    if 'affine' in arg_name:
        out = TRF_F.affine(tensor_data, **kwargs['affine'])
        aug_data_list.append(out)
        aug_label_list.append(tensor_label)
    
    return torch.cat(aug_data_list).type(torch.float32), torch.cat(aug_label_list).type(torch.int64)



# Save samples
def export_images(path, file_name, data, label, class_label, type='ZO'):
    n, c, h, w = data.shape
    data_cpu = data.to('cpu').detach().clone().numpy()
    if type=='ZO':
        data_cpu *= 255
        data_cpu = data_cpu.astype(np.int32)
        
    if type=='MO':
        data_cpu = (data_cpu*0.5)+0.5
        data_cpu *= 255
        data_cpu = data_cpu.astype(np.int32)
        
    lbl_cpu = label.to('cpu').detach().clone().numpy()
    
    if not 'DEG' in os.listdir(path):
        os.mkdir(os.path.join(path,'DEG'))
    
    for i in range(n):
        if not class_label[lbl_cpu[i]] in os.listdir(os.path.join(path,'DEG')):
            os.mkdir(os.path.join(path,'DEG',class_label[lbl_cpu[i]]))
    
    for i in range(n):
        img = data_cpu[i].transpose((1,2,0)).astype(np.uint8).squeeze()
        f_name = os.path.join(path,'DEG',class_label[lbl_cpu[i]],file_name,)
        #plt.imsave(''.join([f_name,'_',class_label[lbl_cpu[i]],'_',str(i),'.jpg']), img)
        cv2.imwrite(''.join([f_name,'_',class_label[lbl_cpu[i]],'_',str(i),'.jpg']), img)




def init_weights(layer):
    cl_name = layer.__class__.__name__
    if cl_name.find('Conv') != -1:
        nn.init.normal_(layer.weight.data, 0.0, 0.02)
    elif cl_name.find('BatchNorm') != -1:
        nn.init.normal_(layer.weight.data, 0.0, 0.02)
        nn.init.constant_(layer.bias.data, 0)

class Generator(nn.Module):
    def __init__(self, input_dim, output_col, num_node=64):
        super(Generator, self).__init__()
        self.in_dim = input_dim
        self.out_c = output_col
        self.n_node = num_node
        
        self.convtr1 = nn.ConvTranspose2d(self.in_dim, self.n_node*16, 4, 1, 0, bias=False) # 4x4
        self.convtr2 = nn.ConvTranspose2d(self.n_node*16, self.n_node*8, 4, 2, 1, bias=False) # 8x8
        self.convtr3 = nn.ConvTranspose2d(self.n_node*8, self.n_node*4, 4, 2, 1, bias=False) # 16x16
        self.convtr4 = nn.ConvTranspose2d(self.n_node*4, self.n_node*4, 4, 2, 1, bias=False) # 32x32
        self.convtr5 = nn.ConvTranspose2d(self.n_node*4, self.out_c, 4, 2, 1, bias=False) # 64x64
        
        self.bn1 = nn.BatchNorm2d(self.n_node*16)
        self.bn2 = nn.BatchNorm2d(self.n_node*8)
        self.bn3 = nn.BatchNorm2d(self.n_node*4)
        self.bn4 = nn.BatchNorm2d(self.n_node*4)
        
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        x = F.relu(self.bn1(self.convtr1(x)))
        x = F.relu(self.bn2(self.convtr2(x)))
        x = F.relu(self.bn3(self.convtr3(x)))
        x = F.relu(self.bn4(self.convtr4(x)))
        x = self.tanh(self.convtr5(x))
        
        return x



class Discriminator(nn.Module):
    def __init__(self, input_col, num_node=64):
        super(Discriminator, self).__init__()

        self.in_c = input_col
        self.n_node = num_node
        
        self.conv1 = nn.Conv2d(1,self.n_node,(4,4),stride=2, padding=1, bias=False)
        self.conv2 = nn.Conv2d(self.n_node,self.n_node*2,(4,4),stride=2, padding=1, bias=False)
        self.conv3 = nn.Conv2d(self.n_node*2,self.n_node*4,(4,4),stride=2, padding=1, bias=False)
        self.conv4 = nn.Conv2d(self.n_node*4,self.n_node*8,(4,4),stride=2, padding=1, bias=False)
        #self.conv5 = nn.Conv2d(self.n_node*8,1,(4,4),stride=1, padding=0, bias=False)
        
        self.bn1 = nn.BatchNorm2d(self.n_node)
        self.bn2 = nn.BatchNorm2d(self.n_node*2)
        self.bn3 = nn.BatchNorm2d(self.n_node*4)
        self.bn4 = nn.BatchNorm2d(self.n_node*8)

        self.fc1 = nn.Linear(4*4*8*self.n_node, 8*self.n_node)
        self.fc2 = nn.Linear(8*self.n_node, 1)                        
    
    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)),negative_slope=0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)),negative_slope=0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)),negative_slope=0.2)
        x = F.leaky_relu(self.bn4(self.conv4(x)),negative_slope=0.2)

        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        #x = torch.clamp(x, min=1e-5, max=1.0-1e-5)
        x = x.view(-1)
        
        #x = self.conv5(x)
        #x = torch.sigmoid(x)
        #x = torch.clamp(x, min=1e-5, max=1.0-1e-5)
        #x = torch.flatten(x)#x.view(-1)
        #x = x.view(-1,8*8*8*self.n_node)
        #x = F.relu(self.fc1(x))
        #x = torch.sigmoid(self.fc2(x))
        #x = x.view(-1)
        
        return x


# Training Loop
def train_gan_model(gan_tr_data, gan_tr_lbl, num_nz=200, num_node=32, out_col = 1, num_epochs=10, 
                    batch_size=48,lr_rate = 0.0002, rec_step=5, device='cuda'):
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    netG = Generator(num_nz, out_col, num_node).to(device)
    netG.apply(init_weights)

    netD = Discriminator(out_col, num_node).to(device)
    netD.apply(init_weights)

    criterion = nn.BCELoss()

    fixed_noise = torch.randn(64, num_nz, 1, 1, device=device)
    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.
    # Setup Adam optimizers for both G and D
    lr = lr_rate
    beta1 = 0.5
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    #optimizerD = optim.SGD(netD.parameters(), lr=lr, momentum=0.9)
    #optimizerG = optim.SGD(netG.parameters(), lr=lr, momentum=0.9)


    gan_tr_data_load = Data_loader(gan_tr_data, gan_tr_lbl, batch_size=batch_size, shuffle=True)
    #print(len(gan_tr_data_load))
    print("Starting Training Loop...")
    for epoch in range(num_epochs):

        gan_tr_data_load.shuffle_ftn()

        for i, (data, y) in enumerate(gan_tr_data_load):

            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            netD.zero_grad()
            # Format batch
            b_size = data.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # Forward pass real batch through D
            output = netD(data)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, num_nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            # (2) Update G network: minimize -log(D(G(z)))
            netG.zero_grad()
            label.fill_(real_label)  
            output = netD(fake).view(-1)
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            with torch.no_grad():
                label.fill_(fake_label)  
                errG2 = criterion(output,label) + errD_real # -log(D(x)) - log(1-D(G(z)))
            # Output training stats
            if iters % rec_step == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, i, len(gan_tr_data_load), errD.item(), errG2.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG2.item())
            D_losses.append(errD.item())
            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % rec_step == 0) or (epoch == num_epochs-1): # and (i == len(gan_tr_data_load)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
                

            iters += 1
            
    return netG, netD, G_losses, D_losses, fake, img_list



