import os
import sys
import random
import pandas as pd
import numpy as np
import time
import utils
import dataset
from unet import UNet
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dsets
from torch.autograd import Variable
import matplotlib.pyplot as plt
import pdb

# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def feed_data(model, loader):
    # for testing
    image = []
    mask = []
    out = []
    
    batch_size = -1
    last_batch_left = False
    
    for images, masks, img_path in loader: 
        images = Variable(images.cuda()).float()
        outputs = model(images)
        outputs = torch.sigmoid(outputs)
        outputs = outputs.cpu().data.numpy()     

        images = images.cpu().numpy()
        masks = masks.cpu().numpy()
        
        # get batch size
        if batch_size <0:
            batch_size = images.shape[0]
            
        if batch_size == images.shape[0]:
            image.append(np.transpose(images, axes=[0, 2, 3, 1]))
            mask.append(np.transpose(masks, axes=[0, 2, 3, 1]))
            out.append(np.transpose(outputs, axes=[0, 2, 3, 1]))
        else:
            last_batch_left = True
            images = np.transpose(images, axes=[0, 2, 3, 1])
            masks = np.squeeze(np.transpose(masks, axes=[0, 2, 3, 1]), axis=3)
            outputs = np.squeeze(np.transpose(outputs, axes=[0, 2, 3, 1]), axis=3)
        
    image = np.array(image)
    mask = np.array(mask)
    out = np.array(out)
    
    image = np.reshape(image, (image.shape[0]*image.shape[1], image.shape[2], image.shape[3], image.shape[4]))
    mask = np.reshape(mask, (mask.shape[0]*mask.shape[1], mask.shape[2], mask.shape[3]))
    out = np.reshape(out, (out.shape[0]*out.shape[1], out.shape[2], out.shape[3]))
    
    if last_batch_left:
        image = np.concatenate((image, images), axis=0)
        mask = np.concatenate((mask, masks), axis=0)
        out = np.concatenate((out, outputs), axis=0)
    
    return image, mask, out

def log_data(text, time):
    print(text, "{:.4f}".format(time), "s")
    print("")
    
dict_parameters = dict()    
def write_data_dict(dict_data, path):  
    file = open(path, "w")
    for key in dict_data:
        file.write(key+" "+dict_data[key]+"\n")
    file.close()

def produceReweightedMask(masks):

    pos_0 = np.where(masks == 0) # bg
    pos_1 = np.where(masks == 1) # fg

    len_bg = len(pos_0[0])                    
    len_fg = len(pos_1[0])

    if len_fg < len_bg:

        new_pos_10 = []
        new_pos_11 = []
        new_pos_12 = []
        new_pos_13 = []

        numOfRepeatPerFG = len_bg/len_fg
        numOfRepeatPerFG /= 10
        numOfRepeatPerFG += 1

        for ii in range(len(pos_1[0])):
            for jj in range(numOfRepeatPerFG):

                new_pos_10.append(pos_1[0][ii])
                new_pos_11.append(pos_1[1][ii])
                new_pos_12.append(pos_1[2][ii])
                new_pos_13.append(pos_1[3][ii])

        new_pos_10 = np.asarray(new_pos_10)
        new_pos_11 = np.asarray(new_pos_11)
        new_pos_12 = np.asarray(new_pos_12)
        new_pos_13 = np.asarray(new_pos_13)

        pos_1 = (new_pos_10, new_pos_11, new_pos_12, new_pos_13)

    elif len_bg < len_fg:

        new_pos_00 = []
        new_pos_01 = []
        new_pos_02 = []
        new_pos_03 = []

        numOfRepeatPerBG = len_fg/len_bg
        numOfRepeatPerBG /= 10
        numOfRepeatPerBG += 1

        for ii in range(len(pos_0[0])):
            for jj in range(numOfRepeatPerBG):

                new_pos_00.append(pos_0[0][ii])
                new_pos_01.append(pos_0[1][ii])
                new_pos_02.append(pos_0[2][ii])
                new_pos_03.append(pos_0[3][ii])

        new_pos_00 = np.asarray(new_pos_00)
        new_pos_01 = np.asarray(new_pos_01)
        new_pos_02 = np.asarray(new_pos_02)
        new_pos_03 = np.asarray(new_pos_03)

        pos_0 = (new_pos_00, new_pos_01, new_pos_02, new_pos_03)
 
    mask_linear = np.ones(len(pos_0[0]) + len(pos_1[0]))
    mask_linear[:len(pos_0[0])] = 0.0 

    return mask_linear, pos_0, pos_1

def main(argv):
 
    torch.cuda.device_count()
    cuda0 = torch.cuda.set_device(1)
    torch.cuda.get_device_name(1)
    
    if len(argv) != 7:
        print("Invalid argument. Exit Program")
        quit()
    
    # args
    experiment_name = argv[0]
    batch_size = int(argv[1])
    learning_rate = float(argv[2])
    include_rgb = (argv[4] == "include_rgb")
    include_edge = (argv[5] == "include_edgemap")
    seq_name = argv[6]
    
    n_channels = 0 # default case
    n_classes = int(argv[3])
   
    # +3: (RGB)
    if include_rgb: 
        n_channels += 3 
    if include_edge:
        n_channels += 1
 
    # check
    if not include_rgb:  
        print("Error! No Features selected! Exit.")
        quit()
        
    # -------------------------------------
    # Parameters
    # -------------------------------------
    image_shape = [480, 854, 3]
    limit = None
    val_size = 0

    # hyperparameters
    epochs = 15
    
    # Learning rate decay
    cycle = 5
    patience = 5
    seed = 1
    
    random.seed(1)
    
    path_data_train = "/data_fbms"    
    path_data_test = "/data_fbms"     
        
    # -------------------------------------
    # Create Directory and write parameters
    # -------------------------------------
    os.makedirs(experiment_name) 
    os.makedirs(experiment_name+"/checkpoint")
    os.makedirs(experiment_name+"/res")
    
    #Log file
    try:
        df = pd.read_csv(experiment_name+"/res/loss_evol.csv")
    except:
        print("Gradients file, not found - create file:", experiment_name+"/res/loss_evol.csv")
        df = pd.DataFrame(columns=['train','val', 'iou'])    
        
    # put in dictionary
    dict_parameters["experiment_name"] = experiment_name
    dict_parameters["batch_size"] = str(batch_size)
    dict_parameters["learning_rate"] = str(learning_rate)
    dict_parameters["val_size"] = str(val_size)
    dict_parameters["epochs"] = str(epochs)
    dict_parameters["cycle"] = str(cycle)
    dict_parameters["patience"] = str(patience)
    dict_parameters["include_rgb"] = str(include_rgb)
    dict_parameters["include_edge"] = str(include_edge)
    dict_parameters["n_channels"] = str(n_channels)
    dict_parameters["n_classes"] = str(n_classes)
    dict_parameters["seq_name"] = seq_name
    write_data_dict(dict_parameters, experiment_name+"/Parameters.txt")

    # -------------------------------------
    # Data loader
    # -------------------------------------
    start = time.time()

    # split train into train/validation
    train_names = next(os.walk(os.path.abspath(path_data_train+"/imgs/"+seq_name)))[2]
    train_names.sort()
    
    if limit is not None:
        train_names = train_names[:limit]
    
    train_names = np.array(train_names)
    
    flipping = True
    train_mode = True
    train_dataset = dataset.Dataset(train_names, path_data_train, image_shape, seq_name, flipping, n_classes, train_mode=train_mode, include_rgb=include_rgb, include_edge=include_edge) 
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
     
    print("Training:", len(train_names))

    end = time.time()
    log_data("Prepare_DataLoader", end - start)
     
    # -------------------------------------
    # Model
    # -------------------------------------
    start = time.time()

    #Define model
    if n_classes == 2:
        model = UNet(n_channels=n_channels, n_classes=1)
    else:
        model = UNet(n_channels=n_channels, n_classes=n_classes)

    model.cuda()

    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
        
    print('Total number of parameters: %d' % num_params)
    
    end = time.time()
    log_data("Define_Model", end - start)
    
    # -------------------------------------
    # Training
    # -------------------------------------
    start = time.time()

    if n_classes == 2:
        criterion = nn.BCEWithLogitsLoss()
    elif n_classes > 2:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    cycle_ = 0
    patience_ = 0
    
    best_val = 9999.0
    eval_time = []
    
    for epoch in range(epochs):

        print(epoch)

        if cycle_ < cycle:
            if patience_ < patience:      
                start2 = time.time()
                train_losses = 0
                iteration = 0

                for images, masks, img_path in train_loader:

                    if n_classes == 2:
                        mask_linear, pos_0, pos_1 = produceReweightedMask(masks)
                    else:             
                        pos = np.where(masks == 1)
  
                    images = Variable(images.cuda()).float()
                    outputs = model(images)
                
                    if n_classes == 2:
                        masks = Variable(torch.tensor(mask_linear).cuda()).float()
                        A_ = outputs[pos_0[0], pos_0[1], pos_0[2], pos_0[3]]
                        B_ = outputs[pos_1[0], pos_1[1], pos_1[2], pos_1[3]]
                        outputs = torch.cat((A_, B_), 0)
                    else:
                        mask_linear = masks[0,:,pos[2],pos[3]]
                        masks = mask_linear.long().cuda()    
                        outputs_linear = outputs[0,:,pos[2],pos[3]]
                        masks = torch.max(masks, 0)[1]
                        outputs = outputs_linear.transpose(0,1)

                    loss = criterion(outputs, masks)
                    train_losses += loss.item()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    print(loss)

                threshold = 0.5

                if n_classes == 2:
                    image, mask, out = feed_data(model, train_loader)
                    pos_0 = np.where(mask==0)
                    pos_1 = np.where(mask==1)
                    out[np.where(out<0.5)] = 0
                    out[np.where(out>=0.5)] = 1
                    A = np.where(out[pos_0[0], pos_0[1], pos_0[2]]==0)
                    B = np.where(out[pos_1[0], pos_1[1], pos_1[2]]==1)
                    print(100.0*(len(A[0]+len(B[0])))/(len(pos_0[0])+len(pos_1[0])))   
                    iou =  100.0*(len(A[0]+len(B[0])))/(len(pos_0[0])+len(pos_1[0]))   
                else:
                    iou = -100
                
                patience_ = 0
                utils.save_checkpoint(experiment_name+'/checkpoint/model_'+seq_name+'.pth', model, optimizer)
                    
                if patience_ >= patience:
                    utils.load_checkpoint(experiment_name+'/checkpoint/model_'+seq_name+'.pth', model, optimizer)
                    patience_=0
                    cycle_ += 1
                    for param_group in optimizer.param_groups:
                        learning_rate=learning_rate/10    
                        param_group['lr'] = learning_rate
                
                end2 = time.time()
                training_time = end2 - start2
                eval_time.append(training_time)
                
                # Print Loss
                if n_classes == 2:
                    print("Epoch:", epoch, 
                            "\tTrain Loss: {:.2f}".format(train_losses), 
                            "\tIoU: {:.2f}".format(iou), 
                            "\tTime: {:.2f}".format(training_time/60), 
                            "\tETA: {:.2f}".format((np.mean(eval_time)*(epochs-epoch))/60), 
                            "\tPatience", patience_, 
                            "\tcycle_", cycle_)
                else:
                    print("Epoch:", epoch, 
                            "\tTime: {:.2f}".format(training_time/60), 
                            "\tPatience", patience_, 
                            "\tcycle_", cycle_)
                
                #Save losses
                df.loc[epoch, ['train']] = train_losses

                if n_classes == 2:
                    df.loc[epoch, ['iou']] = iou
                else:
                    df.loc[epoch, ['iou']] = -100

                #Save losses
                df.to_csv(experiment_name+"/res/loss_evol.csv", encoding='utf-8', index=False)          
                
        else:
            print("Early stop at epoch", epoch)
            break
            
    end = time.time()
    log_data("Training", end - start)
         
if __name__ == "__main__":
    main(sys.argv[1:])
