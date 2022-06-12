import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import h5py
import pdb
import sys
np.random.seed(1337)
torch.manual_seed(1)
print(torch.__version__)
import matplotlib.pyplot as plt
import argparse

def set_the_device(device_number):
    torch.cuda.device_count()
    cuda0 = torch.cuda.set_device(device_number)
    torch.cuda.get_device_name(0)

def load_train_val_set():
    # Load the FBMS59 Dataset (Produced Trajectories)
    trainvalFile = h5py.File('/with_torch/data/fbms/dataset.h5', 'r')
    temp_1 = np.asarray(trainvalFile['training']['train_pairs_for_seq_0']).T
    temp_2 = np.ones((temp_1.shape[0], temp_1.shape[1]+1))*0
    temp_2[:,:-1] = temp_1
    trainval_pairsset = np.copy(temp_2)
    for ii in range(1, 29):
        temp_1 = np.asarray(trainvalFile['training']['train_pairs_for_seq_' + str(ii)]).T
        temp_2 = np.ones((temp_1.shape[0], temp_1.shape[1]+1))*ii
        temp_2[:,:-1] = temp_1
        trainval_pairsset = np.concatenate((trainval_pairsset, temp_2))
    
    main_trainvalTrajsSet = []
    print('Load trajectories from HDF5 file into memory for each sequence (train and val)')
    print('It takes some time ...')

    for id_1 in range(29): # 29 sequences on the train set of FBMS59 dataset
      print(id_1)
      temp_data = []
      for id_2 in range(len(trainvalFile['training']['tracks_for_seq_' + str(id_1)])):
        temp_data.append(np.asarray(trainvalFile['training']['tracks_for_seq_'+ str(id_1)]['tracks_' + str(id_2)]['data']).T.astype('float32'))
      main_trainvalTrajsSet.append(temp_data)
    trainvalTrajsSet = main_trainvalTrajsSet
    trainvalFile.close()
 
    return trainval_pairsset, trainvalTrajsSet

def get_trainvalset_for_model(pairsSet, trajsSet, datahalf):
    seq_pairs = []

    for ii in range(50):
        pos = np.where(pairsSet[:,3]==ii)  
        seq_pair = pairsSet[pos[0]]
        
        if datahalf == 0:
            seq_pairs.append(seq_pair[0:int(seq_pair.shape[0]*0.8),:])
        else:
            seq_pairs.append(seq_pair[int(seq_pair.shape[0]*0.8+1.0):,:])

    # group the seq_pairs
    B = seq_pairs[0]
    for ii in range(1,50):
        B = np.r_[B, seq_pairs[ii]]

    seq_pairs = np.copy(B)
    return seq_pairs

def group_traj_lengths(pairsSet, trajsSet):
  
    all_pairs  = []

    pairs_16   = []
    pairs_32   = []
    pairs_64   = []
    pairs_128  = []
    pairs_256  = []
    pairs_512  = []
    pairs_1024 = []

    pairsSet = pairsSet[0]

    radius = 3
    for idx in range(pairsSet.shape[0]):
        
        left_traj_frames     =  ((np.asarray(trajsSet[pairsSet[idx][3].astype('int')][pairsSet[idx][0].astype('int')])))[:,2].astype('int')
        right_traj_frames    =  ((np.asarray(trajsSet[pairsSet[idx][3].astype('int')][pairsSet[idx][1].astype('int')])))[:,2].astype('int')
        intersectionOfFrames = np.intersect1d(left_traj_frames, right_traj_frames)

        if (len(intersectionOfFrames)-radius)/16. <= 1.0:
            pairs_16.append(pairsSet[idx])
        elif (len(intersectionOfFrames)-radius)/32. <= 1.0:
            pairs_32.append(pairsSet[idx])
        elif (len(intersectionOfFrames)-radius)/64. <= 1.0:
            pairs_64.append(pairsSet[idx])
        elif (len(intersectionOfFrames)-radius)/128. <= 1.0:
            pairs_128.append(pairsSet[idx])
        elif (len(intersectionOfFrames)-radius)/256. <= 1.0:
            pairs_256.append(pairsSet[idx])
        elif (len(intersectionOfFrames)-radius)/512. <= 1.0:
            pairs_512.append(pairsSet[idx])
        elif (len(intersectionOfFrames)-radius)/1024. <= 1.0:
            pairs_1024.append(pairsSet[idx])
            
    all_pairs.append(pairs_16)
    all_pairs.append(pairs_32)
    all_pairs.append(pairs_64)
    all_pairs.append(pairs_128)
    all_pairs.append(pairs_256)
    all_pairs.append(pairs_512)
    all_pairs.append(pairs_1024)

    return all_pairs

# produce trajs set 
class TrajDataset(data.Dataset):
    """Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, trainPairsSet, trainTrajsSet):
        
        self.trainPairsSet  = trainPairsSet
        self.trainTrajsSet  = trainTrajsSet

    def __getitem__(self, index):
        """Returns one data pair (traj_left_motion, traj_right_motion, label)."""
        trainPairsSet = self.trainPairsSet
        
        trainTrajsSet = self.trainTrajsSet

        idx = index
 
        left_traj_frames    =  ((np.asarray(trainTrajsSet[trainPairsSet[idx][3].astype('int')][trainPairsSet[idx][0].astype('int')])))[:,2].astype('int')
        right_traj_frames   =  ((np.asarray(trainTrajsSet[trainPairsSet[idx][3].astype('int')][trainPairsSet[idx][1].astype('int')])))[:,2].astype('int')
        
        intersectionOfFrames = np.intersect1d(left_traj_frames, right_traj_frames)
 
        while len(intersectionOfFrames) < 5:
            idx = int(np.floor(1000*np.random.rand()) + 1)

            left_traj_frames    =  ((np.asarray(trainTrajsSet[trainPairsSet[idx][3].astype('int')][trainPairsSet[idx][0].astype('int')])))[:,2].astype('int')
            right_traj_frames   =  ((np.asarray(trainTrajsSet[trainPairsSet[idx][3].astype('int')][trainPairsSet[idx][1].astype('int')])))[:,2].astype('int')
            intersectionOfFrames = np.intersect1d(left_traj_frames, right_traj_frames)

        if len(intersectionOfFrames) >= 5:
            startIndLeft  = np.where(left_traj_frames == intersectionOfFrames[0])[0][0]
            endIndLeft    = np.where(left_traj_frames == intersectionOfFrames[-1])[0][0]+1
            startIndRight = np.where(right_traj_frames == intersectionOfFrames[0])[0][0]
            endIndRight   = np.where(right_traj_frames == intersectionOfFrames[-1])[0][0]+1
                
            left_vals     = ((np.asarray(trainTrajsSet[trainPairsSet[idx][3].astype('int')][trainPairsSet[idx][0].astype('int')]))[:,0:2].astype('float32'))
            left_flowvar  = ((np.asarray(trainTrajsSet[trainPairsSet[idx][3].astype('int')][trainPairsSet[idx][0].astype('int')]))[:,3].astype('float32'))
                
            right_vals    = ((np.asarray(trainTrajsSet[trainPairsSet[idx][3].astype('int')][trainPairsSet[idx][1].astype('int')]))[:,0:2].astype('float32'))
            right_flowvar = ((np.asarray(trainTrajsSet[trainPairsSet[idx][3].astype('int')][trainPairsSet[idx][1].astype('int')]))[:,3].astype('float32'))
    
            left_vals     = left_vals[startIndLeft:endIndLeft]
            left_flowvar  = left_flowvar[startIndLeft:endIndLeft]
                
            right_vals    = right_vals[startIndRight:endIndRight]
            right_flowvar = right_flowvar[startIndRight:endIndRight]

            left_vals_xy    = np.copy(left_vals)
            right_vals_xy   = np.copy(right_vals)
                
            radius = 3
            if left_vals.shape[0]<radius:
                radius = left_vals.shape[0]

            sum_xy_left  = np.sum(left_vals, axis=0)*1./left_vals.shape[0]
            sum_xy_right = np.sum(right_vals, axis=0)*1./right_vals.shape[0]
        
            left_motion   = [left_vals[i+radius]-left_vals[i] for i in range(left_vals.shape[0]-radius)]
            right_motion  = [right_vals[i+radius]-right_vals[i] for i in range(right_vals.shape[0]-radius)]
            left_motion   = np.asarray(left_motion)
            right_motion  = np.asarray(right_motion)

            left_flowvar_diff  = [left_flowvar[i+radius]-left_flowvar[i] for i in range(left_vals.shape[0]-radius)]
            right_flowvar_diff = [right_flowvar[i+radius]-right_flowvar[i] for i in range(left_vals.shape[0]-radius)]

            min_flowvar = np.min([left_flowvar_diff, right_flowvar_diff], axis=0)
            min_flowvar[np.where(min_flowvar<0.1)] = 0.1

            left_motion[:,0]  = np.divide(left_motion[:,0], min_flowvar)
            left_motion[:,1]  = np.divide(left_motion[:,1], min_flowvar)

            right_motion[:,0] = np.divide(right_motion[:,0], min_flowvar)
            right_motion[:,1] = np.divide(right_motion[:,1], min_flowvar)

            left_right_distance = np.sqrt((left_motion[:,0] - right_motion[:,0])**2 + (left_motion[:,1] - right_motion[:,1])**2)
            max_index = np.argmax(left_right_distance)
            
            begin_index = max_index - 12
            end_index   = max_index + 12 + 1

            temp_length = np.copy(left_motion.shape[0])
        
            if begin_index < 0:
                left_motion_0  = np.pad(left_motion[:, 0], (-begin_index, 0), 'constant', constant_values=left_motion[0, 0])
                left_motion_1  = np.pad(left_motion[:, 1], (-begin_index, 0), 'constant', constant_values=left_motion[0, 1])
                right_motion_0 = np.pad(right_motion[:, 0], (-begin_index, 0), 'constant', constant_values=right_motion[0, 0])
                right_motion_1 = np.pad(right_motion[:, 1], (-begin_index, 0), 'constant', constant_values=right_motion[0, 1])
                left_motion  = np.asarray([left_motion_0, left_motion_1]).T
                right_motion = np.asarray([right_motion_0, right_motion_1]).T
                max_index += (-1*begin_index)
            
            if end_index > temp_length:
                left_motion_0  = np.pad(left_motion[:, 0], (0, end_index - temp_length), 'constant', constant_values=left_motion[-1, 0])
                left_motion_1  = np.pad(left_motion[:, 1], (0, end_index - temp_length), 'constant', constant_values=left_motion[-1, 1])
                right_motion_0 = np.pad(right_motion[:, 0], (0, end_index - temp_length), 'constant', constant_values=right_motion[-1, 0])
                right_motion_1 = np.pad(right_motion[:, 1], (0, end_index - temp_length), 'constant', constant_values=right_motion[-1, 1])
                left_motion  = np.asarray([left_motion_0, left_motion_1]).T
                right_motion = np.asarray([right_motion_0, right_motion_1]).T     
            
            begin_index = max_index - 12
            end_index   = max_index + 12 + 1

            left_motion_divided  = left_motion[begin_index:end_index, :]   
            right_motion_divided = right_motion[begin_index:end_index, :]     
            
        else:
            print('this must not happen!!')

        X_train_motion_left  = [left_motion_divided]
        X_train_motion_right = [right_motion_divided]

        Y_train = trainPairsSet[idx][2].astype('float32')

        return torch.from_numpy(np.asarray(X_train_motion_left)), torch.from_numpy(np.asarray(X_train_motion_right)), torch.from_numpy(np.asarray(Y_train))

    def __len__(self):
        return len(self.trainPairsSet)

# produce trajs set 
class TrajDataset_Test(data.Dataset):
    """Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, trainPairsSet, trainTrajsSet):
        
        self.trainPairsSet  = trainPairsSet
        self.trainTrajsSet  = trainTrajsSet

    def __getitem__(self, index):
        """Returns one data pair (traj_left_motion, traj_right_motion, label)."""
        trainPairsSet = self.trainPairsSet
        trainTrajsSet = self.trainTrajsSet

        idx = index
 
        left_traj_frames    =  ((np.asarray(trainTrajsSet[trainPairsSet[idx][0].astype('int')].T)))[:,2].astype('int')
        right_traj_frames   =  ((np.asarray(trainTrajsSet[trainPairsSet[idx][1].astype('int')].T)))[:,2].astype('int')
        
        intersectionOfFrames = np.intersect1d(left_traj_frames, right_traj_frames)
 
        if len(intersectionOfFrames) < 5:
        
            left_motion_divided = np.zeros((25,2)).astype('float32')
            right_motion_divided = np.zeros((25,2)).astype('float32')

        else:
            startIndLeft  = np.where(left_traj_frames == intersectionOfFrames[0])[0][0]
            endIndLeft    = np.where(left_traj_frames == intersectionOfFrames[-1])[0][0]+1
            startIndRight = np.where(right_traj_frames == intersectionOfFrames[0])[0][0]
            endIndRight   = np.where(right_traj_frames == intersectionOfFrames[-1])[0][0]+1
                
            left_vals     = ((np.asarray(trainTrajsSet[trainPairsSet[idx][0].astype('int')].T))[:,0:2].astype('float32'))
            left_flowvar  = ((np.asarray(trainTrajsSet[trainPairsSet[idx][0].astype('int')].T))[:,3].astype('float32'))
                
            right_vals    = ((np.asarray(trainTrajsSet[trainPairsSet[idx][1].astype('int')].T))[:,0:2].astype('float32'))
            right_flowvar = ((np.asarray(trainTrajsSet[trainPairsSet[idx][1].astype('int')].T))[:,3].astype('float32'))
    
            left_vals     = left_vals[startIndLeft:endIndLeft]
            left_flowvar  = left_flowvar[startIndLeft:endIndLeft]
                
            right_vals    = right_vals[startIndRight:endIndRight]
            right_flowvar = right_flowvar[startIndRight:endIndRight]

            left_vals_xy    = np.copy(left_vals)
            right_vals_xy   = np.copy(right_vals)
                
            radius = 3
            if left_vals.shape[0]<radius:
                radius = left_vals.shape[0]

            sum_xy_left  = np.sum(left_vals, axis=0)*1./left_vals.shape[0]
            sum_xy_right = np.sum(right_vals, axis=0)*1./right_vals.shape[0]
        
            left_motion   = [left_vals[i+radius]-left_vals[i] for i in range(left_vals.shape[0]-radius)]
            right_motion  = [right_vals[i+radius]-right_vals[i] for i in range(right_vals.shape[0]-radius)]
            left_motion   = np.asarray(left_motion)
            right_motion  = np.asarray(right_motion)

            left_flowvar_diff  = [left_flowvar[i+radius]-left_flowvar[i] for i in range(left_vals.shape[0]-radius)]
            right_flowvar_diff = [right_flowvar[i+radius]-right_flowvar[i] for i in range(left_vals.shape[0]-radius)]

            min_flowvar = np.min([left_flowvar_diff, right_flowvar_diff], axis=0)
            min_flowvar[np.where(min_flowvar<0.1)] = 0.1

            left_motion[:,0]  = np.divide(left_motion[:,0], min_flowvar)
            left_motion[:,1]  = np.divide(left_motion[:,1], min_flowvar)

            right_motion[:,0] = np.divide(right_motion[:,0], min_flowvar)
            right_motion[:,1] = np.divide(right_motion[:,1], min_flowvar)

            left_right_distance = np.sqrt((left_motion[:,0] - right_motion[:,0])**2 + (left_motion[:,1] - right_motion[:,1])**2)
            max_index = np.argmax(left_right_distance)
            
            begin_index = max_index - 12
            end_index   = max_index + 12 + 1

            temp_length = np.copy(left_motion.shape[0])
        
            if begin_index < 0:
                left_motion_0  = np.pad(left_motion[:, 0], (-begin_index, 0), 'constant', constant_values=left_motion[0, 0])
                left_motion_1  = np.pad(left_motion[:, 1], (-begin_index, 0), 'constant', constant_values=left_motion[0, 1])
                right_motion_0 = np.pad(right_motion[:, 0], (-begin_index, 0), 'constant', constant_values=right_motion[0, 0])
                right_motion_1 = np.pad(right_motion[:, 1], (-begin_index, 0), 'constant', constant_values=right_motion[0, 1])
                left_motion  = np.asarray([left_motion_0, left_motion_1]).T
                right_motion = np.asarray([right_motion_0, right_motion_1]).T
                max_index += (-1*begin_index)
            
            if end_index > temp_length:
                left_motion_0  = np.pad(left_motion[:, 0], (0, end_index - temp_length), 'constant', constant_values=left_motion[-1, 0])
                left_motion_1  = np.pad(left_motion[:, 1], (0, end_index - temp_length), 'constant', constant_values=left_motion[-1, 1])
                right_motion_0 = np.pad(right_motion[:, 0], (0, end_index - temp_length), 'constant', constant_values=right_motion[-1, 0])
                right_motion_1 = np.pad(right_motion[:, 1], (0, end_index - temp_length), 'constant', constant_values=right_motion[-1, 1])
                left_motion  = np.asarray([left_motion_0, left_motion_1]).T
                right_motion = np.asarray([right_motion_0, right_motion_1]).T     
            
            begin_index = max_index - 12
            end_index   = max_index + 12 + 1

            left_motion_divided  = left_motion[begin_index:end_index, :]   
            right_motion_divided = right_motion[begin_index:end_index, :]     
            
        X_train_motion_left  = [left_motion_divided]
        X_train_motion_right = [right_motion_divided]

        Y_train = 1.0

        return torch.from_numpy(np.asarray(X_train_motion_left)), torch.from_numpy(np.asarray(X_train_motion_right)), torch.from_numpy(np.asarray(Y_train))

    def __len__(self):
        return len(self.trainPairsSet)

def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (left_motion, rigth_motion, label).
    
    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.
    Args:
        data: list of tuple (left_motion, rigth_motion, label). 
            - left_motion, rigth_motion: torch tensor of shape (64, 2).
            - label: torch tensor of shape 1
    """
    
    motion_left, motion_right, label = zip(*data)

    left_motions = torch.stack(motion_left, 0)
    right_motions = torch.stack(motion_right, 0)

    labels = torch.stack(label, 0)

    return left_motions, right_motions, labels

def get_loader(batch_size, shuffle, num_workers, trainPairsSet, trainTrajsSet, small_data, is_train_set, is_test_set):
    """Returns torch.utils.data.DataLoader for custom trajectories dataset."""
    # trajectories dataset
    if is_test_set == False:
        trajs_set = TrajDataset(trainPairsSet, trainTrajsSet)
    else:
        trajs_set = TrajDataset_Test(trainPairsSet, trainTrajsSet)
    
    # This will return (motion_left, motion_right) for each iteration.
    # motion_left, notion_right: a tensor of shape (batch_size, 64, 2).
    data_loader = torch.utils.data.DataLoader(dataset=trajs_set, 
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader

# create the model
class motion_GRU(nn.Module):
    def __init__(self): 
        super(motion_GRU, self).__init__()
        
        self.gru = nn.GRU(2, 2, batch_first=True, dropout=0.05, num_layers=2, bidirectional=True) 
        self.norm1 = nn.BatchNorm1d(10)
        self.drop1 = nn.Dropout(0.01)
        self.linear1 = nn.Linear(25, 25)
        self.linear2 = nn.Linear(25, 1)
        self.sigmoid1 = nn.Sigmoid()
        self.relu1 = nn.ReLU()
         
    def forward_once(self, x):
        output, h = self.gru(x)
        return output 
    
    def forward(self, input_1, input_2): 

        out_1 = self.forward_once(input_1)
        out_2 = self.forward_once(input_2)

        diff = torch.sum((out_1 - out_2)**2, axis=-1)

        out_ = self.sigmoid1(self.linear2(self.linear1(diff)))

        out = torch.squeeze(out_)

        return out

# cut gradients
class UnitNormClipper(object):
    def __init__(self, frequency=5):
        self.frequency = frequency

    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, 'weight'):
            w = module.weight.data
            w.div_(torch.norm(w, 2, 1).expand_as(w))

# initialize the model
def initialize_model(model):
    for name, param in model.named_parameters():
      if 'bias' in name:
         nn.init.constant_(param, 0.0)
      elif 'weight' in name:
         nn.init.xavier_normal_(param)
    return model

def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)

def gradClamp(parameters, clip=5):
    for p in parameters:
        p.grad.data.clamp_(max=clip)

def train_the_model(model, optimizer, train_pairs_group, seq_trajs, val_pairs_group, args):
    shuffle=True
    num_workers= int(args['number_of_workers'])
    batch_size = int(args['batch_size'])
    num_of_epochs = int(args['number_of_epochs'])

    criterion = nn.MSELoss()
    if len(train_pairs_group) != 0:
        seq_pairs = np.copy(train_pairs_group)
        trainloader = get_loader(batch_size, shuffle, num_workers, seq_pairs, seq_trajs, small_data=False, is_train_set=True, is_test_set=False)
        
        for epoch in range(num_of_epochs):
            running_loss = 0.0
            for i_, data_ in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                input_left, input_right, labels = data_
                
                input_left  = torch.squeeze(input_left)
                input_right = torch.squeeze(input_right) 

                input_left  = input_left.cuda()
                input_right = input_right.cuda()
                
                labels = labels.cuda()
                
                # zero the parameter gradients
                optimizer.zero_grad()
                
                outputs = model(input_left, input_right)
                loss = criterion(outputs, labels)
                loss.backward()
                
                if loss!=loss:
                    print('\n\nNaN loss value!\n\n')
                    pdb.set_trace()

                weight_list = []
                grad_list = []
                for name in model.named_parameters():
                    weight_list.append(name)
                    grad_list.append(name[1].grad)

                optimizer.step()

                # print statistics
                running_loss += loss.item()
                
                outputs_ = np.round(outputs.cpu().detach().numpy())
                labels_  = labels.cpu().detach().numpy()
                sum_ = np.sum(outputs_ == labels_)


                if i_%50 == 49:           
                    print('accuracy: ', 100.0 * sum_/labels_.shape[0])
                    print('[%d, %5d] loss: %.3f' %
                            (epoch + 1, i_ + 1, running_loss / 50))
                    running_loss = 0.0

            evaluate_the_models(model, val_pairs_group, seq_trajs, args)

        print('\n\nFinished Training\n\n')

def evaluate_the_models(model, val_pairs_group, seq_trajs, args):
    
    shuffle=False
    num_workers=int(args['number_of_workers'])
    batch_size =int(args['batch_size'])
    
    criterion = nn.MSELoss()

    if len(val_pairs_group[0]) != 0: 

        seq_pairs = np.copy(val_pairs_group)
        valloader = get_loader(batch_size, shuffle, num_workers, seq_pairs, seq_trajs, small_data=False, is_train_set=False, is_test_set=False)
        
        with torch.no_grad():
          final_accuracy = 0.0
          final_loss = 0.0
          counter = 0.0
          for epoch in range(1): 
              running_loss = 0.0
              for i_, data_ in enumerate(valloader, 0):
                  
                  # get the inputs; data is a list of [inputs, labels]
                  input_left, input_right, labels = data_
                  input_left  = torch.squeeze(input_left)
                  input_right = torch.squeeze(input_right)

                  input_left = input_left.cuda()
                  input_right = input_right.cuda()
                  labels = labels.cuda()
                
                  outputs = model(input_left, input_right)

                  loss = criterion(outputs, labels)
                
                  if loss!=loss:
                      print('\n\nNaN loss value!\n\n')
                      pdb.set_trace()

                  # print statistics
                  running_loss += loss.item()
                  final_loss += running_loss

                  outputs_ = np.round(outputs.cpu().detach().numpy())
                  labels_  = labels.cpu().detach().numpy()
                  sum_ = np.sum(outputs_ == labels_)

                  final_accuracy += 100.0 * sum_/labels_.shape[0]  
                  counter += 1.0
                  running_loss = 0.0

        print('validation accuracy: ', final_accuracy/counter)
        print('validation loss: ', final_loss/counter)
                  
        print('Finished Validation Evaluation')

def print_train_pairs(seq_pairs_train, train_val_set_id):
    file1 = open('../output/train_pairs_sequence_' + str(train_val_set_id) + '.txt','w+')
    pairs = seq_pairs_train[0]
    for ii in range(pairs.shape[0]):
        file1.write('%i ' % pairs[ii][0])
        file1.write('%i\n' % pairs[ii][1])

def maxOfIntersectionParts(test_predictions, trajPairsMatrix, firstShape):

    final_results = np.ones(firstShape)*-1

    for idx in range(firstShape, test_predictions.shape[0]):
        test_predictions[int(trajPairsMatrix[idx,3])] = max([test_predictions[idx], test_predictions[int(trajPairsMatrix[idx,3])]])
        final_results[int(trajPairsMatrix[idx,3])] = test_predictions[int(trajPairsMatrix[idx,3])] 
    
    for idx in range(firstShape):
        if final_results[idx] < 0:
            final_results[idx] = test_predictions[idx]
    
    return_results = []
    for idx in range(firstShape):
        return_results.append([final_results[idx]])
    
    return return_results

def remove_train_pairs(test_predictions, seq_pairs_train, testPairsSet):
  
    counter_ = 0
    seq_pairs_train = seq_pairs_train[0][:,0:2]
    test_pairs = testPairsSet[:,0:2]
 
    for ii in range(seq_pairs_train.shape[0]):
        if ii % 2000 == 0:
            print(ii)

        data_pairs = seq_pairs_train[ii]
        pos = np.argwhere(test_pairs[:,0]==data_pairs[0])
        pos_1 = np.argwhere(test_pairs[pos][:,0,1]==data_pairs[1])
        if len(pos) > 0:
            test_predictions[pos[pos_1]] = -10.0
            counter_+=1
            continue

        pos = np.argwhere(test_pairs[:,0]==data_pairs[1])
        pos_1 = np.argwhere(test_pairs[pos][:,0,1]==data_pairs[0])
        if len(pos) > 0:
            test_predictions[pos[pos_1]] = -10.0
            counter_+=1
            continue
           
    return test_predictions

def produce_costs_for_graph(model, seq_pairs_train, args):
    shuffle = False
    num_workers=int(args['number_of_workers'])
    batch_size =int(args['batch_size'])
   
    for seq_id in range(30): 
                
        testsetseqs = h5py.File('/with_torch/data/fbms/dataset.h5', 'r')
        testPairsSet = np.asarray(testsetseqs['testing']['all_pairs_for_seq_' + str(seq_id)]).T

        print('Load pairsSet and trajsSet from HDF5 file for sequence ' + str(seq_id+1))

        print('It may take some time ...')
        testTrajsSet = testsetseqs['testing']['tracks_for_seq_'+str(seq_id)]

        temp_data = []
        for id_2 in range(len(testTrajsSet)):
            temp_data.append(np.asarray(testTrajsSet['tracks_' + str(id_2)]['data']).astype('float32'))

        testTrajsSet = temp_data
        testsetseqs.close()
 
        trajPairsMatrix = testPairsSet

        trajPairsMatrix = np.c_[trajPairsMatrix, np.ones(trajPairsMatrix.shape[0])*-1] 

        firstShape = trajPairsMatrix.shape[0]

        lastShape = trajPairsMatrix.shape[0]

        # Notice: network output is ==> 0: two trajs are most similar, 1: two trajs are most dissimilar
        print('prediction by network ...')
         
        testloader = get_loader(batch_size, shuffle, num_workers, trajPairsMatrix, testTrajsSet, small_data=False, is_train_set=False, is_test_set=True)
        
        test_predictions = []
        
        with torch.no_grad():
          for epoch in range(1):  
              running_loss = 0.0
              for i_, data_ in enumerate(testloader, 0):

                  # get the inputs; data is a list of [inputs, labels]
                  input_left, input_right, label = data_
                  input_left  = torch.squeeze(input_left)
                  input_right = torch.squeeze(input_right)
                  
                  input_left = input_left.cuda()
                  input_right = input_right.cuda()
                
                  outputs = model(input_left, input_right)
                    
                  outputs_ = outputs.cpu().detach().numpy()
            
                  test_predictions.extend(outputs_)

        test_predictions = np.asarray(test_predictions)

        print('max of test predicitons: ')
        print(max(test_predictions))

        print('min of test predicitons: ')
        print(min(test_predictions))
        
        with open(('../output/network_sequence_'+ str(seq_id) + '_net_args_fbms_test.txt'), 'w') as ff:
            ff.write(str(args))

	# results to be saved and given to the multicut solver to decompose the graph 
        np.savetxt(('../output/network_output_'+ str(seq_id) + '_fbms_test.txt'), test_predictions) 