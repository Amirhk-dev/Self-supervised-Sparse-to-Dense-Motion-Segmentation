import os
import glob
import torch.utils.data as data
import cv2
import numpy as np
import random
import h5py
import pdb
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage

class Dataset(data.Dataset):
    
    def __init__(self, img_files, folder_path, image_shape, seq_name, flipping, n_classes, train_mode=True, include_rgb=True, include_edge=True):
        super(Dataset, self).__init__()
        random.seed(1)

        self.img_shape = image_shape
        self.flipping = flipping
        self.seq_name = seq_name
        self.path_img = folder_path + "/imgs/" + self.seq_name
        self.path_label = folder_path + "/labels_flownet2/" + self.seq_name
        self.path_edgemap = folder_path + "/edgemaps/" + self.seq_name
        
        self.img_files = img_files
        self.train_mode = train_mode
        self.n_classes = n_classes
        
        # options
        self.include_rgb = include_rgb
        self.include_edge = include_edge
        
    def __getitem__(self, index):
        # set path
        img_path = self.path_img + "/" + self.img_files[index]
        mask_path = self.path_label + "/" + self.img_files[index] + '.h5'
        # read file
        image = cv2.imread(img_path)

        # produce sobel image derivative 
        edgemap = self.produceSobelDeriv(image)

        mask = np.expand_dims(np.asarray(h5py.File(mask_path, 'r')['img']).T, 2)
        
        if self.n_classes == 2:
            pos_0 = np.where(mask == 0)
            pos_1 = np.where(mask > 0)

            if self.train_mode == True:
                while len(np.unique(mask))<3 or len(pos_1[0]) < 10:
                    index = np.random.randint(20)
                    img_path = self.path_img + "/" + self.img_files[index]
                    mask_path = self.path_label + "/" + self.img_files[index] + '.h5'
                    # read file
                    image = cv2.imread(img_path)
                    # produce sobel image derivative 
                    edgemap = self.produceSobelDeriv(image)
                    mask = np.expand_dims(np.asarray(h5py.File(mask_path, 'r')['img']).T, 2)
                    # remove alone points corresponding to the fg (those points which are far away from group of point (outliers))
                    pos_fg = np.asarray(np.where(mask[:,:,0] == np.unique(mask[:,:,0])[2]))
                    for ii in range(len(pos_fg[0])):
                        distance_ = self.closest_node(pos_fg[:,ii], pos_fg, ii)
                        if distance_>1000:
                            mask[pos_fg[0,ii], pos_fg[1,ii], 0] = -1

                    pos_0 = np.where(mask == 0)
                    pos_1 = np.where(mask > 0)

        if self.n_classes == 2:
            if np.unique(mask)[1] != 0:
                mask[np.where(mask==np.unique(mask)[1])] = 0.0
            
            if self.train_mode == True:
                pos_fg = np.asarray(np.where(mask[:,:,0] == np.unique(mask[:,:,0])[2]))
                for ii in range(len(pos_fg[0])):
                    distance_ = self.closest_node(pos_fg[:,ii], pos_fg, ii)
                    if distance_>1000:
                        mask[pos_fg[0,ii], pos_fg[1,ii], 0] = -1

        image = cv2.GaussianBlur(image,(5,5),0)
        image = self.normalize(image)

        edgemap = self.normalize(edgemap)
        edgemap = cv2.GaussianBlur(edgemap,(5,5),0)
        edgemap = np.expand_dims(edgemap, axis=2)
        
        if self.n_classes == 2:
            if self.train_mode == True:
                if np.unique(mask)[2] != 1:
                    mask[np.where(mask==np.unique(mask)[2])] = 1.0
            
        image, mask, edgemap = self.swap_axis(image, mask, edgemap)
        
        if self.flipping:
            image, mask, edgemap = self.random_flip(image, mask, edgemap)
        
        result = None
        if self.include_rgb:
            result = image
        if self.include_edge:
            result = np.concatenate((result, edgemap), 0)

        if self.n_classes == 2:
            if self.train_mode == True:
            
                pos_0 = np.where(mask == 0)
                pos_1 = np.where(mask == 1)

        if self.n_classes == 2:
            mask[0, 5:15 ,::10] = 0
            mask[0, -15:-5,::10] = 0
            mask[0,::10, 5:15] = 0
            mask[0,::10, -15:-5] = 0                                      
            masks = np.copy(mask)
        else:
            masks = self.produceOneHotMask(mask, self.n_classes)

        return (result, masks, img_path)

    def __len__(self):
        return len(self.img_files)
    
    # --------------------------------------------------------
    # Utility functions
    # --------------------------------------------------------
    def produceOneHotMask(self, mask, n_classes):
        masks = []
        for ii in range(n_classes):
            pos = np.where(mask[0,:,:] == ii)
            img = np.zeros((mask.shape[1], mask.shape[2]))
            img[pos[0], pos[1]] = 1.0
            masks.append(img)

        return np.asarray(masks)

    def produceSobelDeriv(self, image):

        src_ = cv2.GaussianBlur(image, (3, 3), 0)
        gray_ = cv2.cvtColor(src_, cv2.COLOR_BGR2GRAY)
        grad_x = cv2.Sobel(gray_, cv2.CV_16S, 1, 0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
        grad_y = cv2.Sobel(gray_, cv2.CV_16S, 0, 1, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        grad_ = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    
        return grad_

    def closest_node(self, node, nodes, ii):
        nodes = np.asarray(nodes)
        node = np.expand_dims(node, axis=1)
        dist_2 = np.sum((nodes - node)**2, axis=0)
        dist_2[ii] = 10000.0 #insert large value for node itself (o.w. the value will be 0)
        return np.min(dist_2)
    
    def computeFlowMagAngle(self, flow):
        flow_mag = np.sqrt(flow[0,:,:]**2 + flow[1,:,:]**2)
        if flow_mag.max() - flow_mag.min() > 0.1:
            flow_mag = (flow_mag - flow_mag.min())/(flow_mag.max() - flow_mag.min())
        flow_dir = (np.arctan2(-flow[1,:,:], -flow[0,:,:])*180.0)/np.pi

        flow_dir[np.where(flow_mag<0.1)] = 0.0;

        return np.expand_dims(flow_mag, axis=0), np.expand_dims(flow_dir, axis=0)

    def normalize(self, x, is_255=True):
        if is_255:
            return (x / 255.).astype(np.float32)
        else:
            return (x-np.min(np.min(x)))/(np.max(x)-np.min(x)).astype(np.float32)

    def compute_pixel_diff(self, image, by_channel):
        w, h, c = image.shape
        center_x = int(np.ceil(w/2))
        center_y = int(np.ceil(h/2))
        
        err = image.copy()

        # MSE to center
        err[:,:,0] = (err[:,:,0] - err[center_x][center_y][0]) ** 2
        err[:,:,1] = (err[:,:,1] - err[center_x][center_y][1]) ** 2
        err[:,:,2] = (err[:,:,2] - err[center_x][center_y][2]) ** 2
        
        if by_channel:
            return err
        else:
            return np.expand_dims(np.sum((err[:,:,0], err[:,:,1], err[:,:,2]), axis=0)/3, 2)
        
    def remove_one_px(self, image, mask, diff, flow):
        # cut 1 pixel
        image = image[:-1,:-1,:]
        mask = mask[:-1,:-1,:]        
        diff = diff[:-1,:-1,:]
        flow = flow[:-1,:-1,:]
        
        return image, mask, diff, flow
    
    # convert format for UNet: 
    # h x w x c -> c x h x w
    def swap_axis(self, image, mask, edgemap):

        image = self.hwc_to_chw(image)
        edgemap = self.hwc_to_chw(edgemap)
        mask = self.hwc_to_chw(mask)
        
        return image, mask, edgemap #, diff, flow
    
    def hwc_to_chw(self, image):
        return np.transpose(image, axes=[2, 0, 1])
    
    def chw_to_hwc(self, image):
        return np.transpose(image, axes=[1, 2, 0])

    def random_flip(self, image, mask, edgemap):
        if bool(random.getrandbits(1)):
            image[0,:,:] = cv2.flip(image[0,:,:], 1)
            image[1,:,:] = cv2.flip(image[1,:,:], 1)
            image[2,:,:] = cv2.flip(image[2,:,:], 1)
            mask[0,:,:]   = cv2.flip(mask[0,:,:], 1)
            edgemap[0,:,:]   = cv2.flip(edgemap[0,:,:], 1)
        return image, mask, edgemap
    
    def random_brightness(self, img): 
        if bool(random.getrandbits(1)):
            factor = 0.5
            
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            hsv = np.array(hsv, dtype=np.float64)
            hsv[:, :, 2] = hsv[:, :, 2] * (factor + np.random.uniform()) 
            hsv[:, :, 2][hsv[:, :, 2] > 255] = 255 # reset out of range values
            rgb = cv2.cvtColor(np.array(hsv, dtype=np.uint8), cv2.COLOR_HSV2RGB)
            return rgb
        else:
            return img