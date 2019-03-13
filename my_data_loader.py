# -*- coding: utf-8 -*-
'''
 # @Author: binzh 
 # @Date: 2018-10-25 14:14:25 
 # @Last Modified by: binzh 
 # @Last Modified time: 2018-11-25 14:14:25 
 '''

# import PIL
# import cv2
import os
import os.path as osp
from PIL import Image
import torch 
from torch.utils import data
from torchvision import datasets, transforms



class MyDataLoader(data.Dataset):
    """
        when define the son class of torch.utils.data.Dataset, len and getitem function must reload.
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
    """
    def __init__(self, img_root, txt_file, transforms=None, isCaseW=False, train=True):

        self.img_list = []
        self.labels = []
        self.img_root = img_root
        self.isCaseW = isCaseW
        self.read_txt_file(txt_file)
        self.transforms = transforms
        
        
    def __getitem__(self, index):
        """
            return one image and label
        """
        img_path = osp.join(self.img_root, self.img_list[index]) 
        img = Image.open(img_path)
        img = self.transforms(img)
        label = self.labels[index]
#         print label
#         return img, float(label)
        return img, label
    
    def __len__(self, ):
        return len(self.img_list)

    def read_txt_file(self, txt_file):
        """
        Args:
            txt_file (str): txt file path
        Operation:
            analysis the filename to get label
            Case Western:
            0 >>>>> 0
            1 >>>>> 1
            2 >>>>> 2
            3 >>>>> 3
            4 >>>>> 4
            5 >>>>> 5
            6 >>>>> 6
            7 >>>>> 7
            8 >>>>> 8
            9 >>>>> 9
            
            jiangnan: 
            n  >>>>> 0
            ob >>>>> 1
            tb >>>>> 2
            ib >>>>> 3
        """
        with open(txt_file, "r") as fr:
            for line in fr:
                img_path, cls_name = line.strip().split("\t")
                temp_label = int(cls_name)
                self.img_list.append(img_path)
                self.labels.append(temp_label)
                    
                    
#         if self.isCaseW:
#             with open(txt_file, "r") as fr:
#                 for line in fr:
#                     img_path, cls_name = line.strip().split("\t")
#                     temp_label = int(cls_name)
#                     self.img_list.append(img_path)
#                     self.labels.append(temp_label)
#         else:
#             with open(txt_file, "r") as fr:
#                 for line in fr:
# #                     img_path, cls_name = line.strip().split("\t")
#                     img_path = line.strip().split("\t")
#                     img_path = img_path[0]
#                     img_name = os.path.basename(img_path)
# #                     if img_name.startswith("n"):
# #                         temp_label = 0
# #                     elif img_name.startswith("ob"):
# #                         temp_label = 1
# #                     elif img_name.startswith("tb"):
# #                         temp_label = 2
# #                     else:
# #                         temp_label = 3

#                     img_path = img_path
#                     self.img_list.append(img_path)
#                     self.labels.append(temp_label)
        


