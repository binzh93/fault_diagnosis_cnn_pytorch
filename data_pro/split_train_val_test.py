# -*- coding: utf-8 -*-
import os
import random

##########################
# return train, val, test list

'''
jiangnan: 
        n  >>>>> 0
        ob >>>>> 1
        tb >>>>> 2
        ib >>>>> 3
'''
##########################
def split_trainval_test(input_dir, isCaseW=True):
    all_file = [] 
    for cls_name in os.listdir(input_dir):
        if cls_name == ".DS_Store":
            continue
        file_dir = os.path.join(input_dir, cls_name)
        if isCaseW:
            for img_name in os.listdir(file_dir):
                if img_name.endswith(".jpg"):
                    img_path = os.path.join(file_dir, img_name)
                    all_file.append([img_path, cls_name])
        else:
            if cls_name.startswith("n"):
                label = '0'
            elif cls_name.startswith("ob"):
                label = '1'
            elif cls_name.startswith("tb"):
                label = '2'
            else:
                label = '3'
            for img_name in os.listdir(file_dir):
                if img_name.endswith(".jpg"):
                    img_path = os.path.join(file_dir, img_name)
                    all_file.append([img_path, label])
    random.shuffle(all_file)
    train = all_file[: int(len(all_file)*0.6)]
    val   = all_file[int(len(all_file)*0.6): int(len(all_file)*0.8)]
    test  = all_file[int(len(all_file)*0.8): ]
    
    return train, val, test

def generate_train_val_test_txt_file(train, val, test, save_dir):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    train_txt_path = os.path.join(save_dir, "train.txt")
    val_txt_path   = os.path.join(save_dir, "val.txt")
    test_txt_path  = os.path.join(save_dir, "test.txt")

    train_str = ""
    val_str   = ""
    test_str  = ""
   
    for img_path, cls_name in train:
        train_str += img_path + "\t" + cls_name + "\n"
    for img_path, cls_name in val:
        val_str += img_path + "\t" + cls_name + "\n"
    for img_path, cls_name in test:
        test_str += img_path + "\t" + cls_name + "\n"
    with open(train_txt_path, "w") as fw:
        fw.write(train_str)
    with open(val_txt_path, "w") as fw:
        fw.write(val_str)
    with open(test_txt_path, "w") as fw:
        fw.write(test_str)
        

if __name__ == "__main__":
#     isCaseW = True
    isCaseW = False
    if isCaseW:
        input_dir = "/workspace/mnt/group/face1/zhubin/alg_code/fault_diagnosis_cnn/CaseW_data/CaseW_raw_data"
        save_dir  = "/workspace/mnt/group/face1/zhubin/alg_code/fault_diagnosis_cnn_pytorch/CaseW_train_data_file_1"
        train, val, test = split_trainval_test(input_dir, isCaseW)
        generate_train_val_test_txt_file(train, val, test, save_dir)
    else:
        input_dir = "/workspace/mnt/group/face1/zhubin/alg_code/fault_diagnosis_cnn/jiangnan_data/jiangnan_data_2500"
        save_dir  = "/workspace/mnt/group/face1/zhubin/alg_code/fault_diagnosis_cnn_pytorch/jiangnan_train_data_file_1"
        train, val, test = split_trainval_test(input_dir, isCaseW)
        generate_train_val_test_txt_file(train, val, test, save_dir)
        

    

