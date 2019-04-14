# -*- coding: utf-8 -*-
import os
import random

##########################
# return train, val, test list
##########################
def split_trainval_test(input_dir):
    all_file = [] 
    for cls_name in os.listdir(input_dir):
        if cls_name == ".DS_Store":
            continue
        file_dir = os.path.join(input_dir, cls_name)
        for img_name in os.listdir(file_dir):
            if img_name.endswith(".jpg"):
                img_path = os.path.join(file_dir, img_name)
                all_file.append(img_path)
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

    for img_path in train:
        train_str += img_path + "\n"
    for img_path in val:
        val_str += img_path + "\n"
    for img_path in val:
        test_str += img_path + "\n"
        
    with open(train_txt_path, "w") as fw:
        fw.write(train_str)
    with open(val_txt_path, "w") as fw:
        fw.write(val_str)
    with open(test_txt_path, "w") as fw:
        fw.write(test_str)
        

if __name__ == "__main__":
    isCaseW = False
    if isCaseW:
        input_dir = "/workspace/mnt/group/face/zhubin/alg_code/fault_diagnosis_cnn/jiangnan_data/jiangnan_data_2500"
        save_dir  = "/workspace/mnt/group/face/zhubin/alg_code/fault_diagnosis_cnn/jiangnan_data/"
        # trainval, test = split_trainval_test(input_dir)
        # generate_tmp_train_val_img(trainval, test, save_dir)
    else:
        input_dir = "/workspace/mnt/group/face/zhubin/alg_code/fault_diagnosis_cnn/jiangnan_data/jiangnan_data_2500"
        save_dir  = "/workspace/mnt/group/face/zhubin/alg_code/fault_diagnosis_cnn_pytorch/jiangnan_train_data_file"
        train, val, test = split_trainval_test(input_dir)
        generate_train_val_test_txt_file(train, val, test, save_dir)
        

    
    