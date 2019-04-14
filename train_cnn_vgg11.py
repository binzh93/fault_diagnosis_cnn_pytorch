import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import transforms
import time
import os
import copy
import torch.utils.data as Data
from torch.autograd import Variable
from my_data_loader import MyDataLoader
# from my_net import *
from my_cat_net import *


print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

model_name = "vgg11"


def train_model(model, dataloaders, criterion, optimizer, scheduler=None, num_epochs=25):
    since = time.time()
    
    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []

    val_acc_history = []
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                inputs, labels = Variable(inputs), Variable(labels)
#                 print(inputs.shape)
#                 print(labels.shape)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            if phase == 'train':
                train_loss_list.append(epoch_loss)
                train_acc_list.append(epoch_acc)
            else:
                val_loss_list.append(epoch_loss)
                val_acc_list.append(epoch_acc)
#             print(train_acc_list)
#             print(type(train_acc_list[0]))
  
                
            
            
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print("save model...")
        model_save_path = "/workspace/mnt/group/face/zhubin/alg_code/fault_diagnosis_cnn_pytorch/model/"+ model_name + "_" + str(epoch+1) + ".pth"
        torch.save(model, model_save_path)
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    with open("model/train_loss.txt", "w") as fw:
        for val in train_loss_list:
            fw.write(str(val) + "\n")
            
    with open("model/train_acc.txt", "w") as fw:
        for val in train_acc_list:
            fw.write(str(val) + "\n")
            
    with open("model/val_loss.txt", "w") as fw:
        for val in val_loss_list:
            fw.write(str(val) + "\n")
            
    with open("model/val_acc.txt", "w") as fw:
        for val in val_acc_list:
            fw.write(str(val) + "\n")


    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


input_size = 48

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((input_size, input_size)),
        # transforms.RandomResizedCrop(input_size),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((input_size, input_size)),
        # transforms.CenterCrop(input_size),
        transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}



print("Initializing Datasets and Dataloaders...")

# # Create training and validation datasets
# image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
# # Create training and validation dataloaders
# dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}


# isCaseW = True
isCaseW = False
if isCaseW:
    img_root_dir = ""
    train_txt_path = "/workspace/mnt/group/face/zhubin/alg_code/fault_diagnosis_cnn_pytorch/CaseW_train_data_file_1/train.txt"
    val_txt_path = "/workspace/mnt/group/face/zhubin/alg_code/fault_diagnosis_cnn_pytorch/CaseW_train_data_file_1/val.txt"
    train_batch_size = 75
    test_batch_size = 10
    num_calsses = 10
    # lr = 0.1  weight_decay=0.0005
else:
    img_root_dir = ""
    train_txt_path = "/workspace/mnt/group/face/zhubin/alg_code/fault_diagnosis_cnn_pytorch/jiangnan_train_data_file_1/train.txt"
    val_txt_path = "/workspace/mnt/group/face/zhubin/alg_code/fault_diagnosis_cnn_pytorch/jiangnan_train_data_file_1/val.txt"
    train_batch_size = 72
    test_batch_size = 10
    num_calsses = 4
    # lr = 0.1  weight_decay=0.0005

train_dataset = MyDataLoader(img_root=img_root_dir, txt_file=train_txt_path, transforms=data_transforms["train"], isCaseW=isCaseW)
train_dataloader = Data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)

test_dataset = MyDataLoader(img_root=img_root_dir, txt_file=val_txt_path, transforms=data_transforms["val"], isCaseW=isCaseW)
test_dataloader = Data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True)


data_loader = {"train": train_dataloader, "val": test_dataloader}


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = MyNet(num_calsses=num_calsses)
net = net.to(device)
print(net)

params_to_update = net.parameters()
# 0.005
optimizer_ft = optim.SGD(params_to_update, lr=0.1, momentum=0.9, weight_decay=0.0005)

# Decay LR by a factor of 0.1 every 40 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=80, gamma=0.1)





# Setup the loss fxn
criterion = nn.CrossEntropyLoss()

num_epochs = 80

# Train and evaluate
model_ft, hist = train_model(net, data_loader, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=num_epochs)


