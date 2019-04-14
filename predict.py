import torch
from PIL import Image
from torchvision import transforms
from torch.autograd import Variable
import os, time





def load_image(test_txt_file, test_trainsform=None, isCaseW=True):
    img_name_list = []
    label_list    = []
    
    with open(test_txt_file, "r") as fr:
        for line in fr:
            img_path, cls_name = line.strip().split("\t")
            img_name_list.append(img_path)
            label_list.append(int(cls_name))
    
#     if isCaseW:
#         with open(test_txt_file, "r") as fr:
#             for line in fr:
#                 img_path, cls_name = line.strip().split("\t")
#                 img_name_list.append(img_path)
#                 label_list.append(int(cls_name))
#     else:
#         with open(test_txt_file, "r") as fr:
#             for line in fr:
#                 img_path, cls_name = line.strip().split("\t")
#                 img_path = line.strip().split("\t")
#                 img_path = img_path[0]
#                 img_name = os.path.basename(line.strip())
#                 if img_name.startswith("n"):
#                     label = 0
#                 elif img_name.startswith("ob"):
#                     label = 1
#                 elif img_name.startswith("tb"):
#                     label = 2
#                 else:
#                     label = 3     
#                 img_name_list.append(img_path)
#                 label_list.append(label)
    
    for k, v in enumerate(img_name_list):
        img   = Image.open(v)
        img   = test_trainsform(img)
        label = label_list[k]
        yield img, label, v


def predict_image(image, model, device):
    image_tensor = image.unsqueeze_(0)
    input = Variable(image_tensor)
    input = input.to(device)
    output = model(input)
    index = output.data.cpu().numpy().argmax()
    return index



def predict(model_path, test_txt_file, isCaseW=True):
    test_trainsform = transforms.Compose([
#         transforms.Resize((224, 224)),
        transforms.Resize((48, 48)),
        # transforms.RandomResizedCrop(input_size),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_path)
    model.eval()

    correct_nums = 0
    nums = 0
    since = time.time()
    for val in load_image(test_txt_file, test_trainsform, isCaseW):
        img   = val[0]
        label = val[1]
        pre = predict_image(img, model, device)
        if label == pre:
            correct_nums += 1
        nums += 1
    time_elapsed = time.time() - since
    print("time: ", time_elapsed)
    print('Testing complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print("correct_nums: ", correct_nums)
    print("Test nums: ", nums)
    print("Accuracy: ", correct_nums*1.0/nums)
    

if __name__ == "__main__":
    model_path = "/workspace/mnt/group/face/zhubin/alg_code/fault_diagnosis_cnn_pytorch/model/vgg16_8.pth"
    model_path = "/workspace/mnt/group/face/zhubin/alg_code/fault_diagnosis_cnn_pytorch/model/resnet_35.pth"
    model_path = "/workspace/mnt/group/face/zhubin/alg_code/fault_diagnosis_cnn_pytorch/model/alexnet_47.pth"
    model_path = "/workspace/mnt/group/face/zhubin/alg_code/fault_diagnosis_cnn_pytorch/model/vgg11_17.pth"
    
#     isCaseW = True
    isCaseW = False
    if isCaseW:
        test_txt_file = "/workspace/mnt/group/face/zhubin/alg_code/fault_diagnosis_cnn_pytorch/CaseW_train_data_file_1/test.txt"
        predict(model_path, test_txt_file, isCaseW)
    else:
        test_txt_file = "/workspace/mnt/group/face/zhubin/alg_code/fault_diagnosis_cnn_pytorch/jiangnan_train_data_file_1/test.txt"
        predict(model_path, test_txt_file, isCaseW)



