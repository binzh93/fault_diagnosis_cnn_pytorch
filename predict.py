import torch
from PIL import Image
from torchvision import transforms
from torch.autograd import Variable
import os


def load_image(test_txt_file, test_trainsform=None):
    img_name_list = []
    label_list    = []
    with open(test_txt_file) as fr:
        for line in fr:
            img_name = os.path.basename(line.strip())
            if img_name.startswith("n"):
                label = 0
            elif img_name.startswith("ob"):
                label = 1
            elif img_name.startswith("tb"):
                label = 2
            else:
                label = 3     
            img_path = line.strip()
            img_name_list.append(img_path)
            label_list.append(label)
    
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



def predict(model_path, test_txt_file):
    test_trainsform = transforms.Compose([
        transforms.Resize((224, 224)),
        # transforms.RandomResizedCrop(input_size),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_path)
    model.eval()

    correct_nums = 0
    nums = 0
    for val in load_image(test_txt_file, test_trainsform):
        img   = val[0]
        label = val[1]
        pre = predict_image(img, model, device)
        if label == pre:
            correct_nums += 1
        nums += 1
    print("correct_nums: ", correct_nums)
    print("Test nums: ", nums)
    print("Accuracy: ", correct_nums*1.0/nums)

if __name__ == "__main__":
    model_path = "/workspace/mnt/group/face/zhubin/alg_code/fault_diagnosis_cnn_pytorch/model/vgg16_4.pth"
    test_txt_file = "/workspace/mnt/group/face/zhubin/alg_code/fault_diagnosis_cnn_pytorch/jiangnan_train_data_file/test.txt"
    predict(model_path, test_txt_file)



