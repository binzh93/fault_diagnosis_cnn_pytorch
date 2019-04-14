import torch
from torch.autograd import Variable
from torchvision import transforms
import numpy as np
from PIL import Image
import os
import cv2


visual_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        # transforms.RandomResizedCrop(input_size),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


class FeatureVisualization():
    def __init__(self, img_path, layer_num, trainsform, model):
        self.img_path   = img_path
        self.layer_num  = layer_num
        self.model      = model.features
        self.trainsform = trainsform

    def process_image(self):
        image = Image.open(self.img_path)
        image = self.trainsform(image)
        image_tensor = image.unsqueeze_(0)
        image_var = Variable(image_tensor)
        return image_var
    
    def get_feature(self):
        input = self.process_image()
        input = input.to(device)
        x = input
        for index, layer in enumerate(self.model):
            x = layer(x)
            if index == self.layer_num:
                return x

    def save_feature(self, feature_img_save_dir):
        features = self.get_feature()
        for i in range(features.shape[1]):
            feature = features[:, i, :, :]
#             print(feature.shape)
            feature = feature.view(features.shape[2], features.shape[3])
            feature = feature.data.numpy()
            # scale the feature to [0, 1]
            feature = 1.0 / (1 + np.exp(-1.0 * feature))
            feature = np.round(feature*255)
            if not os.path.exists(feature_img_save_dir):
                os.mkdir(feature_img_save_dir)
            if not os.path.exists(os.path.join(feature_img_save_dir, str(self.layer_num))):
                os.mkdir(os.path.join(feature_img_save_dir, str(self.layer_num)))
            fea_img_save_path = os.path.join(feature_img_save_dir, str(self.layer_num), str(i+1)+".jpg")
            cv2.imwrite(fea_img_save_path, feature)
#             feature.save(fea_img_save_path)
            

if __name__ == "__main__":
    img_path = "/workspace/mnt/group/face/zhubin/alg_code/fault_diagnosis_cnn/jiangnan_data/jiangnan_data_2500/ib_2500/ib_108.jpg"
    model_path = "/workspace/mnt/group/face/zhubin/alg_code/fault_diagnosis_cnn_pytorch/model/vgg16_14.pth"
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    model = torch.load(model_path)
    model = model.to(device)
#     print(model)
    model.eval()
    feature_img_save_dir = "feature_img"
    for i in range(30):
        layer_num = i
        fea_visual = FeatureVisualization(img_path, layer_num, visual_transforms, model)
        fea_visual.save_feature(feature_img_save_dir)
    


