import torch
import torchvision.models as models
from torch import nn
from torchvision.models import ResNet50_Weights
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torchvision import transforms

import numpy as np

import pandas as pd
from PIL import Image

import os
import sys
import io
from io import BytesIO
import requests

class ImageDataset(Dataset):
    def __init__(self, image_filenames, transform=None, image_folder=None):
        self.image_filenames = image_filenames
        self.transform = transform
        self.image_folder = image_folder

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_folder, self.image_filenames[idx])
        image = Image.open(img_name).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image

# Modify the model to output a 16-dimensional feature vector
class Custom16(nn.Module):
    def __init__(self):
        # super(Custom16, self).__init__()
        # self.resnet = nn.Sequential(*list(resnet.children())[:-2])  # Keep layers except the last ones
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(2048, 16)

        super(Custom16, self).__init__()
        resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])  # Keep layers except the last ones
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(resnet.fc.in_features, 16)
        self.fc = nn.Linear(resnet.fc.in_features, 40)

    def forward(self, x):
        x = self.resnet(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def feature_vec(dataset, model):
    feature_list = []
    with torch.no_grad():  # No need to track gradients during inference
        for index in range(len(dataset)):
            image = dataset[index]  # Get image from dataset
            if image is not None:
                image = image.unsqueeze(0)  # Add batch dimension (1, C, H, W)
                image = image.to(device)  # Move image to the same device as the model
                features = model(image)  # Extract features
                # print(features)
                feature_list.append(features.cpu().numpy())  # Move features to CPU and convert to numpy array
    return feature_list

if __name__ == "__main__":

    print("(*) Imports downloded.\n")

    # ========================================================
    # PREPARE THE DATASET 
    # - download the image urls (they can all be downloaded)
    # - transform the images to desirable format
    # ========================================================

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Specify the folder and file name
    folder_name = "Dataset_2"
    file_name = "custom_dataset.csv"

    # Construct the file path
    file_path = os.path.join(os.getcwd(), folder_name, file_name)

    print("Retreiving the dataset")
    # file_path = os.path.join(os.getcwd(), "clean_dataset.csv")
    df = pd.read_csv(file_path)
    df['image'] = df['image'].str[1:]
    images = df['image']

    # current_directory = os.getcwd()
    # print("Current Directory:", current_directory)  

    print("Processing images")
    image_folder = "Dataset_2/images" 
    dataset = ImageDataset(image_filenames=images, transform=transform, image_folder=image_folder)

    print("\n(*) Complete download for all images. ")

    # ========================================================
    # INITIALIZE THE MODEL
    #  - call a custom ResnNet model
    #  - model removes the final layer and outputs a 16 length feature vector
    # ========================================================

    # Load the pre-trained ResNet model -> defined in custom16()
    # resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

    # Instantiate the custom model
    model = Custom16()
    model.eval() 

    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    print("\n(*) Loaded the pre-trained ResNet custom model. \n")

    # ========================================================
    # EXTRACT FEATURES
    #  - loop thorugh each entry and store the 16 feature vector into a new csv file. 
    # ========================================================

    # Copy the dataframe to overwrite the feature vectors
    df_copy = df.copy()

    print("Extracting feature vector for images ..")
    features_ = feature_vec(dataset, model)
    features = np.vstack(features_)

    print("All features successfully stored:\n", features)

    df_copy['image'] = features.tolist() 

    # Specify the folder path and file name
    # folder_path = os.path.join(os.getcwd(), "Features_Dataset_2")
    file_name = "custom_features_dataset_2_(40).csv"
    file_path = os.path.join(os.getcwd(), file_name)

    # Save the DataFrame to the specified file path
    df_copy.to_csv(file_path, index=False)

    print("* Saving New Databse * ")
    print("=======================")
    print(f"File saved to: {file_path} as {file_name}")