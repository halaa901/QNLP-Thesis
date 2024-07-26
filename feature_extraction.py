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
    def __init__(self, image_urls, transform = None):
        self.image_urls = image_urls
        self.transform = transform

    def __len__(self):
        return len(self.image_urls)

    def __getitem__(self, idx):
        # print(f"Index: {idx}")
        url = self.image_urls[idx]
        try:
            response = requests.get(url)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert('RGB')
            # print(" Image loaded successfully.")
        except Exception as e:
            print(f" Error loading image at {url}: {e}")
            return None

        if self.transform:
            # print("Applying transform...")
            image = self.transform(image)
            # print(" Transformation applied.")

        return image

# Modify the model to output a 16-dimensional feature vector
class Custom16(nn.Module):
    def __init__(self):
        super(Custom16, self).__init__()
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])  # Keep layers except the last ones
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, 16)

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

    print("Retreiving the dataset")
    file_path = os.path.join(os.getcwd(), "clean_dataset.csv")
    df = pd.read_csv(file_path)
    image_pos_urls = df['image_1']
    image_neg_urls = df['image_2']

    print("Processing images pos and neg")
    dataset_pos = ImageDataset(image_urls=image_pos_urls, transform=transform)
    dataset_neg = ImageDataset(image_urls=image_neg_urls, transform=transform)

    print("\n(*) Complete download for positive and negative images. ")

    # ========================================================
    # INITIALIZE THE MODEL
    #  - call a custom ResnNet model
    #  - model removes the final layer and outputs a 16 length feature vector
    # ========================================================

    # Load the pre-trained ResNet model
    resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

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

    print("Extracting feature vector for positive images ..")
    features_pos = feature_vec(dataset_pos, model)
    features_pos = np.vstack(features_pos)

    print("Extracting feature vector for negative images ..")
    features_neg = feature_vec(dataset_neg, model)
    features_neg = np.vstack(features_neg)

    print("All features successfully stored:\n", features_pos)

    df_copy['image_1'] = features_pos.tolist() 
    df_copy['image_2'] = features_neg.tolist() 

    # Save the updated DataFrame to a new CSV
    df_copy.to_csv('features_dataset_1.csv', index=False)
    print("Dataframe store as dataset_vector.csv")