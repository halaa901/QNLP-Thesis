import torch
import torchvision.models as models
from torch import nn
from torchvision.models import ResNet50_Weights
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

from sklearn.model_selection import train_test_split

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

class CustomDataset(Dataset):
    def __init__(self, image_urls, labels, transform=None):
        self.image_urls = image_urls
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_urls)

    def __getitem__(self, idx):
        # Download image from URL
        try:
            response = requests.get(self.image_urls[idx])
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert('RGB')
        except Exception as error:
            print("Failed to download -> ", error)
        
        # Apply transformations if any
        if self.transform:
            image = self.transform(image)
        
        # Get label
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label


# ===========================================================
# ===========================================================
print("Prepare Dataset")

file_path = os.path.join(os.getcwd(), "database.csv")
df = pd.read_csv(file_path)

# Extract the sentence
sentence = df['sentence']
image_pos = df['pos_url']
image_neg = df['neg_url']

df = pd.DataFrame({
    'sentence': sentence,
    'image_pos': image_pos,
    'image_neg': image_neg})

df['label_image1'] = 0
df['label_image2'] = 1

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224
    transforms.ToTensor(),          # Convert images to tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet stats
])

# Example paths and labels (in practice, gather from your dataset)
image_urls_train = df["image_pos"]
labels_train = df["label_image1"]

image_urls_train, image_urls_val = train_test_split(df["image_pos"], test_size=0.2, random_state=42)


# Instantiate datasets
train_dataset = CustomDataset(image_urls=image_urls_train, labels=labels_train, transform=transform)
val_dataset = CustomDataset(image_urls=image_urls_val, labels=labels_val, transform=transform)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

# ===========================================================
# ===========================================================
print("Load Resnet Model")

# Load the pre-trained ResNet model
resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

# Instantiate the custom model
model = Custom16()

# Add the classification layer for binary output
classifier = nn.Sequential(nn.Linear(16, 1), nn.Sigmoid())

# Combine the feature extractor and the classifier
full_model = nn.Sequential(model, classifier)

# Set model to training mode
full_model.train()

# ===========================================================
# ===========================================================
print("Define Loss function")

# Define loss function and optimizer
criterion = nn.BCELoss()  # Or BCEWithLogitsLoss without sigmoid in the classifier
optimizer = torch.optim.Adam(full_model.parameters(), lr=0.001)

# If using GPU
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# full_model.to(device)

# ===========================================================
# ===========================================================
print("Training Loop")

# Assuming 'train_dataset' and 'val_dataset' are instances of CustomDataset
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

num_epochs = 10
for epoch in range(num_epochs):
    full_model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = full_model(inputs)
        loss = criterion(outputs, labels.unsqueeze(1))
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}')

# ===========================================================
# ===========================================================
print("Validation")

# Validation
full_model.eval()
val_loss = 0.0
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = full_model(inputs)
        loss = criterion(outputs, labels.unsqueeze(1))
        val_loss += loss.item()

print(f'Validation Loss: {val_loss/len(val_loader)}')


