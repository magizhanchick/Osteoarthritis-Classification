from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

# Define the classes
classes = ['Normal', 'Osteoarthritis']

# Function to load dataset
def load_dataset_as_dataframe(subdir):
    data_dir = Path('dataset') / subdir  # Use Path for directory handling
    filepaths = []
    labels = []

    classes = [d for d in data_dir.iterdir() if d.is_dir()]
    classes.sort()  
    print('Classes:', classes)

    for class_dir in classes:
        for file in class_dir.iterdir():
            if file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.JPG']:
                filepaths.append(file)
                labels.append(class_dir.name)

    return pd.DataFrame({'Filepath': filepaths, 'Label': labels})

# Load dataset
train_df = load_dataset_as_dataframe('train')
test_df = load_dataset_as_dataframe('test')
val_df = load_dataset_as_dataframe('val')

# Define transformations
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Dataset Class
class KneeDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.loc[idx, 'Filepath']
        label = self.df.loc[idx, 'Label']
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return None, None

        if self.transform:
            image = self.transform(image)

        label = classes.index(label)
        return image, label

# Load datasets
train_dataset = KneeDataset(train_df, transform=train_transform)
test_dataset = KneeDataset(test_df, transform=test_transform)
val_dataset = KneeDataset(val_df, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Streamlit UI for model training and evaluation
st.title("Knee X-ray Classification")

# Select model type
model_type = st.selectbox("Select Model", ["ResNet18", "Enhanced ResNet18"])

# Function to evaluate model
def evaluate_model(model, loader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    
    return accuracy, precision, recall, f1

# Training and evaluation logic
def train_and_evaluate(model, optimizer, criterion, num_epochs=5):
    best_val_accuracy = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            total_correct += torch.sum(preds == labels.data)
            total_samples += labels.size(0)
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{(total_correct/total_samples*100):.2f}%'})

        val_accuracy, _, _, _ = evaluate_model(model, val_loader)
        best_val_accuracy = max(best_val_accuracy, val_accuracy)

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = total_correct.double() / len(train_dataset)
        st.write(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')
    
    return model

# Model selection
if model_type == "ResNet18":
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
elif model_type == "Enhanced ResNet18":
    model = models.resnet18(pretrained=True)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, len(classes))
    )
else:
    raise ValueError("Invalid model type")

model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Train the model
if st.button('Train Model'):
    st.write("Training the model...")
    trained_model = train_and_evaluate(model, optimizer, criterion, num_epochs=5)
    st.write("Training complete!")

    # Evaluate on the test set
    accuracy, precision, recall, f1 = evaluate_model(trained_model, test_loader)
    st.write(f"Accuracy: {accuracy:.4f}")
    st.write(f"Precision: {precision:.4f}")
    st.write(f"Recall: {recall:.4f}")
    st.write(f"F1-Score: {f1:.4f}")

# Display images with predictions
if st.button('Show Predictions'):
    model.eval()
    pred = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            pred.extend(preds.cpu().numpy())

    random_index = np.random.randint(0, len(test_df) - 1, 15)
    fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(25, 15),
                            subplot_kw={'xticks': [], 'yticks': []})

    for i, ax in enumerate(axes.flat):
        ax.imshow(plt.imread(test_df.Filepath.iloc[random_index[i]]))
        if test_df.Label.iloc[random_index[i]] == classes[pred[random_index[i]]]:
            color = "green"
        else:
            color = "red"
        ax.set_title(f"True: {test_df.Label.iloc[random_index[i]]}\nPredicted: {classes[pred[random_index[i]]]}",
                     color=color)

    st.pyplot(fig)
