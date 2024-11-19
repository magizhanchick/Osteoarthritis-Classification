#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models, utils

from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,precision_score,recall_score,f1_score

from tqdm import tqdm


# In[2]:


def load_dataset_as_dataframe(subdir):
    data_dir = os.path.join('dataset', subdir)  # Join 'dataset' and subdir with the correct separator
    filepaths = []
    labels = []

    
    classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    classes.sort()  
    print('Classes:', classes)

    
    for class_name in classes:
        class_dir = os.path.join(data_dir, class_name)
        for file in os.listdir(class_dir):
            if file.endswith(('.jpg', '.jpeg', '.png', '.JPG')):
                filepaths.append(os.path.join(class_dir, file))
                labels.append(class_name)

   
    return pd.DataFrame({'Filepath': filepaths, 'Label': labels})


# In[3]:


classes = ['Normal', 'Osteoarthritis',]
train_df = load_dataset_as_dataframe('train')
test_df = load_dataset_as_dataframe('test')
val_df = load_dataset_as_dataframe('val')


# In[4]:


train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet standards
                         std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# In[5]:


random_index = np.random.randint(0, len(train_df), 16)
fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(10, 10),
                        subplot_kw={'xticks': [], 'yticks': []})

for i, ax in enumerate(axes.flat):
    ax.imshow(plt.imread(train_df.Filepath[random_index[i]]))
    ax.set_title(train_df.Label[random_index[i]])
plt.tight_layout()
plt.show()


# In[6]:


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
            return None, None  # Return None to indicate a problematic sample

        if self.transform:
            image = self.transform(image)

        label = classes.index(label)  # Convert label to index
        return image, label


# In[7]:


train_dataset = KneeDataset(train_df, transform=train_transform)
test_dataset = KneeDataset(test_df, transform=test_transform)
val_dataset = KneeDataset(val_df, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[8]:


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

        
        val_accuracy = evaluate_model(model, val_loader, phase='Validation')
        best_val_accuracy = max(best_val_accuracy, val_accuracy)

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = total_correct.double() / len(train_dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')
    
    return model


# In[20]:


def evaluate_model(model, loader, phase='Test'):
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
    print(f"{phase} Metrics: Accuracy={accuracy:.4f}")
    return accuracy, precision, recall, f1


# In[10]:


baseline_model = models.resnet18(pretrained=True)
baseline_model.fc = nn.Linear(baseline_model.fc.in_features, len(classes))
baseline_model = baseline_model.to(device)
optimizer = optim.Adam(baseline_model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
train_and_evaluate(baseline_model, optimizer, criterion, num_epochs=10)
print("Training Baseline Model...")


# In[11]:


print("\nEvaluating Baseline Model on Test Set...")
evaluate_model(baseline_model, test_loader)


# In[12]:


class EnhancedResNet(nn.Module):
    def __init__(self, num_classes):
        super(EnhancedResNet, self).__init__()
        self.model = models.resnet18(pretrained=True)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.model(x)


# In[13]:


print("\nTraining Enhanced Model...")
enhanced_model = EnhancedResNet(len(classes)).to(device)
optimizer = optim.Adam(enhanced_model.parameters(), lr=0.0005)
train_and_evaluate(enhanced_model, optimizer, criterion, num_epochs=10)


# In[14]:


print("\nEvaluating Enhanced Model on Test Set...")
evaluate_model(enhanced_model, test_loader)


# In[21]:


baseline_accuracy, baseline_precision, baseline_recall, baseline_f1 = evaluate_model(
    baseline_model, test_loader, phase="Baseline Test",
)

print("\nBaseline Model Scores:")
print("Accuracy:", baseline_accuracy)
print("Precision:", baseline_precision)
print("Recall:", baseline_recall)
print("F1-Score:", baseline_f1)


# In[23]:


enhanced_accuracy, enhanced_precision, enhanced_recall, enhanced_f1 = evaluate_model(
    enhanced_model, test_loader, phase="Enhanced Test",
)


print("\nEnhanced Model Scores:")
print(f"Accuracy: {enhanced_accuracy:.4f}")
print(f"Precision: {enhanced_precision:.4f}")
print(f"Recall: {enhanced_recall:.4f}")
print(f"F1-Score: {enhanced_f1:.4f}")


# In[34]:


import matplotlib.pyplot as plt
import numpy as np


baseline_scores = [0.8920, 0.8981, 0.8988, 0.8920]  # [Accuracy, Precision, Recall, F1-Score]
enhanced_scores = [0.9635, 0.9627, 0.9638, 0.9632]  # [Accuracy, Precision, Recall, F1-Score]
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']


x = np.arange(len(metrics)) 
width = 0.35  


plt.figure(figsize=(10, 6))
plt.bar(x - width/2, baseline_scores, width, label='Baseline Model', color='skyblue')
plt.bar(x + width/2, enhanced_scores, width, label='Enhanced Model', color='lightgreen')


plt.xlabel('Metrics', fontsize=12)
plt.ylabel('Scores', fontsize=12)
plt.title('Model Performance Comparison', fontsize=14)
plt.xticks(x, metrics, fontsize=10)
plt.ylim(0, 1.1)
plt.legend(fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show the plot
plt.tight_layout()
plt.show()


# In[36]:


baseline_model.eval()  


pred = []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = baseline_model(images)  
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

plt.show()
plt.tight_layout()

