import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Define the CNN model class for brand classification
class CNN(nn.Module):
    def __init__(self, K):  # K is the number of brand classes
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),
        )

        self.dense_layers = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(256 * 14 * 14, 1024),  # Adjusted based on the output size after convolutions
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, K),  # K is the number of brands
        )

    def forward(self, X):
        out = self.conv_layers(X)
        out = torch.flatten(out, start_dim=1)  # Flatten dynamically
        out = self.dense_layers(out)
        return out

# Function to verify the dataset structure
def verify_data_structure(data_path):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Training data path '{data_path}' does not exist.")

    class_folders = [f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))]
    if not class_folders:
        raise ValueError(f"No class folders found in '{data_path}'. Please organize your dataset.")

    print("Class folders found:", class_folders)

# Training the model
def train_model(train_data_path, num_epochs=10):
    # Verify the dataset structure
    verify_data_structure(train_data_path)

    # Set transformations for training images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Load dataset
    train_dataset = datasets.ImageFolder(root=train_data_path, transform=transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)

    # Initialize the CNN model
    model = CNN(K=len(train_dataset.classes))  # Number of brand classes
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()  # Zero the parameter gradients
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Calculate loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update parameters

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}")

    # Save the trained model
    save_path = r"C:\Users\RAMACHANDRAN.B\Desktop\brand_model.pt"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved as {save_path}")

# Call train_model to train and save the model
train_data_path = r"C:\Users\RAMACHANDRAN.B\OneDrive\test images"  # Specify the training data path
train_model(train_data_path=train_data_path)
