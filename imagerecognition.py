import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import torch
import torch.nn as nn

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
            nn.Linear(256 * 14 * 14, 1024),  # Adjusted for output size after convolutions
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, K),  # K is the number of brands
        )

    def forward(self, X):
        out = self.conv_layers(X)
        out = out.view(-1, 256 * 14 * 14)  # Flatten
        out = self.dense_layers(out)
        return out

# Mapping index to brand names
idx_to_classes = {0: 'saffola', 1: 'gold winner'}  # Update according to your brand classes

# CNN prediction function
def prediction(image_path):
    # Load the pre-trained model
    model = CNN(len(idx_to_classes))  # Number of brand classes
    model.load_state_dict(torch.load(r"C:\Users\RAMACbrand_model.pt", weights_only=True))  # Load saved model
    model.eval()

    # Preprocess the image
    image = Image.open(image_path).convert('RGB')  # Ensure the image is RGB
    image = image.resize((224, 224))  # Resize to 224x224
    input_data = TF.to_tensor(image)
    input_data = input_data.unsqueeze(0)  # Add batch dimension

    # Perform prediction
    with torch.no_grad():
        output = model(input_data)
        index = np.argmax(output.numpy())  # Get the index of the predicted class

    return index

# Specify the image path for prediction
image_path = "C:\Users\Tamilarasan S\OneDrive\Desktop\flipkart Grid 6.0\ir test_image\goldwinner_cookingoil.jpeg" # Update with your image path

# Predict the brand
brand_class = prediction(image_path)

# Print the predicted brand name
brand_name = idx_to_classes[brand_class]  # Using idx_to_classes mapping
print(f"Predicted Brand: {brand_name}")
