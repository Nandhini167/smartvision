
import cv2
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import pandas as pd
import torch.nn as nn

# Define the CNN model class
class CNN(nn.Module):
    def __init__(self, K):
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
            nn.Linear(50176, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, K),
        )

    def forward(self, X):
        out = self.conv_layers(X)
        out = out.view(-1, 50176)  # Flatten
        out = self.dense_layers(out)
        return out

# Mapping index to disease names
idx_to_classes = {0: 'pomegranate_rotten', 1: 'pomegranate_fresh', 2: 'banana_fresh',
                  3: 'banana_rotten', 4: 'apple_fresh', 5: 'apple_rotten',
                  6: 'cucumber_fresh', 7: 'orange_fresh', 8: 'orange_rotten',
                  9: 'guava_fresh', 10: 'guava_rotten', 11: 'watermelon_fresh',
                  12: 'watermelon_rotten', 13: 'papaya_fresh', 14: 'papaya_rotten',
                  15: 'sapota_fresh', 16: 'cucumber_rotten', 17: 'pineapple_fresh',
                  18: 'pineapple_rotten', 19: 'kiwi_fresh', 20: 'kiwi_rotten',
                  21: 'muskmelon_fresh', 22: 'muskmelon_rotten', 23: 'avacado_rotten', 24: 'drumstick_fresh',
                  25: 'dragonfruit_fresh', 26: 'dragonfruit_rotten', 27: 'pear_fresh',
                  28: 'tomato_fresh', 29: 'tomato_rotten', 30: 'onion_fresh',
                  31: 'onion_rotten', 32: 'potato_fresh', 33: 'potato_rotten',
                  34: 'brocolli_fresh', 35: 'carrot_fresh',
                  36: 'carrot_rotten', 37: 'ladiesfinger_fresh', 38: 'ladiesfinger_rotten'}

# CNN prediction function
def prediction(image):
    # Load the pre-trained model
    model = CNN(39)  # 39 classes
    model.load_state_dict(torch.load(r"C:\Users\Tamilarasan S\OneDrive\Desktop\New folder\fruittt_freshness_model.pt\fruit_freshness_model.pt", weights_only=True))
    model.eval()

    # Preprocess the image
    image = image.resize((224, 224))
    input_data = TF.to_tensor(image)
    input_data = input_data.view((-1, 3, 224, 224))

    # Perform prediction
    with torch.no_grad():
        output = model(input_data)
        output = output.numpy()
        index = np.argmax(output)  # Get the index of the predicted class

    return index

# Webcam capture and prediction
def capture_and_predict():
    cap = cv2.VideoCapture(0)  # Start video capture from webcam

    while True:
        ret, frame = cap.read()  # Capture frame-by-frame
        if not ret:
            break

        cv2.imshow('Press "q" to capture image', frame)  # Show the video feed

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Capture image on pressing 'q'
            # Convert the frame (BGR from OpenCV) to RGB and save as PIL image
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Perform prediction
            freshness_class = prediction(image)

            # Print the predicted freshness class
            freshness_name = idx_to_classes[freshness_class]  # Using idx_to_classes mapping
            print(f"Predicted freshness: {freshness_name}")
            break

    cap.release()
    cv2.destroyAllWindows()

# Start webcam capture and prediction
capture_and_predict()
