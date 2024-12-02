import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms

N_EPOCHS = 15

# Define transformations (resize to a fixed size and convert to tensor)
dt_transform = transforms.Compose(
    [
        transforms.Resize((14, 14)),  # Resize all images to 14x14 (or any other size)
        transforms.ToTensor(),  # Convert PIL image to PyTorch tensor
    ]
)


# Define a simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # First convolutional layer: input channels = 1 (grayscale), output channels = 16
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1
        )
        # Second convolutional layer: output channels = 32
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1
        )
        # Fully connected layers
        self.fc1 = nn.Linear(32 * 3 * 3, 64)  # 32 feature maps of 7x7 after pooling
        self.fc2 = nn.Linear(64, 2)  # Output layer for 2 classes

    def forward(self, x):
        # Convolutional layer 1 + ReLU + MaxPooling
        # print(f"1 {x.shape}")
        x = F.relu(self.conv1(x))
        # print(f"1 {x.shape}")
        x = F.max_pool2d(x, 2)
        # print(f"1 {x.shape}")
        # Convolutional layer 2 + ReLU + MaxPooling
        x = F.relu(self.conv2(x))
        # print(f"1 {x.shape}")
        # print(f"1 {x.shape}")
        x = F.max_pool2d(x, 2)
        # print(f"1 {x.shape}")
        # Flatten the output to feed it into fully connected layers
        x = x.view(x.size(0), -1)
        # print(f"1 {x.shape}")
        # Fully connected layer 2 + ReLU
        x = F.relu(self.fc1(x))
        # print(f"1 {x.shape}")
        # Output layer (2 classes)
        x = self.fc2(x)
        # print(f"1 {x.shape}")
        return x


class PredImageDataset(Dataset):
    def __init__(self, image_list):
        self.image_list = image_list
        self.transform = dt_transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        # Get the image at index idx
        image = self.image_list[idx]

        # Convert the 2D NumPy array to a 3D tensor (1, height, width)
        # image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        image = Image.fromarray(image, mode="L")

        # Apply transformations if any (e.g., normalization)
        if self.transform:
            image = self.transform(image)

        return image


# Custom dataset class
class CustomImageDataset(Dataset):
    def __init__(self, img_dir=None, img_lst=None, transform=None):
        self.transform = transform
        self.data_as_files = False

        if img_dir is not None:
            self.img_dir = img_dir
            self.img_files = [
                f for f in os.listdir(img_dir) if f.endswith(".png")
            ]  # Only PNG images
            self.data_as_files = True
            self.len = len(self.img_files)
        elif img_lst is not None:
            # self.img_lst = img_lst
            self.img_lst = [Image.fromarray(im, mode="L") for im in img_lst]
            # self.img_lst = [torch.tensor(im, dtype=torch.float32).unsqueeze(0) for im in img_lst]
            self.len = len(img_lst)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        label = -1

        if self.data_as_files:
            img_name = self.img_files[idx]
            img_path = os.path.join(self.img_dir, img_name)
            # Load image as grayscale
            image = Image.open(img_path).convert("L")  # "L" mode is for grayscale

            # Assign label based on first character of filename
            label = 1 if img_name[0].lower() == "y" else 0
        else:
            image = self.img_lst[idx]
            label = 0

        # Apply transformations if any
        if self.transform:
            image = self.transform(image)

        return image, label


# Training loop
def train(model, train_loader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Accumulate the loss
            running_loss += loss.item()

        print(
            f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}"
        )


# Evaluation function
def evaluate(model, test_loader):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():  # Disable gradient calculation
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")


def build_model():
    # Path to the directory containing images
    img_dir = "img/detect/norm/"
    valid_dir = "img/detect/eval/"

    # Create the dataset
    dataset = CustomImageDataset(img_dir=img_dir, transform=dt_transform)
    valid_dataset = CustomImageDataset(img_dir=valid_dir, transform=dt_transform)

    # Split the dataset into train and test (e.g., 80% train, 20% test)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create DataLoaders for train and test sets
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

    # Initialize the model, loss function, and optimizer
    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model for 10 epochs
    train(model, train_loader, criterion, optimizer, num_epochs=N_EPOCHS)

    # Evaluate the model
    evaluate(model, test_loader)
    evaluate(model, valid_loader)

    # torch.save(model.state_dict(), "models/model_1")
    torch.save(model.state_dict(), "models/model_2")
    return model


def load_model(model_path):
    # Load the model
    # model_path = "models/model_1"
    model = SimpleCNN()
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode

    return model


def image_predict(model, arr):
    # dataset = PredImageDataset(arr)
    dataset = CustomImageDataset(img_lst=arr, transform=dt_transform)
    # dataset = CustomImageDataset(img_dir="/tmp/img/", transform=dt_transform)
    data_loader = DataLoader(dataset, batch_size=len(arr), shuffle=False)

    images, _ = next(iter(data_loader))
    # print(f"len:{len(images)} images")
    with torch.no_grad():  # No need to track gradients for inference
        outputs = model(images)

        # If the model outputs probabilities, you can use argmax to get the predicted class
        _, predicted = torch.max(outputs, 1)
        probabilities = torch.argmax(outputs, 1)

        # print(f"Predicted class: {predicted}")

        # probs = probabilities.tolist()
        pred = predicted.tolist()
        outs = outputs[:, 1].tolist()
        # print(f"outputs:{}")
        # print(f"pred:{pred}")
        # print(f"probs:{probs}")

        return pred, outs


if __name__ == "__main__":
    build_model()

    # valid_dir = "img/detect/eval/"
    # valid_dataset = CustomImageDataset(img_dir=valid_dir, transform=dt_transform)
    # valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
    # model = load_model("models/model_1")
    # evaluate(model, valid_loader)
