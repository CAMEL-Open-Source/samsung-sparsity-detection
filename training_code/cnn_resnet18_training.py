import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
import os


class CachingDataset(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super(CachingDataset, self).__init__(root, transform)
        self.cache = {}

    def __getitem__(self, index):
        path, target = self.samples[index]
        if path in self.cache:
            sample = self.cache[path]
        else:
            sample = self.loader(path)
            if self.transform is not None:
                sample = self.transform(sample)
            self.cache[path] = sample
        return sample, target


# Define the ResNet18 model
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 1000)  # ImageNet has 1000 classes

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Hyperparameters
batch_size = 256
learning_rate = 0.001
num_epochs = 50

# Data loaders
transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Replace `path_to_imagenet_val` with the actual path to the extracted ILSVRC2012_img_val directory
train_dataset = CachingDataset(
    root="/mnt/670942ac-3399-4d79-ba6e-ddde092fb57d/dataset/ImageNet/imagenet_val",
    transform=transform,
)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Since we're using the validation set for training in this example, use the same loader for testing
test_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
training_idx = 0

# Training the model
for epoch in range(num_epochs):
    model.train()
    p_bar = tqdm(enumerate(train_loader), total=len(train_loader))
    for batch_idx, (data, targets) in enumerate(train_loader):
        # Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)

        # Forward
        scores = model(data)
        loss = criterion(scores, targets)

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Gradient descent or adam step
        optimizer.step()
        p_bar.update(1)
        training_idx += 1
        if training_idx % 100 == 0:
            # Save the model checkpoint
            checkpoint_path = os.path.join(
                "/root/workspace/samsungstf/training_results",
                f"resnet18_{training_idx}.pth",
            )
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "batch_idx": batch_idx,
                    "loss": loss.item(),
                    "training_idx": training_idx,
                },
                checkpoint_path,
            )

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")


# Check accuracy on training & test to see how good our model
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    model.train()
    return num_correct / num_samples


print(f"Accuracy on training set: {check_accuracy(train_loader, model)*100:.2f}")
print(f"Accuracy on test set: {check_accuracy(test_loader, model)*100:.2f}")
