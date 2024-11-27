import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import Subset

# Hyperparameters
BATCH_SIZE = 64
TIMESTEPS = 100  # Number of diffusion steps
EPOCHS = 10
LEARNING_RATE = 0.001
SUBSET_SIZE = 200  # Only use 200 images from MNIST

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load MNIST dataset and select a subset of 200 images
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
full_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)

# Extract a subset of 200 images
subset_indices = list(range(SUBSET_SIZE))
subset_dataset = Subset(full_dataset, subset_indices)
train_loader = torch.utils.data.DataLoader(dataset=subset_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Simple convolutional denoising model
class DenoiseModel(nn.Module):
    def __init__(self):
        super(DenoiseModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, padding=1),
            nn.Tanh()  # Output values between -1 and 1
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Add noise to the images (forward diffusion process)
def add_noise(images, t, timesteps):
    noise = torch.randn_like(images).to(device)  # Gaussian noise
    noisy_images = images * (1 - t / timesteps) + noise * (t / timesteps)
    return noisy_images

# Mean Squared Error loss function
def compute_loss(predicted_noise, true_noise):
    return torch.mean((predicted_noise - true_noise) ** 2)

# Initialize model, optimizer, and loss function
model = DenoiseModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
for epoch in range(EPOCHS):
    for batch_idx, (real_images, _) in enumerate(train_loader):
        real_images = real_images.to(device)  # Move images to GPU/CPU
        t = torch.randint(1, TIMESTEPS, (1,)).item()  # Random time step
        
        # Add noise to the real images
        noisy_images = add_noise(real_images, t, TIMESTEPS)

        # Predict the noise added by the model
        predicted_noise = model(noisy_images)

        # Compute the true noise
        true_noise = (noisy_images - real_images) / (t / TIMESTEPS)

        # Compute the loss
        loss = compute_loss(predicted_noise, true_noise)

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print(f'Epoch [{epoch+1}/{EPOCHS}], Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}')

# Reverse diffusion process: generating new images
def sample_images(model, timesteps):
    with torch.no_grad():
        noisy_images = torch.randn((BATCH_SIZE, 1, 28, 28)).to(device)  # Start with random noise
        for t in reversed(range(1, timesteps)):
            predicted_noise = model(noisy_images)
            noisy_images = noisy_images - predicted_noise * (t / timesteps)  # Gradually denoise
        return noisy_images

# Generate some images from noise
generated_images = sample_images(model, TIMESTEPS)

# Plot generated images
def plot_images(images, num_images=10):
    images = images[:num_images].cpu().numpy()
    fig, axes = plt.subplots(1, num_images, figsize=(num_images, 1))
    for i in range(num_images):
        axes[i].imshow(images[i][0], cmap='gray')
        axes[i].axis('off')
    plt.show()

plot_images(generated_images)

