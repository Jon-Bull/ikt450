import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import Subset

# Hyperparameters
BATCH_SIZE = 64
LATENT_DIM = 100  # Size of the random noise vector (input to generator)
EPOCHS = 50
LEARNING_RATE = 0.0002
SUBSET_SIZE = 2000  # Only use 200 images from MNIST

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load MNIST dataset and select a subset of 200 images
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
full_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)

# Extract a subset of 200 images
subset_indices = list(range(SUBSET_SIZE))
subset_dataset = Subset(full_dataset, subset_indices)
train_loader = torch.utils.data.DataLoader(dataset=subset_dataset, batch_size=BATCH_SIZE, shuffle=True)


# Define the Generator model
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(LATENT_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 28 * 28),  # MNIST images are 28x28
            nn.Tanh()  # Output values between -1 and 1
        )

    def forward(self, z):
        img = self.model(z)
        # Correctly reshape the output into a 4D tensor (batch_size, 1, 28, 28)
        img = img.view(z.size(0), 1, 28, 28)  # Here, z.size(0) gives the batch size
        return img



# Define the Discriminator model
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Output a probability between 0 and 1
        )

    def forward(self, img):
        # Ensure img is a tensor in the shape (batch_size, 1, 28, 28)
        if isinstance(img, tuple):
            img = img[0]  # If DataLoader returns a tuple, extract the first element (the image tensor)

        # Flatten the image into a vector of size 28*28 for each image in the batch
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity



# Initialize generator and discriminator
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Loss function and optimizers
adversarial_loss = nn.BCELoss()  # Binary Cross Entropy loss
optimizer_G = optim.Adam(generator.parameters(), lr=LEARNING_RATE)
optimizer_D = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE)

# Function to create random noise vectors
def generate_noise(batch_size, latent_dim):
    return torch.randn(batch_size, latent_dim).to(device)

# Training loop
for epoch in range(EPOCHS):
    for batch_idx, (real_images, _) in enumerate(train_loader):
        real_images = real_images.to(device)

        # ---------------------
        #  Train Discriminator
        # ---------------------

        # Generate random noise and fake images
        batch_size = real_images.size(0)
        noise = generate_noise(batch_size, LATENT_DIM)
        fake_images = generator(noise)

        # Real images label: 1, Fake images label: 0
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # Compute discriminator loss on real images
        real_loss = adversarial_loss(discriminator(real_images), real_labels)

        # Compute discriminator loss on fake images
        fake_loss = adversarial_loss(discriminator(fake_images.detach()), fake_labels)

        # Total discriminator loss
        d_loss = (real_loss + fake_loss) / 2

        # Backprop and optimize the discriminator
        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()

        # -----------------
        #  Train Generator
        # -----------------

        # Generate new fake images and compute loss
        fake_images = generator(noise)
        g_loss = adversarial_loss(discriminator(fake_images), real_labels)  # Generator wants the discriminator to label fake as real (1)

        # Backprop and optimize the generator
        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()

        # Print losses occasionally
        if batch_idx % 10 == 0:
            print(f'Epoch [{epoch+1}/{EPOCHS}], Step [{batch_idx}/{len(train_loader)}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}')

# Function to generate images after training
def sample_images(generator, num_images=10):
    noise = generate_noise(num_images, LATENT_DIM)
    fake_images = generator(noise).detach().cpu()
    return fake_images

# Generate and plot images
generated_images = sample_images(generator)

# Plot generated images
def plot_images(images, num_images=10):
    images = images[:num_images].numpy()
    fig, axes = plt.subplots(1, num_images, figsize=(num_images, 1))
    for i in range(num_images):
        axes[i].imshow(images[i][0], cmap='gray')
        axes[i].axis('off')
    plt.show()

plot_images(generated_images)

