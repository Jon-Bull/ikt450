import numpy as np
import numpy
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

numpy.random.seed(7)

input_dim = 784  # 28x28
encoding_dim = 32  # latent space dimension

# Encoder Network (now outputs mean and log variance for latent space)
class EncNet(nn.Module):
    def __init__(self):
        super(EncNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc_mu = nn.Linear(256, encoding_dim)  # Mean of the latent space
        self.fc_logvar = nn.Linear(256, encoding_dim)  # Log variance of the latent space

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

# Decoder Network
class DecNet(nn.Module):
    def __init__(self):
        super(DecNet, self).__init__()
        self.fc1 = nn.Linear(encoding_dim, 256)
        self.fc2 = nn.Linear(256, input_dim)

    def forward(self, z):
        x = torch.relu(self.fc1(z))
        x = torch.sigmoid(self.fc2(x))  # Sigmoid to bound output between 0 and 1
        return x

# Reparameterization Trick: z = mu + sigma * epsilon
def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)  # Standard deviation
    eps = torch.randn_like(std)  # Sample from standard normal
    return mu + std * eps  # Reparameterized latent variable

# VAE Loss Function: Reconstruction Loss + KL Divergence
def loss_function(recon_x, x, mu, logvar):
    # Reconstruction loss (binary cross-entropy for MNIST images)
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    
    # KL Divergence loss: measure the difference between the learned latent distribution and a unit Gaussian
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + kl_loss


def sample_new_data(dec, encoding_dim, num_samples=10):
    """ Function to sample new data points and visualize them """
    # Sample random points from the standard normal distribution
    z = torch.randn(num_samples, encoding_dim)

    # Generate new images using the decoder
    with torch.no_grad():
        generated_images = dec(z)

    # Plot the generated images
    plt.figure(figsize=(20, 4))
    for i in range(num_samples):
        ax = plt.subplot(1, num_samples, i + 1)
        plt.imshow(generated_images[i].view(28, 28).numpy(), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

def train():
    transform = transforms.Compose([transforms.ToTensor()])
    
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
    
    # Initialize models and optimizer
    enc = EncNet()
    dec = DecNet()
    optimizer = optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=0.001)
    
    epochs = 2#10
    for epoch in range(epochs):
        enc.train()
        dec.train()
        train_loss = 0
        for batch_idx, (data, _) in enumerate(trainloader):
            data = data.view(-1, input_dim)  # Flatten the images
            
            # Forward pass through the encoder
            mu, logvar = enc(data)
            
            # Sample latent space using the reparameterization trick
            z = reparameterize(mu, logvar)
            
            # Decode the latent variable to reconstruct the image
            recon_data = dec(z)
            
            # Compute the loss
            loss = loss_function(recon_data, data, mu, logvar)
            train_loss += loss.item()
            
            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f'Epoch {epoch+1}, Loss: {train_loss / len(trainloader.dataset):.4f}')
    
    # Display original, encoded (latent space), and decoded images
    enc.eval()
    dec.eval()
    with torch.no_grad():
        dataiter = iter(trainloader)
        images, _ = dataiter.next()
        images = images.view(-1, input_dim)
        
        # Forward pass through the VAE
        mu, logvar = enc(images)
        z = reparameterize(mu, logvar)
        decoded_images = dec(z)
        
        # Visualize results
        n = 10  # how many digits we will display
        plt.figure(figsize=(30, 4))
        for i in range(n):
            # Display original
            ax = plt.subplot(3, n, i + 1)
            plt.imshow(images[i].view(28, 28).numpy(), cmap='gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # Display latent space
            ax = plt.subplot(3, n, i + 1 + n)
            plt.imshow(z[i].view(encoding_dim, 1).numpy(), cmap='gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # Display reconstruction
            ax = plt.subplot(3, n, i + 1 + n*2)
            plt.imshow(decoded_images[i].view(28, 28).numpy(), cmap='gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()
    sample_new_data(dec,encoding_dim)


def run():
    torch.multiprocessing.freeze_support()
    print('loop')

if __name__ == '__main__':
    run()
    train()

