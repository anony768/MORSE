import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
import os

class TransitionDataset(Dataset):
    def __init__(self, data, scaler_S):
        data = np.array(data)
        S = data[:, :125]
        S_prime = data[:, 126:251]
        unique_states = np.unique(np.vstack((S, S_prime)), axis=0)
        self.S = scaler_S.fit_transform(unique_states)
    
    def __len__(self):
        return len(self.S)
    
    def __getitem__(self, idx):
        return torch.tensor(self.S[idx], dtype=torch.float32), torch.tensor(self.S[idx], dtype=torch.float32)

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(125, 64),
            nn.ReLU()
        )
        self.mu_layer = nn.Linear(64, 10)
        self.logvar_layer = nn.Linear(64, 10)
        self.decoder = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 125)
        )
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        encoded = self.encoder(x)
        mu, logvar = self.mu_layer(encoded), self.logvar_layer(encoded)
        z = self.reparameterize(mu, logvar)
        decoded = self.decoder(z)
        return decoded, mu, logvar

def main():
    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    with open("data.pkl", "rb") as f:
        full_transition_buffer = pickle.load(f)
    
    scaler_S = StandardScaler()
    train_dataset = TransitionDataset(full_transition_buffer, scaler_S)
    dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)

    model = Autoencoder().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.0002)
    criterion = nn.MSELoss()
    
    num_epochs = 2000
    patience = 500
    losses = []
    best_loss = float("inf")
    trigger_times = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        
        for batch_inputs, batch_targets in dataloader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            
            optimizer.zero_grad()
            outputs, mu, logvar = model(batch_inputs)
            recon_loss = criterion(outputs, batch_targets)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_inputs.size(0)
            loss = recon_loss + 0.01 * kl_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += loss.item() * batch_inputs.size(0)
        
        epoch_loss = running_loss / len(train_dataset)
        losses.append(epoch_loss)
        
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            trigger_times = 0
            torch.save(model.encoder.state_dict(), "encoder_best.pt")
            torch.save(model.decoder.state_dict(), "decoder_best.pt")
            torch.save(model.state_dict(), "autoencoder_best.pt")
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break
        
        if epoch % 200 == 0:
            print(f"Epoch [{epoch}/{num_epochs}], Loss: {epoch_loss:.6f}")
    
    torch.save(model.encoder.state_dict(), "encoder_final.pt")
    torch.save(model.decoder.state_dict(), "decoder_final.pt")
    torch.save(model.state_dict(), "autoencoder_final.pt")
    
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("loss.png")
    plt.show()

if __name__ == "__main__":
    main()
