import torch
from torch.utils.data import Dataset

class HyperspectralDataset(Dataset):
    def __init__(self, data_cube):
        self.data = torch.tensor(data_cube, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
        import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

# Step 2: 3D-CNN + ViT Hybrid (Simple version for INFOTAC)
class Simple3DCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(8, 16, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(16, 2) # 2 = Healthy vs Stressed

    def forward(self, x):
        # x shape: (batch, bands, h, w) -> we make it 5D for 3D conv
        x = x.unsqueeze(1) # (batch, 1, bands, h, w)
        x = self.conv(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# Step 3: FAKE TRAINING LOOP
if __name__ == "__main__":
    print("Creating fake data cube 200x100x100...")
    fake_cube = np.random.rand(100, 200, 5, 5).astype(np.float32) # 100 samples
    
    dataset = HyperspectralDataset(fake_cube)
    loader = DataLoader(dataset, batch_size=10, shuffle=True)
    
    model = Simple3DCNN()
    print(model)
    print(f"Dataset length: {len(dataset)}")
    
    # Test one batch
    for batch in loader:
        print(f"Batch shape: {batch.shape}") # should be [10, 200, 5, 5]
        output = model(batch)
        print(f"Output shape: {output.shape}") # should be [10, 2]
        print("Week 2 Task SUCCESS - Model is working!")
        break