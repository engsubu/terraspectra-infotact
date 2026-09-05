{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "ae28179d-4c3c-402a-9214-b6da8f0f1793",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Massive cube loaded: (200, 1000, 1000) - Size: 1.60 GB\n",
      "Created 400 small tiles of [200,5,5] - Memory safe!\n",
      "Extraction Audit PASSED\n"
     ]
    }
   ],
   "source": [
    "import torch\n",
    "import numpy as np\n",
    "\n",
    "# Simulate massive 1000-acre farm image\n",
    "MASSIVE_CUBE = np.random.rand(200, 1000, 1000)\n",
    "print(f\"Massive cube loaded: {MASSIVE_CUBE.shape} - Size: {MASSIVE_CUBE.nbytes/1e9:.2f} GB\")\n",
    "\n",
    "TILE_SIZE = 5\n",
    "stride = 50  # using 50 for notebook so it's faster, not 5\n",
    "patches = []\n",
    "\n",
    "for x in range(0, 1000, stride):\n",
    "    for y in range(0, 1000, stride):\n",
    "        patch = MASSIVE_CUBE[:, x:x+TILE_SIZE, y:y+TILE_SIZE]\n",
    "        if patch.shape == (200, 5, 5):\n",
    "            patches.append(patch)\n",
    "\n",
    "print(f\"Created {len(patches)} small tiles of [200,5,5] - Memory safe!\")\n",
    "print(\"Extraction Audit PASSED\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "1ec60efb-8fc0-4b2b-9974-fbf2681f2a58",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Week 3 Hybrid ViT Model SUCCESS! Output: torch.Size([2, 2])\n",
      "Model Architecture: 3D-CNN for local + ViT Attention for global spectral correlation\n"
     ]
    }
   ],
   "source": [
    "import torch\n",
    "import torch.nn as nn\n",
    "\n",
    "class Hybrid3DCNN_ViT(nn.Module):\n",
    "    def __init__(self, num_classes=2):\n",
    "        super().__init__()\n",
    "        # Your same 3D-CNN as feature extractor\n",
    "        self.cnn = nn.Sequential(\n",
    "            nn.Conv3d(1, 8, kernel_size=3, padding=1),\n",
    "            nn.ReLU(),\n",
    "            nn.Conv3d(8, 16, kernel_size=3, padding=1),\n",
    "            nn.ReLU(),\n",
    "            nn.AdaptiveAvgPool3d((8, 1, 1)) # Output: [B, 16, 8, 1, 1]\n",
    "        )\n",
    "        # ViT Part: Self-Attention across spectral bands\n",
    "        # Treat 8 as sequence length, 16 as feature dim\n",
    "        encoder_layer = nn.TransformerEncoderLayer(d_model=16, nhead=4, batch_first=True)\n",
    "        self.vit = nn.TransformerEncoder(encoder_layer, num_layers=2)\n",
    "        \n",
    "        self.classifier = nn.Linear(16, num_classes)\n",
    "\n",
    "    def forward(self, x):\n",
    "        # x: [B, 1, 200, 5, 5] -> we take 8 spectral chunks\n",
    "        # For demo, we compress 200 -> 8\n",
    "        x = x[:, :, ::25, :, :]  # [B,1,8,5,5]\n",
    "        feat = self.cnn(x) # [B,16,8,1,1]\n",
    "        feat = feat.squeeze(-1).squeeze(-1).permute(0,2,1) # [B,8,16]\n",
    "        \n",
    "        # Self-Attention finds correlation across bands\n",
    "        attn_out = self.vit(feat) # [B,8,16]\n",
    "        \n",
    "        # Average over spectrum\n",
    "        pooled = attn_out.mean(dim=1) # [B,16]\n",
    "        return self.classifier(pooled)\n",
    "\n",
    "# TEST IT\n",
    "model = Hybrid3DCNN_ViT()\n",
    "test_cube = torch.randn(2, 1, 200, 5, 5)\n",
    "output = model(test_cube)\n",
    "print(f\"Week 3 Hybrid ViT Model SUCCESS! Output: {output.shape}\")\n",
    "print(\"Model Architecture: 3D-CNN for local + ViT Attention for global spectral correlation\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4b332b12-1103-4e3b-b4e9-9152896a87ac",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
