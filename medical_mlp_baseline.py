import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import nibabel as nib

IMG_SIZE = 64
BATCH_SIZE = 2
LR = 0.001
EPOCHS = 20

class MedicalImageDataset(Dataset):
    def __init__(self, root_dir, mode='train'):
        self.data = []
        self.labels = []
        self.class_map = {'CN': 0, 'AD': 1}
        
        for category, label in self.class_map.items():
            dir_path = os.path.join(root_dir, category)
            if not os.path.exists(dir_path):
                continue
                
            files = sorted([f for f in os.listdir(dir_path) if f.endswith('.nii')])
            
            if mode == 'train':
                file_list = files[:6]
            else:
                file_list = files[6:9]
                
            for f_name in file_list:
                f_path = os.path.join(dir_path, f_name)
                try:
                    nii = nib.load(f_path)
                    vol = nii.get_fdata()
                    
                    mid = vol.shape[2] // 2
                    slice_2d = vol[:, :, mid]
                    
                    t = torch.FloatTensor(slice_2d).unsqueeze(0).unsqueeze(0)
                    t = F.interpolate(t, size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False)
                    t = (t - t.min()) / (t.max() - t.min())
                    
                    self.data.append(t.squeeze(0))
                    self.labels.append(label)
                    
                except Exception:
                    pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx].view(-1), self.labels[idx]

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(IMG_SIZE * IMG_SIZE, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.net(x)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_set = MedicalImageDataset('./ADNI', mode='train')
    test_set = MedicalImageDataset('./ADNI', mode='test')
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    model = SimpleMLP().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
        print(f"Epoch {epoch+1} | Loss: {epoch_loss/len(train_loader):.4f} | Acc: {100*correct/total:.1f}%")

    model.eval()
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            
            test_correct += (preds == labels).sum().item()
            test_total += labels.size(0)
            
            print(f"Pred: {preds.item()} | True: {labels.item()}")

    print(f"Final Test Acc: {100 * test_correct / test_total:.2f}%")