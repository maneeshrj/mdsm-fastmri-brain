import torch
import zipfile
from torchvision import transforms, datasets



class PreloadedDataset():
    def __init__(self, current_dataset,preload):

        # Load all images into memory
        self.images = []
            
        print("Preloading ",preload," training samples")
        for i in range(preload):
            self.images.append(current_dataset[i])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx]
    
    

def inf_train_gen(batch_size,data_path='default path',preload=50000):
    transf = transforms.Compose([
        transforms.CenterCrop(176),
        transforms.RandomHorizontalFlip
        (p=0.5),
        transforms.ToTensor()
        
    ])
    
    dataset = datasets.ImageFolder(
            data_path,
            transform=transf
        )    
    
    preloaded = PreloadedDataset(dataset,preload)
    loader = torch.utils.data.DataLoader(preloaded, batch_size, drop_last=True, shuffle=True,pin_memory=True
    )
    
    
    while True:
        for img, labels in loader:
            yield img
