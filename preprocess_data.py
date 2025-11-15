import os
import numpy as np
import torch
from torchvision import transforms


def preprocess_raw_data():
    transform = transforms.RandomAffine(degrees=15, translate=(0.12, 0.12), scale=(0.75, 0.8), interpolation=transforms.InterpolationMode.BILINEAR)
    print("Processing...")
    for i, fn in enumerate(os.listdir('raw_training_data')):
        if (fn !=".DS/store"):
            data = np.load(f"raw_training_data/{fn}")
            data = np.reshape(data, [len(data), 28, 28]) # Reshape into 2D array to be able to random rotate
            
            # Convert to tensor
            data_tensor = torch.tensor(data)
            data_tensor = data_tensor.unsqueeze(1) # Turns the shape [N, 28, 28] -> [N, 1, 28, 28] torch model needs to format to be this way
            
            # Transform each image
            data_tensor = torch.stack([transform(img) for img in data_tensor])
            
            torch.save(data_tensor, f"processed_training_data/{fn[:-4]}.pt")
            
            print(f"Finished preprocess for {fn}")
