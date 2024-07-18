import torch
from torch.utils.data import DataLoader, random_split
import pickle
from custom_dataset import CustomDataset  # Import the CustomDataset class

def generate_custom_data():
    dataset = CustomDataset()

    # Create DataLoaders
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=50, shuffle=False)

    return train_loader, test_loader, dataset.observed_data.shape[2]

train_loader, test_loader, input_dim = generate_custom_data()

custom_dataset = {
    'train_dataloader': train_loader,
    'test_dataloader': test_loader,
    'input_dim': input_dim
}

# Save the dataset
with open('custom_dataset.pkl', 'wb') as f:
    pickle.dump(custom_dataset, f)
