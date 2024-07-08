import torch
import numpy as np
import os
import argparse

def load_mtan_output(file_path):
    checkpoint = torch.load(file_path)
    rec_state_dict = checkpoint['rec_state_dict']
    dec_state_dict = checkpoint['dec_state_dict']
    
    return rec_state_dict, dec_state_dict

def prepare_data_for_primenet(T, X, M):
    # Ensure T, X, and M have the correct shapes
    T_expanded = T.unsqueeze(-1).expand_as(X)

    # Combine T, X, M into a single tensor
    combined_data = torch.stack((T_expanded, X, M), dim=2)

    # Split data into train, validation, and test sets
    train_data = combined_data[:60]
    val_data = combined_data[60:80]
    test_data = combined_data[80:]

    # Generate random labels for fine-tuning (assuming binary classification)
    labels = np.random.randint(0, 2, size=(combined_data.shape[0],))  # Binary labels for classification, example
    labels_train, labels_val, labels_test = labels[:60], labels[60:80], labels[80:]

    # Create directories if they don't exist
    os.makedirs('PrimeNet/data/pretrain', exist_ok=True)
    os.makedirs('PrimeNet/data/finetune', exist_ok=True)

    # Save the data in the required format for pre-training
    torch.save(train_data, 'PrimeNet/data/pretrain/X_train.pt')
    torch.save(val_data, 'PrimeNet/data/pretrain/X_val.pt')

    # Save the fine-tuning data
    torch.save(train_data, 'PrimeNet/data/finetune/X_train.pt')
    torch.save(val_data, 'PrimeNet/data/finetune/X_val.pt')
    torch.save(test_data, 'PrimeNet/data/finetune/X_test.pt')

    torch.save(torch.tensor(labels_train, dtype=torch.long), 'PrimeNet/data/finetune/y_train.pt')
    torch.save(torch.tensor(labels_val, dtype=torch.long), 'PrimeNet/data/finetune/y_val.pt')
    torch.save(torch.tensor(labels_test, dtype=torch.long), 'PrimeNet/data/finetune/y_test.pt')

    print("Data generation and saving completed.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input .h5 file')
    args = parser.parse_args()

    file_path = args.input_file
    rec_state_dict, dec_state_dict = load_mtan_output(file_path)

    # Example tensors (replace these with actual data extracted from the state dicts or another source)
    # Assuming T, X, M are part of the state dict or derived from the checkpoint
    T = torch.linspace(0, 1, 100)  # Replace with actual time points
    X = torch.randn(100, 5)  # Replace with actual data
    M = torch.randint(0, 2, (100, 5))  # Replace with actual mask

    prepare_data_for_primenet(T, X, M)
