import torch
import numpy as np
import os
import argparse

def prepare_data_for_primenet(T, X, M):
    # Ensure T_expanded has the same shape as X and M
    T_expanded = np.expand_dims(T, axis=2)
    print(T_expanded.shape)
    print(X.shape)
    print(M.shape)
    assert T_expanded.shape == X.shape == M.shape, f"Shapes do not match: T_expanded: {T_expanded.shape}, X: {X.shape}, M: {M.shape}"
    
    combined_data = np.concatenate((T_expanded, X, M), axis=2)
    
    num_samples = combined_data.shape[0]
    train_size = int(0.75 * num_samples)
    val_size = (num_samples - train_size) // 2  # Ensure we have a test set

    train_data = combined_data[:train_size]
    val_data = combined_data[train_size:train_size + val_size]
    test_data = combined_data[train_size + val_size:]

    labels = np.random.randint(0, 2, size=(num_samples,))
    labels_train, labels_val, labels_test = labels[:train_size], labels[train_size:train_size + val_size], labels[train_size + val_size:]

    os.makedirs('PrimeNet/data/pretrain', exist_ok=True)
    os.makedirs('PrimeNet/data/finetune', exist_ok=True)

    torch.save(torch.tensor(train_data, dtype=torch.float32), 'PrimeNet/data/pretrain/X_train.pt')
    torch.save(torch.tensor(val_data, dtype=torch.float32), 'PrimeNet/data/pretrain/X_val.pt')

    torch.save(torch.tensor(train_data, dtype=torch.float32), 'PrimeNet/data/finetune/X_train.pt')
    torch.save(torch.tensor(val_data, dtype=torch.float32), 'PrimeNet/data/finetune/X_val.pt')
    torch.save(torch.tensor(test_data, dtype=torch.float32), 'PrimeNet/data/finetune/X_test.pt')

    torch.save(torch.tensor(labels_train, dtype=torch.long), 'PrimeNet/data/finetune/y_train.pt')
    torch.save(torch.tensor(labels_val, dtype=torch.long), 'PrimeNet/data/finetune/y_val.pt')
    torch.save(torch.tensor(labels_test, dtype=torch.long), 'PrimeNet/data/finetune/y_test.pt')

    print(f"Data generation and saving completed.")
    print(f"X_train: {train_data.shape}")
    print(f"X_val: {val_data.shape}")
    print(f"X_test: {test_data.shape}")

if __name__ == '__main__':
    T_train = np.load("mTAN/src/T_train.npy")
    X_train = np.load("mTAN/src/X_train.npy")
    M_train = np.load("mTAN/src/M_train.npy")
    T_test = np.load("mTAN/src/T_test.npy")
    X_test = np.load("mTAN/src/X_test.npy")
    M_test = np.load("mTAN/src/M_test.npy")

    T = np.concatenate((T_train, T_test), axis=0)
    X = np.concatenate((X_train, X_test), axis=0)
    M = np.concatenate((M_train, M_test), axis=0)

    print(f"Shapes - T: {T.shape}, X: {X.shape}, M: {M.shape}")
    assert T.shape[0] == X.shape[0] == M.shape[0], "Number of samples in T, X, and M do not match."
    assert X.shape[1] == M.shape[1], "Number of features in X and M do not match."

    prepare_data_for_primenet(T, X, M)
