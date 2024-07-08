import h5py
import os

def check_h5_file(file_path):
    try:
        with h5py.File(file_path, 'r') as f:
            print("Successfully opened the file.")
            print("File keys:", list(f.keys()))
            # If 'time', 'data', 'mask' are expected datasets, check their existence
            if 'time' in f and 'data' in f and 'mask' in f:
                print("Datasets found: 'time', 'data', 'mask'")
                print("time shape:", f['time'].shape)
                print("data shape:", f['data'].shape)
                print("mask shape:", f['mask'].shape)
            else:
                print("Expected datasets not found. Available datasets:", list(f.keys()))
    except OSError as e:
        print(f"Error opening file: {e}")

# Verify the file path
print("Current directory contents:", os.listdir('.'))
print("mTAN/src directory contents:", os.listdir('mTAN/src/'))

# Check the .h5 file
check_h5_file('mTAN/src/toy_mtan_rnn_mtan_rnn_85419.h5')
