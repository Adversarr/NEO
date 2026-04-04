"""Test script for SMPLXDeformationDataset."""

import sys
sys.path.insert(0, "/home/adversarr/Repo/g2pt/src")

from g2pt.data.datasets.smplx_deform import SMPLXDeformationDataset, collate_fn
from torch.utils.data import DataLoader

# Test with training data
train_file = "/home/adversarr/Repo/g2pt/exp/downstream/train_baked_hands_30_5.0.hdf5"
val_file = "/home/adversarr/Repo/g2pt/exp/downstream/val_baked_hands_20_5.0.hdf5"

print("Testing SMPLXDeformationDataset...")
print("=" * 50)

# Create dataset
dataset = SMPLXDeformationDataset(
    data_file=train_file,
    enable_rotate=0.0,
    target_k=128,
    delta=1.0,
)

print(f"\nDataset length: {len(dataset)}")

# Get a single sample
print("\nFetching first sample...")
sample = dataset[0]

print("\nSample keys:", sample.keys())
print("\nSample shapes:")
for key, value in sample.items():
    if hasattr(value, "shape"):
        print(f"  {key}: {value.shape}")
    else:
        print(f"  {key}: {type(value)}")

print("\nSample dtypes:")
for key, value in sample.items():
    if hasattr(value, "dtype"):
        print(f"  {key}: {value.dtype}")

# Test DataLoader
print("\n" + "=" * 50)
print("Testing DataLoader...")
dataloader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=0,
)

batch = next(iter(dataloader))
print("\nBatch keys:", batch.keys())
print("\nBatch shapes:")
for key, value in batch.items():
    if isinstance(value, list):
        print(f"  {key}: list of {len(value)} items")
    elif hasattr(value, "shape"):
        print(f"  {key}: {value.shape}")

dataset.close()
print("\n" + "=" * 50)
print("Test completed successfully!")
