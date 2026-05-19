"""
Omer Seyfettin hikayelerini karakter seviyesinde tokenize eder.
"""
import os
import pickle
import numpy as np

# input.txt oku
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
if not os.path.exists(input_file_path):
    raise FileNotFoundError("input.txt yok! Once download_data.py calistir.")

with open(input_file_path, 'r', encoding='utf-8') as f:
    data = f.read()

print(f"Veri uzunlugu: {len(data):,} karakter")

# Benzersiz karakterler
chars = sorted(list(set(data)))
vocab_size = len(chars)
print(f"Vocab size: {vocab_size}")
print(f"Karakterler: {''.join(chars)}")

# Mapping
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(s):
    return [stoi[c] for c in s]


# Train / val split
n = len(data)
train_data = data[:int(n * 0.9)]
val_data = data[int(n * 0.9):]

train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"Train tokens: {len(train_ids):,}")
print(f"Val tokens:   {len(val_ids):,}")

# Binary'ye kaydet
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)

train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# Meta
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

print("\nOK! train.bin, val.bin, meta.pkl olusturuldu.")