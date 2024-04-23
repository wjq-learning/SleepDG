import numpy as np
from tqdm import tqdm
import os

dir_path = r'/data/datasets/hang7_pre_100hz/'
f_names = os.listdir(dir_path)

seq_dir = r'/data/datasets/hang7_pre_100hz_seq/seq/'
label_dir = r'/data/datasets/hang7_pre_100hz_seq/labels/'


# print(f_names)

seq_f_names = []
label_f_names = []

for f_name in f_names:
    if 'seq' in f_name:
        seq_f_names.append(f_name)
    if 'label' in f_name:
        label_f_names.append(f_name)

seq_f_names.sort()
label_f_names.sort()

print(seq_f_names)
print(label_f_names)

for seq_f_name in tqdm(seq_f_names):
   if not os.path.exists(label_dir + seq_f_name[:8]):
       os.makedirs(label_dir + seq_f_name[:8])

i = 0
for seq_f_name in tqdm(seq_f_names):
    seqs = np.load(dir_path + seq_f_name)
    for seq in seqs:
        seq_name = f'/data/datasets/hang7_pre_100hz_seq/seq/{seq_f_name[:8]}/{seq_f_name[:8]}-{str(i)}.npy'
        with open(seq_name, 'wb') as f:
            np.save(f, seq)
        i += 1

j = 0
for label_f_name in tqdm(label_f_names):
    labels = np.load(dir_path + label_f_name)
    for label in labels:
        label_name = f'/data/datasets/hang7_pre_100hz_seq/labels/{label_f_name[:8]}/{label_f_name[:8]}-{str(j)}.npy'
        with open(label_name, 'wb') as f:
            np.save(f, label)
        j += 1

