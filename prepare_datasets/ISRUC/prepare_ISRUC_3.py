# %%
from mne.io import concatenate_raws
from prepare_datasets.ISRUC.edf import read_raw_edf
import matplotlib.pyplot as plt
import mne
import os
import numpy as np
from tqdm import tqdm
import xml.etree.ElementTree as ET
from sklearn.preprocessing import StandardScaler


dir_path = r'/data/datasets2/ISRUC_extracted/group3'

seq_dir = r'/data/datasets/GSS_datasets/ISRUC/seq'
label_dir = r'/data/datasets/GSS_datasets/ISRUC/labels'

psg_f_names = []
label_f_names = []
for i in range(1, 11):
    numstr = str(i)
    psg_f_names.append(f'{dir_path}/{numstr}/{numstr}.rec')
    label_f_names.append(f'{dir_path}/{numstr}/{numstr}_1.txt')

# psg_f_names.sort()
# label_f_names.sort()

psg_label_f_pairs = []
for psg_f_name, label_f_name in zip(psg_f_names, label_f_names):
    if psg_f_name[:-4] == label_f_name[:-6]:
        psg_label_f_pairs.append((psg_f_name, label_f_name))
for item in psg_label_f_pairs:
    print(item)

label2id = {'0': 0,
            '1': 1,
            '2': 2,
            '3': 3,
            '5': 4,}
print(label2id)
# %%
# signal_name = ['LOC-A2', 'F4-A1']
n = 0
num_seqs = 0
num_labels = 0
for psg_f_name, label_f_name in tqdm(psg_label_f_pairs):
    n += 1
    labels_list = []

    raw = read_raw_edf(os.path.join(dir_path, psg_f_name), preload=True)
    # raw.pick_channels(signal_name)
    raw.resample(sfreq=100)
    raw.filter(0.3, 35, fir_design='firwin')
    print(raw.info)

    psg_array = raw.to_data_frame().values
    # print(psg_array[:10, 0])
    print(psg_array.shape)
    psg_array = psg_array[:, 1:]
    eeg_array = psg_array[:, 5:6]
    eog_array = psg_array[:, 0:1]
    psg_array = np.concatenate((eeg_array, eog_array), axis=1)
    print(psg_array.shape)

    std = StandardScaler()
    psg_array = std.fit_transform(psg_array)

    i = psg_array.shape[0] % (30 * 100)
    if i > 0:
        psg_array = psg_array[:-i, :]
    print(psg_array.shape)
    psg_array = psg_array.reshape(-1, 30 * 100, 2)
    print(psg_array.shape)

    a = psg_array.shape[0] % 20
    if a > 0:
        psg_array = psg_array[:-a, :, :]
    print(psg_array.shape)
    psg_array = psg_array.reshape(-1, 20, 30 * 100, 2)
    epochs_seq = psg_array.transpose(0, 1, 3, 2)
    print(epochs_seq.shape)
    # print(epochs_seq[39, 19, 0, 1:9])

    for line in open(os.path.join(dir_path, label_f_name)).readlines():
        line_str = line.strip()
        if line_str != '':
            labels_list.append(label2id[line_str])
    labels_array = np.array(labels_list)
    if a > 0:
        labels_array = labels_array[:-a]
    labels_seq = labels_array.reshape(-1, 20)
    print(labels_seq.shape)

    if not os.path.isdir(f'{seq_dir}/ISRUC-group3-{str(n)}'):
        os.makedirs(f'{seq_dir}/ISRUC-group3-{str(n)}')
    for seq in epochs_seq:
        seq_name = f'{seq_dir}/ISRUC-group3-{str(n)}/ISRUC-group3-{str(n)}-{str(num_seqs)}.npy'
        with open(seq_name, 'wb') as f:
            np.save(f, seq)
        num_seqs += 1

    if not os.path.isdir(f'{label_dir}/ISRUC-group3-{str(n)}'):
        os.makedirs(f'{label_dir}/ISRUC-group3-{str(n)}')
    for label in labels_seq:
        label_name = f'{label_dir}/ISRUC-group3-{str(n)}/ISRUC-group3-{str(n)}-{str(num_labels)}.npy'
        with open(label_name, 'wb') as f:
            np.save(f, label)
        num_labels += 1



