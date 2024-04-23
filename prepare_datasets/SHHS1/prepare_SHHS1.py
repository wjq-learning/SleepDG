# %%
from mne.io import concatenate_raws, read_raw_edf
import matplotlib.pyplot as plt
import mne
import os
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import xml.etree.ElementTree as ET

dir_path_psg = r'/data/datasets/shhs/polysomnography/edfs/shhs1'
dir_path_ann = r'/data/datasets/shhs/polysomnography/annotations-events-profusion/shhs1'

seq_dir = r'/data/datasets/GSS_datasets/SHHS1/seq'
label_dir = r'/data/datasets/GSS_datasets/SHHS1/labels'

psg_f_names = os.listdir(dir_path_psg)
label_f_names = os.listdir(dir_path_ann)

psg_f_names.sort()
label_f_names.sort()
# psg_f_names = psg_f_names[53:150]
# label_f_names = label_f_names[53:150]

psg_label_f_pairs = []
for psg_f_name, label_f_name in zip(psg_f_names, label_f_names):
    if psg_f_name[:12] == label_f_name[:12]:
        psg_label_f_pairs.append((psg_f_name, label_f_name))
print(psg_label_f_pairs)
print(len(psg_label_f_pairs))
label2id = {'0': 0,
            '1': 1,
            '2': 2,
            '3': 3,
            '4': 3,
            '5': 4,
            '9': 0}
print(label2id)
# %%
num_seqs = 0
num_labels = 0
signal_name = ['EEG', 'EOG(L)']

for psg_f_name, label_f_name in tqdm(psg_label_f_pairs[:150]):
    labels_list = []

    raw = read_raw_edf(os.path.join(dir_path_psg, psg_f_name), preload=True)
    print(raw.info)
    raw.pick_channels(signal_name)
    raw.resample(sfreq=100)
    raw.filter(0.3, 35, fir_design='firwin')
    print(raw.info)

    psg_array = raw.to_data_frame().values
    # print(psg_array[:10, 0])
    print(psg_array.shape)
    psg_array = psg_array[:, 1:]

    std = StandardScaler()
    psg_array = std.fit_transform(psg_array)
    print(psg_array[:10, :])

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

    tree = ET.parse(os.path.join(dir_path_ann, label_f_name))
    root = tree.getroot()
    for child in root.iter('SleepStage'):
        labels_list.append(label2id[child.text])
    labels_array = np.array(labels_list)
    if a > 0:
        labels_array = labels_array[:-a]
    print(labels_array.shape)
    labels_seq = labels_array.reshape(-1, 20)
    print(labels_seq.shape)

    # if not os.path.isdir(f'{seq_dir}/{psg_f_name[:12]}'):
    #     os.makedirs(f'{seq_dir}/{psg_f_name[:12]}')
    # for seq in epochs_seq:
    #     seq_name = f'{seq_dir}/{psg_f_name[:12]}/{psg_f_name[:12]}-{str(num_seqs)}.npy'
    #     with open(seq_name, 'wb') as f:
    #         np.save(f, seq)
    #     num_seqs += 1
    #
    # if not os.path.isdir(f'{label_dir}/{label_f_name[:12]}'):
    #     os.makedirs(f'{label_dir}/{label_f_name[:12]}')
    # for label in labels_seq:
    #     label_name = f'{label_dir}/{label_f_name[:12]}/{label_f_name[:12]}-{str(num_labels)}.npy'
    #     with open(label_name, 'wb') as f:
    #         np.save(f, label)
    #     num_labels += 1


# %%
