# %%
from mne.io import concatenate_raws, read_raw_edf
import matplotlib.pyplot as plt
import mne
import os
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler


dir_path = r'/data/datasets/haaglanden-medisch-centrum-sleep-staging-database-1.1/recordings/'

seq_dir = r'/data/datasets/GSS_datasets/HMC/seq'
label_dir = r'/data/datasets/GSS_datasets/HMC/labels'

f_names = os.listdir(dir_path)

print(f_names)
# %%
psg_f_names = []
label_f_names = []
for f_name in f_names:
    if 'sleepscoring.edf' in f_name:
        label_f_names.append(f_name)
    elif '.edf' in f_name:
        psg_f_names.append(f_name)

psg_f_names.sort()
label_f_names.sort()

psg_label_f_pairs = []
for psg_f_name, label_f_name in zip(psg_f_names, label_f_names):
    if psg_f_name[:5] == label_f_name[:5]:
        psg_label_f_pairs.append((psg_f_name, label_f_name))
print(psg_label_f_pairs)

# for item in psg_label_f_pairs:
#     print(item)

# %%
# dddd = [('SC4651E0-PSG.edf', 'SC4651EP-Hypnogram.edf'), ('SC4652E0-PSG.edf', 'SC4652EG-Hypnogram.edf')]
label2id = {'Sleep stage W': 0,
            'Sleep stage N1': 1,
            'Sleep stage N2': 2,
            'Sleep stage N3': 3,
            'Sleep stage R': 4,
            'Lights off@@EEG F4-A1': 0}
print(label2id)
# %%
num_seqs = 0
num_labels = 0
signal_name = ['EEG F4-M1', 'EOG E1-M2']
for psg_f_name, label_f_name in tqdm(psg_label_f_pairs):
    epochs_list = []
    labels_list = []

    raw = read_raw_edf(os.path.join(dir_path, psg_f_name), preload=True)
    print(raw.info)
    raw.pick_channels(signal_name)
    raw.resample(sfreq=100)
    # raw.filter(0.3, 35, fir_design='firwin')
    annotation = mne.read_annotations(os.path.join(dir_path, label_f_name))
    raw.set_annotations(annotation, emit_warning=False)

    events_train, event_id = mne.events_from_annotations(
        raw, chunk_duration=30.)
    print(event_id)

    key_list = []
    for key in event_id.keys():
        if 'Light' in key:
            key_list.append(key)
    for key in key_list:
        event_id.pop(key)
    print(event_id)
    tmax = 30. - 1. / raw.info['sfreq']  # tmax in included
    epochs_train = mne.Epochs(raw=raw, events=events_train,
                              event_id=event_id, tmin=0., tmax=tmax, baseline=None)
    # print(print(len(epochs_train.get_annotations_per_epoch())))
    print(epochs_train.event_id)
    labels = []
    for epoch_annotation in epochs_train.get_annotations_per_epoch():
        labels.append(epoch_annotation[0][2])
    # for epoch, label in zip(epochs_train, labels):
    #     print(epoch.shape, label)
    length = len(labels)

    epochs = epochs_train[:]
    labels_ = labels[:]
    print(epochs)
    print(labels_)

    for epoch in epochs:
        epochs_list.append(epoch)
    for label in labels_:
        labels_list.append(label2id[label])

    index = len(epochs_list)
    while index % 20 != 0:
        index -= 1
    epochs_list = epochs_list[:index]
    labels_list = labels_list[:index]
    print(len(epochs_list), len(labels_list))
    epochs_array_ = np.array(epochs_list)
    labels_array_ = np.array(labels_list)
    print(epochs_array_.shape, labels_array_.shape)

    epochs_array_ = epochs_array_.transpose(0, 2, 1)
    epochs_array_ = epochs_array_.reshape(-1, 2)
    std = StandardScaler()
    epochs_array_ = std.fit_transform(epochs_array_)
    print(epochs_array_.shape)
    epochs_array_ = epochs_array_.reshape(-1, 3000, 2)
    epochs_array_ = epochs_array_.transpose(0, 2, 1)
    print(epochs_array_.shape)

    epochs_seq = epochs_array_.reshape(-1, 20, 2, 3000)
    labels_seq = labels_array_.reshape(-1, 20)
    print(epochs_seq.shape, labels_seq.shape)
    print(epochs_seq.dtype, labels_seq.dtype)
    # print(epochs_seq[0, 0, 0, 687:697])

    if not os.path.isdir(f'{seq_dir}/{psg_f_name[:5]}'):
        os.makedirs(f'{seq_dir}/{psg_f_name[:5]}')
    for seq in epochs_seq:
        seq_name = f'{seq_dir}/{psg_f_name[:5]}/{psg_f_name[:5]}-{str(num_seqs)}.npy'
        with open(seq_name, 'wb') as f:
            np.save(f, seq)
        num_seqs += 1

    if not os.path.isdir(f'{label_dir}/{label_f_name[:5]}'):
        os.makedirs(f'{label_dir}/{label_f_name[:5]}')
    for label in labels_seq:
        label_name = f'{label_dir}/{label_f_name[:5]}/{label_f_name[:5]}-{str(num_labels)}.npy'
        with open(label_name, 'wb') as f:
            np.save(f, label)
        num_labels += 1


# %%
