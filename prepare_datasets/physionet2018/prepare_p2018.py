import wfdb
from wfdb.processing import resample_sig, resample_multichan
from mne.io import concatenate_raws, read_raw_edf
import matplotlib.pyplot as plt
import mne
import os
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from scipy import signal

dir_path = r'/data/datasets/Physionet2018/'

seq_dir = r'/data/datasets/GSS_datasets/P2018/seq'
label_dir = r'/data/datasets/GSS_datasets/P2018/labels'

subject_dirs = os.listdir(dir_path)
subject_dirs.sort()

psg_label_f_pairs = []
for item in subject_dirs:
    psg_label_f_pairs.append((f'{dir_path}{item}/{item}.mat', f'{dir_path}{item}/{item}.arousal'))

for item in psg_label_f_pairs:
    print(item)

num_seqs = 0
num_labels = 0

label2id = {'W': 0,
            'N1': 1,
            'N2': 2,
            'N3': 3,
            'R': 4}

for psg_f_name, label_f_name in tqdm(psg_label_f_pairs):
    signals, fields = wfdb.rdsamp(psg_f_name[:-4], channel_names=['C3-M2', 'O1-M2'])
    print(fields)
    ann = wfdb.rdann(label_f_name[:-8], label_f_name[-7:])
    signals, ann = resample_multichan(signals, ann, 200, 100)
    print(signals.shape)
    # print(ann.sample)
    # print(ann.aux_note)
    b, a = signal.butter(8, [0.006, 0.7], 'bandpass')
    signals = signal.filtfilt(b, a, signals, axis=0)
    signals = signals[ann.sample[0]:, :]
    temp = signals.shape[0] % 3000
    if temp != 0:
        signals = signals[:-temp]
    epochs_num = signals.shape[0] // 3000

    ann_labels = []
    start = 0
    for i, label in enumerate(ann.aux_note):
        if label in label2id.keys():
            if start == 0:
                start = ann.sample[i]
            ann_labels.append((ann.sample[i]-start, label))
    print(ann_labels)
    std = StandardScaler()
    signals = std.fit_transform(signals)
    print(signals.shape)
    # print(signals[:10, :])
    signals = signals.reshape(-1, 3000, 2)
    print(signals.shape)
    signals = signals.transpose(0, 2, 1)

    labels = []
    begin = 0
    end = 0
    for k in range(len(ann_labels)-1):
        begin = int(ann_labels[k][0]) // 3000
        end = int(ann_labels[k+1][0]) // 3000
        for i in range(begin, end):
            labels.append(label2id[ann_labels[k][1]])
    for i in range(end, epochs_num):
        labels.append(label2id[ann_labels[-1][1]])
    labels = np.array(labels)
    print(labels.shape)

    index = signals.shape[0]
    if index % 20 != 0:
        a = index % 20
        signals = signals[:-a, :, :]
        labels = labels[:-a]

    print(signals.shape)
    print(labels.shape)
    epochs_seq = signals.reshape(-1, 20, 2, 3000)
    labels_seq = labels.reshape(-1, 20)
    print(epochs_seq.shape)
    print(labels_seq.shape)
    #
    if not os.path.isdir(f'{seq_dir}/{psg_f_name[-13:-4]}'):
        os.makedirs(f'{seq_dir}/{psg_f_name[-13:-4]}')
    for seq in epochs_seq:
        seq_name = f'{seq_dir}/{psg_f_name[-13:-4]}/{psg_f_name[-13:-4]}-{str(num_seqs)}.npy'
        with open(seq_name, 'wb') as f:
            np.save(f, seq)
        num_seqs += 1

    if not os.path.isdir(f'{label_dir}/{label_f_name[-17:-8]}'):
        os.makedirs(f'{label_dir}/{label_f_name[-17:-8]}')
    for label in labels_seq:
        label_name = f'{label_dir}/{label_f_name[-17:-8]}/{label_f_name[-17:-8]}-{str(num_labels)}.npy'
        with open(label_name, 'wb') as f:
            np.save(f, label)
        num_labels += 1