import librosa
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class MyDataset(Dataset):
    def __init__(self, data):
        super(MyDataset, self).__init__()

        self.data = data
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Resize((84, 84))])

        self.label_tag = {'Hat': 0, 'Snare': 1, 'Kick': 2, 'Clap': 3, 'Cymbals': 4}
        for i in range(len(self.data)):
            self.data.iloc[i, 1] = self.label_tag[self.data.iloc[i, 1]]

    def get_img(self, file_path):
        frame_length = 0.005
        frame_stride = 0.002

        # mel-spectrogram
        y, sr = librosa.load(file_path, sr=16000)

        # wav_length = len(y)/sr
        input_nfft = int(round(sr * frame_length))
        input_stride = int(round(sr * frame_stride))

        S = librosa.feature.melspectrogram(y=y, n_mels=84, n_fft=input_nfft, hop_length=input_stride)
        # print("Wav length: {}, Mel_S shape:{}".format(len(y) / sr, np.shape(S)))
        return S

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path = self.data.iloc[idx][0]
        label = self.data.iloc[idx][1]

        img = self.transform(self.get_img(file_path))
        target = F.one_hot(torch.tensor(label), num_classes=len(self.label_tag))

        return img, target.float()

    def show_item(self, idx=0):
        feature, label = self.__getitem__(idx)

        print("Feature's Shape : {}".format(feature.shape))
        print("Label's Shape : {}".format(label.shape))

        return feature, label


class MyDataset_triplet(Dataset):
    def __init__(self, data):
        super(MyDataset_triplet, self).__init__()

        self.data = data
        self.index = np.array(range(len(self.data)))  # [0, 1, ,,, , n-1, n]
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Resize((84, 84))])

        self.label_tag = {'Hat': 0, 'Snare': 1, 'Kick': 2, 'Clap': 3, 'Cymbals': 4}
        for i in range(len(self.data)):
            self.data.iloc[i, 1] = self.label_tag[self.data.iloc[i, 1]]

    def get_img(self, file_path):
        frame_length = 0.005
        frame_stride = 0.002

        # mel-spectrogram
        y, sr = librosa.load(file_path, sr=16000)

        # wav_length = len(y)/sr
        input_nfft = int(round(sr * frame_length))
        input_stride = int(round(sr * frame_stride))

        S = librosa.feature.melspectrogram(y=y, n_mels=84, n_fft=input_nfft, hop_length=input_stride)
        # print("Wav length: {}, Mel_S shape:{}".format(len(y) / sr, np.shape(S)))
        return S

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path = self.data.iloc[idx][0]
        label = self.data.iloc[idx][1]
        base_query = self.index != idx

        anc_img = self.transform(self.get_img(file_path))
        target = F.one_hot(torch.tensor(label), num_classes=len(self.label_tag))

        pos_query = (self.data['label'] == label).to_numpy().reshape(-1)
        pos_index = self.index[base_query & pos_query]
        pos_idx = np.random.choice(pos_index)

        neg_query = (self.data['label'] != label).to_numpy().reshape(-1)
        neg_index = self.index[base_query & neg_query]
        neg_idx = np.random.choice(neg_index)

        pos_img = self.transform(self.get_img(self.data.iloc[pos_idx][0]))
        neg_img = self.transform(self.get_img(self.data.iloc[neg_idx][0]))

        return anc_img, pos_img, neg_img, target.float()

    def show_item(self, idx=0):
        feature, label = self.__getitem__(idx)

        print("Feature's Shape : {}".format(feature.shape))
        print("Label's Shape : {}".format(label.shape))

        return feature, label


def main():
    train = pd.read_csv('train.csv')

    md = MyDataset(train)
    loader = DataLoader(md, batch_size=4)

    i = 0
    for img, target in loader:
        print(img.shape)
        print(target.shape)
        if i == 5:
            break
        i += 1


if __name__ == "__main__":
    main()
