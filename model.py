import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
import torchvision
import librosa


class MyModel(nn.Module):
    def __init__(self, num_classes=5):
        super(MyModel, self).__init__()

        self.input_layer = nn.Conv2d(1, 3, 3, 1, 1)
        self.input_batnorm = nn.BatchNorm2d(3)

        self.resnet = torchvision.models.resnet50(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 512)
        self.fc_batnorm = nn.BatchNorm1d(512)
        self.fc = nn.Linear(512, num_classes)

        self.act_fn = nn.ReLU()

    def forward(self, x):
        x = self.input_layer(x)
        x = self.input_batnorm(x)
        x = self.act_fn(x)

        x = self.resnet(x)
        x = self.fc_batnorm(x)
        x = self.act_fn(x)

        x = self.fc(x)
        return x

    def get_img(self, file_path):
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Resize((84, 84))])

        frame_length = 0.005
        frame_stride = 0.002

        # mel-spectrogram
        y, sr = librosa.load(file_path, sr=16000)

        # wav_length = len(y)/sr
        input_nfft = int(round(sr * frame_length))
        input_stride = int(round(sr * frame_stride))

        S = librosa.feature.melspectrogram(y=y, n_mels=84, n_fft=input_nfft, hop_length=input_stride)
        # print("Wav length: {}, Mel_S shape:{}".format(len(y) / sr, np.shape(S)))

        result = transform(S)
        return result

    def inference(self, wav_file):
        self.eval()
        with torch.no_grad():
            img = self.get_img(wav_file)
            output = self.forward(img.view(-1, 1, 84, 84))

        return output


class MyModel_1(nn.Module):
    def __init__(self, num_classes=5):
        super(MyModel_1, self).__init__()

        self.input_layer = nn.Conv2d(1, 3, 3, 1, 1)
        self.input_batnorm = nn.BatchNorm2d(3)

        self.resnet = torchvision.models.resnet50(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 512)
        self.att1 = nn.Linear(512, 64)
        self.att2 = nn.Linear(64, 512)
        self.fc_batnorm = nn.BatchNorm1d(512)
        self.fc = nn.Linear(512, num_classes)

        self.act_fn = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.input_layer(x)
        x = self.input_batnorm(x)
        x = self.act_fn(x)

        x = self.resnet(x)

        att = self.att1(x)
        att = self.act_fn(att)
        att = self.att2(att)
        att = self.sigmoid(att)

        x = self.fc_batnorm(x)
        x = self.act_fn(x)

        x = x * att

        x = self.fc(x)
        return x

    def get_img(self, file_path):
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Resize((84, 84))])

        frame_length = 0.005
        frame_stride = 0.002

        # mel-spectrogram
        y, sr = librosa.load(file_path, sr=16000)

        # wav_length = len(y)/sr
        input_nfft = int(round(sr * frame_length))
        input_stride = int(round(sr * frame_stride))

        S = librosa.feature.melspectrogram(y=y, n_mels=84, n_fft=input_nfft, hop_length=input_stride)
        # print("Wav length: {}, Mel_S shape:{}".format(len(y) / sr, np.shape(S)))

        result = transform(S)
        return result

    def inference(self, wav_file):
        self.eval()
        with torch.no_grad():
            img = self.get_img(wav_file)
            output = self.forward(img.view(-1, 1, 84, 84))

        return output


class MyModel_2(nn.Module):
    def __init__(self, num_classes=5):
        super(MyModel_2, self).__init__()

        self.input_layer = nn.Conv2d(1, 3, 3, 1, 1)
        self.input_batnorm = nn.BatchNorm2d(3)

        self.resnet = torchvision.models.resnet50(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 512)
        self.fc_batnorm = nn.BatchNorm1d(512)
        self.fc = nn.Linear(512, num_classes)

        self.act_fn = nn.ReLU()

    def forward(self, x, pos=None, neg=None):
        x = self.input_layer(x)
        x = self.input_batnorm(x)
        x = self.act_fn(x)

        anc_embed = self.resnet(x)
        output = self.fc_batnorm(anc_embed)
        output = self.act_fn(output)

        output = self.fc(output)

        if pos is None or neg is None:
            return output, anc_embed
        else:
            p = self.input_layer(pos)
            p = self.input_batnorm(p)
            p = self.act_fn(p)
            pos_embed = self.resnet(p)

            n = self.input_layer(neg)
            n = self.input_batnorm(n)
            n = self.act_fn(n)
            neg_embed = self.resnet(n)
            return output, anc_embed, pos_embed, neg_embed



    def get_img(self, file_path):
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Resize((84, 84))])

        frame_length = 0.005
        frame_stride = 0.002

        # mel-spectrogram
        y, sr = librosa.load(file_path, sr=16000)

        # wav_length = len(y)/sr
        input_nfft = int(round(sr * frame_length))
        input_stride = int(round(sr * frame_stride))

        S = librosa.feature.melspectrogram(y=y, n_mels=84, n_fft=input_nfft, hop_length=input_stride)
        # print("Wav length: {}, Mel_S shape:{}".format(len(y) / sr, np.shape(S)))

        result = transform(S)
        return result

    def inference(self, wav_file):
        self.eval()
        with torch.no_grad():
            img = self.get_img(wav_file)
            output, _ = self.forward(img.view(-1, 1, 84, 84))

        return output


def get_Model(class_name):
    try:
        Myclass = eval(class_name)()
        return Myclass
    except NameError as e:
        print("Class [{}] is not defined".format(class_name))


def main():
    pass


if __name__ == "__main__":
    main()
