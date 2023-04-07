import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from learning import evaluate
import librosa

from dataset import *
from model import *

def inference(model, wav_file):
    label_tag = {'Hat': 0, 'Snare': 1, 'Kick': 2, 'Clap': 3, 'Cymbals': 4}
    tag_label = {0: 'Hat', 1: 'Snare', 2: 'Kick', 3: 'Clap', 4: 'Cymbals'}

    model.cpu()
    predict = model.inference(wav_file)
    predict = predict.view(-1)
    predict = F.softmax(predict, dim=-1)
    m_v, m_i = torch.max(predict, dim=-1)

    return predict.numpy() * 100, m_i.item(), tag_label[m_i.item()]


def main():
    # model = torch.load('models/model.pt')
    #
    # result = inference(model, 'C:/Users/조상연/Downloads/clap_test.wav')
    # print(result)
    return


if __name__ == '__main__':
    main()