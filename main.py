# https://github.com/HideOnHouse/TorchBase

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from dataset import *
from model import *
from learning import *
from inference import inference
from GUI import *


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def calc_euclidean(self, x1, x2):
        return (x1 - x2).pow(2).sum(1)

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor):
        distance_positive = self.calc_euclidean(anchor, positive)
        distance_negative = self.calc_euclidean(anchor, negative)
        losses = torch.relu(distance_positive - distance_negative + self.margin)

        return losses.mean()


class My_Triplet_Loss(nn.Module):
    def __init__(self, p=0.1, margin=2.0):
        super(My_Triplet_Loss, self).__init__()

        self.p = p
        self.triplet = TripletLoss(margin=1.0)
        self.CE = nn.CrossEntropyLoss()

    def forward(self, predict, x_embed, p_embed, n_embed, target):
        triplet = self.triplet(x_embed, p_embed, n_embed)
        ce = self.CE(predict, target)
        return ce + (self.p * triplet)


def draw_history(history):
    train_loss = history["train_loss"]
    train_acc = history["train_acc"]
    valid_loss = history["valid_loss"]
    valid_acc = history["valid_acc"]

    plt.subplot(2, 1, 1)
    plt.plot(train_loss, label="train")
    plt.plot(valid_loss, label="valid")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(train_acc, label="train")
    plt.plot(valid_acc, label="valid")
    plt.legend()

    plt.show()


def training(model_name='MyModel_1', mode='basis'):
    train_path = "train.csv"
    test_path = "test.csv"

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    # your Data Pre-Processing
    train_x, train_y = train_data.iloc[:, :1], train_data.iloc[:, 1:]
    test_x, test_y = test_data.iloc[:, :1], test_data.iloc[:, 1:]

    # data split
    train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, stratify=train_y, random_state=17,
                                                          test_size=0.05)

    # Check Train, Valid, Test Image's Shape
    print("The Shape of Train Images: ", train_x.shape)
    print("The Shape of Valid Images: ", valid_x.shape)
    print("The Shape of Test Images: ", test_x.shape)

    # Check Train, Valid Label's Shape
    print("The Shape of Train Labels: ", train_y.shape)
    print("The Shape of Valid Labels: ", valid_y.shape)
    print("The Shape of Valid Labels: ", test_y.shape)

    train_df = pd.concat([train_x, train_y], axis=1)
    valid_df = pd.concat([valid_x, valid_y], axis=1)
    test_df = pd.concat([test_x, test_y], axis=1)

    # Create Dataset and DataLoader
    if mode == 'basis':
        train_dataset = MyDataset(train_df)
        valid_dataset = MyDataset(valid_df)
        test_dataset = MyDataset(test_df)
    else:  # mode == 'triplet
        train_dataset = MyDataset_triplet(train_df)
        valid_dataset = MyDataset_triplet(valid_df)
        test_dataset = MyDataset_triplet(test_df)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # label_tags
    label_tag = ['Hat', 'Snare', 'Kick', 'Clap', 'Cymbals']

    model = get_Model(model_name)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    optimizer = torch.optim.AdamW(model.parameters())
    criterion = torch.nn.CrossEntropyLoss() if mode == 'basis' else My_Triplet_Loss()

    # train
    print("============================= Train =============================")
    if mode == 'basis':
        history = train(model, device, optimizer, criterion, 10, train_loader, valid_loader)
    else:
        history = train_triplet(model, device, optimizer, criterion, 10, train_loader, valid_loader)

    # Test
    print("============================= Test =============================")
    if mode == 'basis':
        test_loss, test_acc = evaluate(model, device, criterion, test_loader)
    else:
        test_loss, test_acc = evaluate_triplet(model, device, criterion, test_loader)
    print("test loss : {:.6f}".format(test_loss))
    print("test acc : {:.3f}".format(test_acc))

    file_name = model_name
    torch.save(model, f"models/{file_name}.pt")
    with open(f"models/{file_name}_history.pickle", 'wb') as f:
        pickle.dump(history, f, pickle.HIGHEST_PROTOCOL)

    # print(history)
    draw_history(history)


def model_inference(model_name, mode):
    model = torch.load('models/{}.pt'.format(model_name))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    criterion = torch.nn.CrossEntropyLoss() if mode == 'basis' else My_Triplet_Loss()

    test_path = "test.csv"
    test_df = pd.read_csv(test_path)
    if mode == 'basis':
        test_dataset = MyDataset(test_df)
        test_loader = DataLoader(test_dataset, batch_size=32)
        test_loss, test_acc = evaluate(model, device, criterion, test_loader)
    else:  # mode == 'triplet
        test_dataset = MyDataset_triplet(test_df)
        test_loader = DataLoader(test_dataset, batch_size=32)
        test_loss, test_acc = evaluate_triplet(model, device, criterion, test_loader)

    print("test loss : {:.6f}".format(test_loss))
    print("test acc : {:.3f}".format(test_acc))


def main():
    cmd = input('Train or Inference or GUI?(t or f or g) >>> ')

    if cmd == 't':
        model_name = input('Model name? >>> ')
        mode = input('basis or triplet? >>> ')
        training(model_name, mode)
    elif cmd == 'f':
        model_name = input('Model name? >>> ')
        mode = input('basis or triplet? >>> ')
        model_inference(model_name, mode)
    elif cmd == 'g':
        p = main_GUI()
        p.mainloop()
    else:
        print('Error')


if __name__ == '__main__':
    main()
