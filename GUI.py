import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk
import tkinter.messagebox as msg
from tkinter import filedialog
import numpy as np
import pandas as pd
from playsound import playsound

from dataset import MyDataset
from model import MyModel
from learning import train, evaluate, calc_acc
from inference import inference

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class main_GUI(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)

        self.font = {"title": tkfont.Font(family="Malgun Gothic", size=32, weight="bold"),
                     "sub_title": tkfont.Font(family="Malgun Gothic", size=16, weight="bold"),
                     "contents1": tkfont.Font(family="Malgun Gothic", size=10, weight="bold"),
                     "contents2": tkfont.Font(family="Malgun Gothic", size=9),
                     "contents3": tkfont.Font(family="Malgun Gothic", size=12, weight="bold")}

        self.iconphoto(False, tk.PhotoImage(file='icon/logo.png'))
        self.title('JSY Percussion Classification')
        self.geometry('640x600')
        self.resizable(False, False)

        self.select_img = tk.PhotoImage(file='icon/playlist.png')
        self.clear_img = tk.PhotoImage(file='icon/reset.png')
        self.analysis_img = tk.PhotoImage(file='icon/analysis.png')
        self.exit_img = tk.PhotoImage(file='icon/exit.png')
        self.play_img = tk.PhotoImage(file='icon/play.png').subsample(2, 2)

        self._frame = None
        self.switch_frame(main_frame)

    def switch_frame(self, frame_class):
        """
        frame간 전환을 구현한 메소드
        :param frame_class: 스위치할 frame class명을 입력
        """
        new_frame = frame_class(self)  # 매개변수로 받은 frame 객체를 생성
        if self._frame is not None:  # 기존에 있던 frame 삭제
            self._frame.destroy()
        self._frame = new_frame  # 새로운 프레임 적용
        self._frame.pack()  # pack()을 통한 배치


class main_frame(ttk.Frame):
    def __init__(self, app):
        ttk.Frame.__init__(self, app)
        self.model = torch.load('models/prototype.pt')
        self.label_tag = {'Hat': 0, 'Snare': 1, 'Kick': 2, 'Clap': 3, 'Cymbals': 4}
        self.tag_label = {0: 'Hat', 1: 'Snare', 2: 'Kick', 3: 'Clap', 4: 'Cymbals'}

        self.app = app

        ttk.Label(self, text="타악기 분류기", font=self.app.font["title"]).pack()
        ttk.Label(self, text="학습된 AI 모델이 타악기 소리를 분류합니다.", font=self.app.font["sub_title"]).pack()
        ttk.Label(self, text="타악기 구분 : [Hat   Snare   Kick   Clap   Cymbals]", font=self.app.font["contents1"]).pack()

        self.select_file_frame = tk.LabelFrame(self, text='Select File', font=self.app.font["contents3"])

        self.select_model_frame = tk.Frame(self.select_file_frame)
        self.model_combo = ttk.Combobox(self.select_model_frame, values=['Prototype', 'Attention', 'Triplet'],
                                        state='readonly')
        self.model_combo.bind("<<ComboboxSelected>>", self.model_select)
        self.model_combo.current(0)
        self.model_combo.pack(side='right')
        tk.Label(self.select_model_frame, text='AI 모델').pack(side='right')
        self.select_model_frame.pack(anchor='e')

        self.display_file = tk.Listbox(self.select_file_frame, selectmode='browse', width=75, height=15)
        self.display_file.pack()

        self.btn_frame = ttk.Frame(self.select_file_frame)
        ttk.Button(self.btn_frame, text="파일선택", image=self.app.select_img, compound='left',
                   command=self.open_file).pack(side='right')
        ttk.Button(self.btn_frame, text='초기화', image=self.app.clear_img, compound='left',
                   command=self.selected_files_clear).pack(side='right')
        ttk.Button(self.btn_frame, text="파일분석", image=self.app.analysis_img, compound='left',
                   command=self.model_inference).pack(side='right')
        ttk.Button(self.btn_frame, text='재생', image=self.app.play_img, compound='left', command=self.play_wav).pack(
            side='right')
        self.btn_frame.pack()

        self.analysis_frame = tk.Frame(self.select_file_frame, background='white')
        self.info1 = tk.Label(self.analysis_frame, text='해당 파일의 소리는\n_____% 확률로 ____입니다.',
                              font=self.app.font["contents3"], background='white')
        self.info2 = tk.Label(self.analysis_frame, text='Hat :\t\nSnare :\t\nKick :\t\nClap :\t\nCymabals :\t',
                              font=self.app.font["contents2"], background='white')
        self.info1.pack(side='left', anchor='w');
        self.info2.pack(side='right', anchor='e', pady=5)
        self.analysis_frame.pack(expand=True, padx=10)

        self.select_file_frame.pack()

        ttk.Button(self, text="종료하기", image=self.app.exit_img, compound='left', command=self.app.quit).pack(
            side='bottom')

    def model_select(self, event):
        model_dict = {'Prototype': 'models/prototype.pt',
                      'Attention': 'models/MyModel_1.pt',
                      'Triplet': 'models/MyModel_2.pt'}
        model_path = model_dict[self.model_combo.get()]
        self.model = torch.load(model_path)

    def open_file(self):
        files = filedialog.askopenfilenames(initialdir='', title='select wav files',
                                            filetypes=(('wav files', '*.wav'),
                                                       ('all files', '*.*')))
        for i, f in enumerate(files):
            self.display_file.insert(i, f)

    def selected_files_clear(self):
        self.display_file.delete(0, self.display_file.size() - 1)

    def model_inference(self):
        selected_list = self.display_file.get(0, self.display_file.size() - 1)
        selected_idx = self.display_file.curselection()[0]
        selected_file = selected_list[selected_idx]

        predict, idx, label_tag = inference(self.model, selected_file)
        info1 = '해당 파일의 소리는\n{:.2f}% 확률로 {}입니다.'.format(predict[idx], label_tag)
        info2 = ''
        for i, p in enumerate(predict):
            info2 += self.tag_label[i] + " : " + '{:.3f}%'.format(p) + '\n'
        self.info1.config(text=info1)
        self.info2.config(text=info2[:-1])

    def play_wav(self):
        selected_list = self.display_file.get(0, self.display_file.size() - 1)
        selected_idx = self.display_file.curselection()[0]
        selected_file = selected_list[selected_idx]
        playsound(selected_file)
        print('play : {}'.format(selected_file))


def main():
    p = main_GUI()
    p.mainloop()


if __name__ == '__main__':
    main()
