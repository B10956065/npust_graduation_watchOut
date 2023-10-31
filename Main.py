import threading
import tkinter as tk
import time
from tkinter import font, filedialog, messagebox
from tkinter import ttk
from datetime import datetime

import cv2
from PIL import Image, ImageTk

import predictVideo


class App(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title('WatchOut! 你敢撞我試試看')
        self.iconbitmap('mainTkinterIcon.ico')

        self.photo = Image.open('bgi2.png')
        self.photo = ImageTk.PhotoImage(self.photo)
        self.background = None

        self.window_width = self.winfo_screenwidth()
        self.window_height = self.winfo_screenheight()

        self.width = 1600
        self.height = 900
        self.left = int((self.window_width - self.width) / 2)
        self.top = int((self.window_height - self.height) / 2)

        self.geometry(f'{self.width}x{self.height}+{self.left}+{self.top}')

        self.resizable(width=False, height=False)

        # self.configure(background='#ffef99')

        self.cap = None
        self.cap_count = 0
        self.iff_playing_video = False  # 控制 撥放原始影片暫停或停止

        # set elements
        self.frame_function_buttons = None
        self.button_upload_video = None
        self.button_start_to_detect = None
        self.button_save_to_results = None
        self.button_pause = None
        self.radio_button_class_a = None
        self.radio_button_class_b = None
        self.radio_variable = tk.IntVar()

        self.frame_play_video = None
        self.label_text_play_video = None
        self.label_play_video = None

        self.frame_detect_results = None
        self.label_text_detect_results = None
        self.text_detect_results = None

        self.file_path = None

        self.set_elements()

        self.modelY = None  # object detection model (YOLOv8)
        self.modelD = None  # depth estimation model (AdaBins)
        self.flag_can_predict = False  # 控制 撥放原始影片 or 進行預測
        self.iff_keep_predict = True  # 控制 預測暫停或停止
        self.predict_thread = threading.Thread(target=self.predict_video_frame)
        self.predict_thread.daemon = True
        self.predict_thread.start()

        self.mainloop()

    @staticmethod
    def button_on_hover(button, foreground_hover, background_hover, foreground_leave, background_leave):
        button.bind(
            '<Enter>',
            func=lambda e: button.config(
                foreground=foreground_hover,
                background=background_hover
            )
        )

        button.bind(
            '<Leave>',
            func=lambda e: button.config(
                foreground=foreground_leave,
                background=background_leave
            )
        )

    @staticmethod
    def button_style(button):
        button['font'] = font.Font(family='微軟正黑體', size=24, weight='bold')
        button['foreground'] = 'black',
        button['background'] = '#FFBFD5',
        button['activeforeground'] = 'white',
        button['activebackground'] = '#E60553',
        button['cursor'] = 'hand2',
        button['relief'] = 'solid',
        button['width'] = 10,
        button['height'] = 2

    def upload_video(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mkv")])
        if self.file_path:
            self.flag_can_predict = False
            self.iff_keep_predict = False  # 暫停預測
            self.iff_playing_video = True  # 開始撥放
            self.cap = cv2.VideoCapture(self.file_path)
            self.play_video()

    def play_video(self):
        if self.iff_playing_video:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (1080, 720))
                image = ImageTk.PhotoImage(Image.fromarray(frame))
                self.label_play_video.config(image=image)
                self.label_play_video.image = image
                self.update()
            else:  # if video end than play again from the beginning
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # reset cap
        self.after(33, self.play_video)

    def start_to_detect(self):
        print('Start to initialize')
        self.iff_playing_video = False  # 暫停播放
        self.text_detect_results['state'] = tk.NORMAL  # enable text area
        self.text_detect_results.delete(1.0, tk.END)  # reset text area
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # reset cap
        self.cap_count = 0  # reset cap count
        self.modelY, self.modelD = predictVideo.initialize()  # init ai model
        self.flag_can_predict = True  # enable to predict
        self.iff_keep_predict = True  # start to predict
        print('Initialize finish')

    def predict_video_frame(self):
        while True:
            if self.flag_can_predict and self.iff_keep_predict:
                ret, frame = self.cap.read()
                if ret:
                    # predict frame and update window
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # camera_setting = self.radio_variable.get()
                    result_v, result_t = predictVideo.main(frame, self.modelY, self.modelD)
                    result_v = cv2.resize(result_v, (1080, 720))
                    new_image = ImageTk.PhotoImage(Image.fromarray(result_v))
                    self.label_play_video.config(image=new_image)
                    self.label_play_video.image = new_image
                    self.update()

                    # print text info of result
                    if len(result_t) <= 0:
                        print(f"{self.cap_count}=>此幀無異常")
                    else:
                        self.text_detect_results.insert(tk.END, f"\n第{int(self.cap_count):03}幀 =>\n")
                        for text in result_t:
                            self.text_detect_results.insert(tk.END, f"{text}\n")
                    self.text_detect_results.see(tk.END)
                    self.cap_count += 1
                else:
                    print('Predict End')
                    self.iff_keep_predict = False
                    self.flag_can_predict = False
                    self.text_detect_results['state'] = tk.DISABLED
            else:
                time.sleep(0.3)  # Waiting a little longer to save my CPU from getting killed

    def save_to_results(self):
        current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_result = self.text_detect_results.get(1.0, tk.END)

        file_path = filedialog.asksaveasfilename(
            initialfile=f"{current_datetime}.txt",
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")])

        if file_path:
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(save_result)
            print(f"文件已保存到: {file_path}")
            messagebox.showinfo("儲存成功", f"文件已保存到: {file_path}")
        else:  # user cancel to save file
            print("用户取消了保存操作")
            messagebox.showwarning("儲存失敗", '儲存未完成')

    def pause(self):
        if self.flag_can_predict is False:
            if self.iff_playing_video is False:
                self.iff_playing_video = True
            elif self.iff_playing_video is True:
                self.iff_playing_video = False
        elif self.flag_can_predict is True:
            if self.iff_keep_predict is False:
                self.iff_keep_predict = True
            else:
                self.iff_keep_predict = False

    def set_elements(self):
        self.background = tk.Label(
            master=self,
            image=self.photo
        )
        self.background.place(x=0, y=0)

        self.frame_function_buttons = tk.Frame(
            master=self,
            background=''
        )
        self.frame_function_buttons.pack(padx=20, pady=20)

        self.button_upload_video = tk.Button(
            master=self.frame_function_buttons,
            text='上 傳 影 片',
            command=self.upload_video
        )
        self.button_style(button=self.button_upload_video)
        self.button_on_hover(
            button=self.button_upload_video,
            foreground_hover='white',
            background_hover='#E60553',
            foreground_leave='black',
            background_leave='#FFBFD5'
        )
        self.button_upload_video.grid(row=0, column=0, padx=40, pady=40)

        self.button_start_to_detect = tk.Button(
            master=self.frame_function_buttons,
            text='開 始 偵 測',
            command=self.start_to_detect
        )
        self.button_style(button=self.button_start_to_detect)
        self.button_on_hover(
            button=self.button_start_to_detect,
            foreground_hover='white',
            background_hover='#E60553',
            foreground_leave='black',
            background_leave='#FFBFD5'
        )
        self.button_start_to_detect.grid(row=0, column=1, padx=40, pady=40)

        self.button_save_to_results = tk.Button(
            master=self.frame_function_buttons,
            text='儲 存 結 果',
            command=self.save_to_results
        )
        self.button_style(button=self.button_save_to_results)
        self.button_on_hover(
            button=self.button_save_to_results,
            foreground_hover='white',
            background_hover='#E60553',
            foreground_leave='black',
            background_leave='#FFBFD5'
        )
        self.button_save_to_results.grid(row=0, column=2, padx=40, pady=40)

        self.button_pause = tk.Button(
            master=self.frame_function_buttons,
            text='暫停/繼續',
            command=self.pause
        )
        self.button_style(button=self.button_pause)
        self.button_on_hover(
            button=self.button_pause,
            foreground_hover='white',
            background_hover='#E60553',
            foreground_leave='black',
            background_leave='#FFBFD5'
        )
        self.button_pause.grid(row=0, column=3, padx=40, pady=40)

        self.radio_button_class_a = tk.Radiobutton(
            master=self.frame_function_buttons,
            variable=self.radio_variable,
            text='A',
            value=1
        )
        # self.radio_button_class_a.grid(row=0, column=4, padx=40, pady=40)
        self.radio_button_class_a.select()

        self.radio_button_class_b = tk.Radiobutton(
            master=self.frame_function_buttons,
            variable=self.radio_variable,
            text='B',
            value=2
        )
        # self.radio_button_class_b.grid(row=0, column=5, padx=40, pady=40)

        self.frame_play_video = tk.Frame(
            master=self,
            width=1080,
            height=720
        )
        self.frame_play_video.pack_propagate(False)
        self.frame_play_video.pack(side='left', padx=20, pady=20)

        self.label_text_play_video = tk.Label(
            master=self.frame_play_video,
            text='影 片 播 放',
            font=('微軟正黑體', 20, 'bold'),
            width=1080,
            height=1,
            background='#e1dddd'  # TODO: change color
        )
        self.label_text_play_video.pack(side='top')

        self.label_play_video = tk.Label(
            master=self.frame_play_video,
            width=1080,
            height=720,
            background='LightGray',
            relief='solid',
            borderwidth=2
        )
        self.label_play_video.pack(side='bottom')

        self.frame_detect_results = tk.Frame(
            master=self,
            width=480,
            height=720
        )
        self.frame_detect_results.pack_propagate(False)
        self.frame_detect_results.pack(side='right', padx=20, pady=20)

        self.label_text_detect_results = tk.Label(
            master=self.frame_detect_results,
            text='偵 測  結果',
            font=('微軟正黑體', 20, 'bold'),
            width=250,
            height=1,
            background='#ffeea6'  # TODO: change color
        )
        self.label_text_detect_results.pack(side='top')

        self.text_detect_results = tk.Text(
            master=self.frame_detect_results,
            font=('微軟正黑體', 14),
            background='white',
            foreground='black',
            width=250,
            height=720,
            cursor='xterm',
            relief='solid',
            borderwidth=2,
        )
        self.text_detect_results.pack()
        self.text_detect_results['state'] = tk.DISABLED


if __name__ == '__main__':
    app = App()
