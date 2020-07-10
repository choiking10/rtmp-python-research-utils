from ffpyplayer.player import MediaPlayer
import queue
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import threading
import numpy as np
import io
import time
import cv2
import datetime

# https://currentmillis.com/

q = queue.Queue()
#wins3 = "rtmp://143.248.55.86/live/wins3"
wins3 = "rtmp://143.248.55.86:1913/live/wins3"
wins2 = "rtmp://143.248.55.86:1911/live/wins2"
wins = "rtmp://143.248.55.86:31935/live/wins"
local_addr = "rtmp://127.0.0.1/live/wins"

def get_frame():
    player = MediaPlayer(wins3, ff_opts={'out_fmt': 'rgb24'})
    while 1:
        frame, val = player.get_frame()
        if val == 'eof':
            break
        elif frame is None:
            time.sleep(0.002)
        else:
            q.put(frame)

def play():
    flag = True
    bef_t = 0
    bef_show = datetime.datetime.now()
    frame_jitter = 33.333333333333333333
    while 1:
        if q.empty():
            time.sleep(0.001)
        else:
            frame = q.get()
            img, t = frame
            w, h = img.get_size()

            nparray = np.frombuffer(img.to_bytearray()[0], dtype=np.uint8)
            nparray = nparray.reshape(h, w, 3)
            nparray = cv2.cvtColor(nparray, cv2.COLOR_RGB2BGR)
            while datetime.datetime.now() - bef_show <= datetime.timedelta(milliseconds=frame_jitter):
                time.sleep(0.001)
            if datetime.datetime.now() - bef_show >= datetime.timedelta(milliseconds=100) or True:
                print(datetime.datetime.now().strftime("%H:%M:%S.%f"), int(q.qsize() * 33.333333333), (datetime.datetime.now() - bef_show).microseconds//1000, t)
            bef_show = datetime.datetime.now()
            cv2.imshow("test", nparray)
            # delay = 1 if flag else 30

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if bef_t - t >= 0.05:
                print("skip")
            if key == ord('t'):
                print("toggle to " + str(1))
                frame_jitter = 1 if flag else 33
                flag = not flag


t = threading.Thread(target=get_frame)
t.start()

t = threading.Thread(target=play)
t.start()
