from tkinter import *
from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
import os
import pyttsx3
import speech_recognition 
import threading 
import tensorflow as tf








import pyaudio
import wave
import matplotlib.pyplot as plt
import numpy as np

FRAMES_PER_BUFFER = 3200
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

pa = pyaudio.PyAudio()

stream = pa.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    frames_per_buffer=FRAMES_PER_BUFFER
)

print('start recording')

seconds = 3
frames = []
second_tracking = 0
second_count = 0
for i in range(0, int(RATE/FRAMES_PER_BUFFER*seconds)):
    data = stream.read(FRAMES_PER_BUFFER)
    frames.append(data)
    second_tracking += 1
    if second_tracking == RATE/FRAMES_PER_BUFFER:
        second_count += 1
        second_tracking = 0
        print(f'Time Left: {seconds - second_count} seconds')


stream.stop_stream()
stream.close()
pa.terminate()

obj = wave.open('lemaster_tech.wav', 'wb')
obj.setnchannels(CHANNELS)
obj.setsampwidth(pa.get_sample_size(FORMAT))
obj.setframerate(RATE)
obj.writeframes(b''.join(frames))
obj.close()


file = wave.open('lemaster_tech.wav', 'rb')
sample_freq = file.getframerate()
frames = file.getnframes()
signal_wave = file.readframes(-1)

file.close()

time = frames / sample_freq

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# if one channel use int16, if 2 use int32
audio_array = np.frombuffer(signal_wave, dtype=np.float32)
audio_array.reshape(600,40,1)
audio_array.shape =(600,40,1)
times = np.linspace(0, time, num=frames)

"""def extract_mfcc(filename):
    y,sr = librosa.load(filename, duration=3,offset=0.5)#we have audios with differant duration so we will fixe it on 3
    mfcc = np.mean(librosa.feature.mfcc(y=y,sr=sr,n_mfcc=40).T,axis=0)
    return mfcc
extract_mfcc(obj)
x_mfcc = np.array(x_mfcc)"""
print(audio_array)
print(len(audio_array))
import tensorflow as tf
tf.convert_to_tensor(audio_array) 
print(audio_array.shape)
model = tf.keras.models.load_model("aud_ct.h5")
y_pred=model.predict(audio_array)
print(y_pred)
print (len(y_pred[0]))


prediction=y_pred[:]
for index, probability in enumerate(prediction):
    if probability[0] > 0.5:
        k=('%.2f' % (probability[0]*100) + '% angry')
    elif probability[1] > 0.5:
        k=('%.2f' % ((probability[1])*100) + '% disgust')
    elif probability[2] > 0.5:
        k=('%.2f' % ((probability[2])*100) + '% fear')
    elif probability[3] > 0.5:
        k=('%.2f' % ((probability[3])*100) + '% happy')
    elif probability[4] > 0.5:
        k=('%.2f' % ((probability[4])*100) + '% neutral')
    elif probability[5] > 0.5:
        k=('%.2f' % ((probability[5])*100) + '% ps')
    elif probability[6] > 0.5:
        k=('%.2f' % ((probability[6])*100) + '% sad')
#plt.show()
print(k)
print('done')




################################################################


bot=ChatBot('Bot')
trainer=ListTrainer(bot)

for files in os.listdir('data/english/'):

    data=open('data/english/'+files,'r',encoding='utf-8').readlines()

    trainer.train(data)

def botReply():
    question=questionField.get()
    question=question.capitalize()
    answer=bot.get_response(question)
    textarea.insert(END,'You: '+question+'\n\n')
    textarea.insert(END,'Bot: '+str(answer)+'you seem '+str(k)+'\n\n')
    pyttsx3.speak(answer)
    questionField.delete(0,END)

def audioToText():
    while True:
        sr=speech_recognition.Recognizer()
        try:
            with speech_recognition.Microphone() as m:# hia eli chtod5l mel micro
                sr.adjust_for_ambient_noise(m,duration=0)
                audio=sr.listen(m)
                query=sr.recognize_google(audio)
                query=query.capitalize()
                questionField.delete(0,END)
                questionField.insert(0,query)
                botReply()
        except Exception as e:
            print(e)




root=Tk()

root.geometry('500x570+100+30')
root.title('MyChatBot ')
root.config(bg='black')

logoPic=PhotoImage(file='pic.png')

logoPicLabel=Label(root,image=logoPic,bg='black')
logoPicLabel.pack(pady=5)

centerFrame=Frame(root)
centerFrame.pack()

scrollbar=Scrollbar(centerFrame)
scrollbar.pack(side=RIGHT)

textarea=Text(centerFrame,font=('times new roman',20,'bold'),height=10,yscrollcommand=scrollbar.set
              ,wrap='word')
textarea.pack(side=LEFT)
scrollbar.config(command=textarea.yview)

questionField=Entry(root,font=('verdana',20,'bold'))
questionField.pack(pady=15,fill=X)

askPic=PhotoImage(file='ask.png')


askButton=Button(root,image=askPic,command=botReply)
askButton.pack()

def click(event):
    askButton.invoke()


root.bind('<Return>',click)
thread=threading.Thread(target=audioToText)#which fonction you want to put in this thread
thread.setDaemon(True)
thread.start()
root.mainloop()
