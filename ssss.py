#import speech_recognition as s_r
#print(s_r.__version__) # just to print the version not required
#r = s_r.Recognizer()
#my_mic = s_r.Microphone(device_index=1) #my device index is 1, you have to put your device index
#with my_mic as source:
  #  print("Say now!!!!")
  #  r.adjust_for_ambient_noise(source) #reduce noise
  #  audio = r.listen(source)#take voice input from the microphone
  #  text=r.recognize_google(audio)
#print(text) #to print voice into text
#from gtts import gTTS
#def speak(text):
  #  tts =gTTS(text=text, lang="en")
 ##   filename = "voice.wav"
   # tts.save(filename)
    #playsound.playsound(filename)
   # speak(text)

#import librosa

#fname = 'voice.wav'


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
