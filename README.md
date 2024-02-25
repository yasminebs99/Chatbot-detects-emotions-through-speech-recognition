<h3>Description:</h3>
<p>Build a chatbot that uses verbal responses to identify a person's emotions.
<p>•Ensured that the preprocessed dataset meets defined quality standards and is ready for training machine learning.</p>
<p>•Achieved a high accuracy of the model of emotion type extraction.</p>
<p>•Created a chatbot to convert audio into text and presenting results using our trained model.</p>
<h3>Needs:</h3>
<p> -OpenCV</p>
<p> -Python</p>
<p> -Putty</p>
<p> -WinSCP</p>
<p> -VNC viewer</p>
<p> -Numpy</p>
<p> -Raspberry Pi</p>
<h3>Speech Emotion Recognition model Building</h3>
We used a dataset of images downloaded from the internet via "Kaggle", comprising approximately 5600 audios classified into 7 classes.
Here is the download link: https://www.kaggle.com/datasets/dmitrybabko/speech-emotion-recognition-en/data
There are four steps in the deep learning process:
<h5>1-Define the model.</h5>
<h5>2-Compile the model by calling the compile() function.</h5>
<h5>3-Fit the model with the training dataset. Test the model on test data by calling the fit() function.</h5>
<h5>4-Save the model in the (tflite) format.</h5>
<h3>Chatbot model Building</h3>
<h5>1-Import the model into the chatbot code and make predictions</h5>
Import the "aud_ct.h5" which represent the file with h5 extention of the Speech recognition model in the code of the Chatbot
<h5>2-Create the chatbot interface</h5>
<img src="https://github.com/yasminebs99/Chatbot-detects-emotions-through-speech-recognition/assets/160682389/ce09fd1a-9f56-49ff-b312-be3b1929231f" />
<h5>3-Convert audio to text</h5>
<img src="https://github.com/yasminebs99/Chatbot-detects-emotions-through-speech-recognition/assets/160682389/b219c858-25fe-4039-937a-4b0c456638d5"/>
<h3>Results:</h3>
<img src="https://github.com/yasminebs99/Chatbot-detects-emotions-through-speech-recognition/assets/160682389/9f2a184d-2f68-4c3c-b77b-afe519926366"width="400" height="300"/>
<img src="https://github.com/yasminebs99/Chatbot-detects-emotions-through-speech-recognition/assets/160682389/bd450761-c61e-42bd-93bb-702c23133de0" width="400" height="300"/>
<img src="https://github.com/yasminebs99/Chatbot-detects-emotions-through-speech-recognition/assets/160682389/bc259622-6944-44e6-b344-16f1139e7c37" width="300" height="200"/>
<img src="https://github.com/yasminebs99/Chatbot-detects-emotions-through-speech-recognition/assets/160682389/43f0f9d5-4a6c-48f4-b564-f286561149fd" width="300" height="200"/>
