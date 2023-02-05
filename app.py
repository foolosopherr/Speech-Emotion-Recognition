from keras.models import load_model
import streamlit as st
import plotly.express as px
import plotly.figure_factory as ff
import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import confusion_matrix
import os
from audio_recorder_streamlit import audio_recorder

st.header('Speech Emotion Recognition')

def get_csv():
    return pd.read_csv('real_paths.csv')

df = get_csv()

def waveplot(data, sr, emotion):
    fig = plt.figure(figsize=(10, 4))
    plt.title(emotion, size=20)
    librosa.display.waveshow(data, sr=sr)
    plt.show()
    return fig

def spectrogram(data, sr, emotion):
    x = librosa.stft(data)
    xdb = librosa.amplitude_to_db(abs(x))
    fig = plt.figure(figsize=(10, 4))
    plt.title(emotion, size=20)
    img = librosa.display.specshow(xdb, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar(img, format="%+2.f dB")
    return fig

tab1, tab2, tab3 = st.tabs(['Data EDA', 'Model description', 'Try the model yourself'])

with tab1:

    st.markdown('## Choose audio file')
    t1c1, t1c2 = st.columns(2)
    with t1c1:
        emotion_options = df['label'].unique()
        emotion = st.selectbox('Select emotion', emotion_options)
    with t1c2:
        number = st.number_input('Select Nth audio file (from 1 to 24)', 1, 24, 10) - 1
    
    path = df[df['label'] == emotion].iloc[number, 0]
    data, sampling_rate = librosa.load(path)
    plot1 = waveplot(data, sampling_rate, emotion)
    plot2 = spectrogram(data, sampling_rate, emotion)
    
    audio_file = open(path, 'rb')
    audio_bytes = audio_file.read()
    st.markdown('### Original audio file')
    st.audio(audio_bytes, format='audio/wav')
    
    st.markdown('## Waveplot')
    st.pyplot(plot1)
    st.markdown('## Spectrogram')
    st.pyplot(plot2)


model = load_model('my_model.h5')

def plot_train_val(train, val, name):
    x = [i for i in range(1, len(train)+1)]
    fig = px.line(x=x, y=[train, val], title=None,
    labels={'value': name, 'x': 'Epochs', 'variable': 'Set'})
    newnames = {'wide_variable_0': 'Train', 'wide_variable_1': 'Validation'}
    fig.for_each_trace(lambda t: t.update(name = newnames[t.name],
                                      legendgroup = newnames[t.name],
                                      hovertemplate = t.hovertemplate.replace(t.name, newnames[t.name])
                                     )
                  )
    st.markdown(f'## {name}')
    return fig

def plot_cm(y_true, y_pred):
    x = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'pleasant surprise', 'sad']
    y = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'pleasant surprise', 'sad']
    z = confusion_matrix(y_test, y_pred)
    z_text = [[str(y) for y in x] for x in z]
    fig = ff.create_annotated_heatmap(z, x=x, y=y, annotation_text=z_text, colorscale='Viridis')
    fig.add_annotation(dict(font=dict(color="white",size=20),
                        x=0.5,
                        y=-0.15,
                        showarrow=False,
                        text="Predicted value",
                        xref="paper",
                        yref="paper"))

    fig.add_annotation(dict(font=dict(color="white",size=20),
                            x=-0.35,
                            y=0.5,
                            showarrow=False,
                            text="True value",
                            textangle=-90,
                            xref="paper",
                            yref="paper"))

    fig.update_layout(margin=dict(t=50, l=200))

    fig['data'][0]['showscale'] = True
    return fig


with tab2:
    st.markdown('## Model Summary')
    model.summary(print_fn=lambda x: st.text(x))

    with open('trainHistoryDict', "rb") as file_pi:
        history = pickle.load(file_pi)
    with open('predandtest', "rb") as file_pi:
        predandtest = pickle.load(file_pi)

    acc_train = history['accuracy']
    acc_val = history['val_accuracy']
    loss_train = history['loss']
    loss_val = history['val_loss']

    y_test = predandtest['test']
    y_pred = predandtest['pred']

    st.plotly_chart(plot_train_val(acc_train, acc_val, 'Accuracy'))
    st.plotly_chart(plot_train_val(loss_train, loss_val, 'Loss'))


    st.markdown('## Confusion Matrix')
    st.plotly_chart(plot_cm(y_test, y_pred))

def save_file(sound_file):
    # save your sound file in the right folder by following the path
    with open(os.path.join('audio_files/', sound_file.name),'wb') as f:
         f.write(sound_file.getbuffer())
    return sound_file.name

def for_tab3(uploaded_file, kek=True):
    if kek:
        file_details = {'filename':uploaded_file.name, 'filetype':uploaded_file.type, 'filesize':uploaded_file.size}
        st.write(file_details)
        st.write('### Play audio')
        audio_bytes = uploaded_file.read()
        st.audio(audio_bytes, format='audio/wav')
        sound = uploaded_file.name
        save_file(uploaded_file)
        sound_name = f'audio_files/{sound}'
    else:
        sound_name = uploaded_file
    y, sr = librosa.load(sound_name, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=100).T, axis=0)
    X = [x for x in mfcc]
    X = np.array(X).reshape((1, 100, 1))
    model_pred = model.predict(X)[0]
    emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'pleasant surprise', 'sad']
    res = np.argmax(model_pred)
    res = emotions[res]
    if kek:
        st.write(f'## Emotion for your .wav file - {res}')
    else:
        model_pred *= 100
        st.write(f'## Results for your voice')
        voice_df = pd.DataFrame({'Emotion': emotions, 'Probability, %': model_pred})
        # voice_df['Probability, %'] = voice_df['Probability, %'].round(2)
        voice_df = voice_df.sort_values(by='Probability, %', ascending=False).reset_index(drop=True)
        st.dataframe(voice_df.style.format(formatter={'Probability, %': "{:.2f}"}))


with tab3:
    st.markdown('## Upload your audio')

    user_audio = audio_recorder('Click to record your voice', pause_threshold=3.0, energy_threshold=(1.0, 4.0), sample_rate=21_000)
    uploaded_file = st.file_uploader('## Upload .wav file', type=['wav'])
    if uploaded_file:
        for_tab3(uploaded_file)
    if user_audio:
        with open('audio_files/input.wav', 'wb') as f:
            f.write(user_audio)
        for_tab3('audio_files/input.wav', False)
    else:
        st.write('## I am waiting...')