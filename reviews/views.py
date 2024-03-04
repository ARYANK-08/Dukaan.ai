from django.shortcuts import render

# Create your views here.
from django.shortcuts import render
from django.http import HttpResponse
from urllib.parse import urlparse

def get_video(video_id):
    if not video_id:
        return {"error": "video_id is required"}

    comments = get_video_comments(video_id)
    predictions = predict_sentiments(comments)

    positive = predictions.count("Positive")
    negative = predictions.count("Negative")

    summary = {
        "positive": positive,
        "negative": negative,
        "num_comments": len(comments),
        "rating": (positive / len(comments)) * 100
    }

    return {"predictions": predictions, "comments": comments, "summary": summary}

def getvideo_id(value):
    """
    Examples:
    - http://youtu.be/SA2iWivDJiE
    - http://www.youtube.com/watch?v=_oPAwA_Udwc&feature=feedu
    - http://www.youtube.com/embed/SA2iWivDJiE
    - http://www.youtube.com/v/SA2iWivDJiE?version=3&amp;hl=en_US
    """
    query = urlparse(value)
    if query.hostname == 'youtu.be':
        return query.path[1:]
    if query.hostname in ('www.youtube.com', 'youtube.com'):
        if query.path == '/watch':
            p = urlparse(query.query)
            return str(p.path[2:]).split('&')[0]
        if query.path[:7] == '/embed/':
            return query.path.split('/')[2]
        if query.path[:3] == '/v/':
            return query.path.split('/')[2]
    # fail?
    return None

def reviews(request):
    summary = None
    comments = []
    
    if request.method == 'POST':
        print('hi')
        video_url = request.POST.get('video_url')
        video_id = getvideo_id(video_url)
        data = get_video(video_id)

        summary = data['summary']
        comments = list(zip(data['comments'], data['predictions']))
    
    return render(request, 'reviews/index.html', {'summary': summary, 'comments': comments})

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import pickle

VOCAB_SIZE = 20000
MAX_LEN = 250
MODEL_PATH = "yt_models\\sentiment_analysis_model.h5"

# Load the saved model
model = load_model(MODEL_PATH)

# Load the tokenizer
with open('yt_models\\tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


def encode_texts(text_list):
    encoded_texts = []
    for text in text_list:
        tokens = tf.keras.preprocessing.text.text_to_word_sequence(text)
        tokens = [tokenizer.word_index.get(word, 0) for word in tokens]
        encoded_texts.append(tokens)
    return pad_sequences(encoded_texts, maxlen=MAX_LEN, padding='post', value=VOCAB_SIZE-1)


def predict_sentiments(text_list):
    encoded_inputs = encode_texts(text_list)
    predictions = np.argmax(model.predict(encoded_inputs), axis=-1)
    sentiments = []
    for prediction in predictions:
        if prediction == 0:
            sentiments.append("Negative")
        elif prediction == 1:
            sentiments.append("Neutral")
        else:
            sentiments.append("Positive")
    return sentiments

import os
import googleapiclient.discovery
import googleapiclient.errors
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("API_KEY")

def get_comments(youtube, **kwargs):
    comments = []
    results = youtube.commentThreads().list(**kwargs).execute()

    while results:
        for item in results['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            comments.append(comment)

        # check if there are more comments
        if 'nextPageToken' in results:
            kwargs['pageToken'] = results['nextPageToken']
            results = youtube.commentThreads().list(**kwargs).execute()
        else:
            break

    return comments

def main(video_id, api_key):
    # Disable OAuthlib's HTTPs verification when running locally.
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

    youtube = googleapiclient.discovery.build(
        "youtube", "v3", developerKey=api_key)

    comments = get_comments(youtube, part="snippet", videoId=video_id, textFormat="plainText")
    return comments


def get_video_comments(video_id):
    return main(video_id, api_key)


