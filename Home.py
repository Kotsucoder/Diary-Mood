import streamlit as st
import glob
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import plotly.express as px
from pathlib import Path

nltk.download('vader_lexicon') # For some reason this is required?


filepaths = sorted(glob.glob("diary/*.txt"))
diary = []
for path in filepaths:
    with open(path, "r") as entry:
        diary.append(entry.read())


diarytone = []
analyzer = SentimentIntensityAnalyzer()
for entry in diary:
    diarytone.append(analyzer.polarity_scores(entry))


positive_scores = []
for tone in diarytone:
    positive_scores.append(tone["pos"])


negative_scores = []
for tone in diarytone:
    negative_scores.append(tone["neg"])


dates = [Path(name).stem for name in filepaths]
labels = {"x": "Date", "y": "Score"}


st.title("Diary Tone")

st.header("Positivity")
positive_graph = px.line(x=dates, y=positive_scores, labels=labels)
st.plotly_chart(positive_graph)

st.header("Negativity")
negative_graph = px.line(x=dates, y=negative_scores, labels=labels)
st.plotly_chart(negative_graph)