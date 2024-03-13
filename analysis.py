from textblob import TextBlob
import pandas as pd
import streamlit as st
import cleantext

st.header('Sentiment Analysis - Shamitr Mardikar [21BCE0695]')
with st.expander('Analyze Text'):
    text = st.text_input('Enter Text Here : ')
    if text:
        blob = TextBlob(text)
        st.write('Polarity (Measure of Negativity or Positivity): ', round(blob.sentiment.polarity, 2))
        st.write('Subjectivity (Determines whether text is Subjective or Objective) : ', round(blob.sentiment.subjectivity,2))

    pre = st.text_input('Clean text : ')
    if pre:
        st.write(cleantext.clean(pre, clean_all = False, extra_spaces = True, stopwords = True, lowercase = True, numbers = True, punct = True))

with st.expander('Analyse Dataset'):
    upl = st.file_uploader('Upload File here')

    def score(x):
        blob1 = TextBlob(x)
        return blob1.sentiment.polarity
    
    def analyze(x):
        if x >= 0.5:
            return 'Positive'
        elif x <= -0.5:
            return 'Negative'
        else:
            return 'Neutral'
        
    if upl:
        df = pd.read_excel(upl)
        del df['Unnamed: 0']
        df['score'] = df['tweets'].apply(score)
        df['analysis'] = df['score'].apply(analyze)
        st.write(df.head(10))

        @st.cache
        def convert_df(df):
            return df.to_csv().encode('utf-8')
        
        CSV = convert_df(df)

        st.download_button(
            label = "Download all data as CSV",
            data = csv,
            file_name = 'sentiment.csv',
            mime = 'text/csv'
        )