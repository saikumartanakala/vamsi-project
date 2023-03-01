

# importing relevant python packages
import streamlit as st
import pandas as pd
import numpy as np
import pickle

from PIL import Image
# preprocessing
import re
import string
import nltk
#
# nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
# modeling
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text

# sentiment analysis
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.sentiment.vader import SentimentIntensityAnalyzer as VS

business_context = st.container()
data_desc = st.container()
performance = st.container()
tweet_input = st.container()
model_results = st.container()
sentiment_analysis1 = st.container()



with business_context:
    st.header('The Problem of Content Moderation')
    st.write("""
    
    **Human content moderation exploits people by consistently traumatizing and underpaying them.** In 2019, an [article](https://www.theverge.com/2019/6/19/18681845/facebook-moderator-interviews-video-trauma-ptsd-cognizant-tampa) on The Verge exposed the extensive list of horrific working conditions that employees faced at Cognizant, which was Facebook’s primary moderation contractor. Unfortunately, **every major tech company**, including **Twitter**, uses human moderators to some extent, both domestically and overseas.
    
    Hate speech is defined as **abusive or threatening speech that expresses prejudice against a particular group, especially on the basis of race, religion or sexual orientation.**  Usually, the difference between hate speech and offensive language comes down to subtle context or diction.
    
    """)

# with data_desc:
#     understanding, venn = st.columns(2)
#     with understanding:
#         st.text('')
#         st.write("""
#         The **data** for this project was sourced from a Cornell University [study](https://github.com/t-davidson/hate-speech-and-offensive-language) titled *Automated Hate Speech Detection and the Problem of Offensive Language*.
        
#         The `.csv` file has **24,802 rows** where **6% of the tweets were labeled as "Hate Speech".**

#         Each tweet's label was voted on by crowdsource and determined by majority rules.
#         """)
#     with venn:
#         st.image(Image.open('visualizations/word_venn.png'), width = 400)

# with performance:
#     description, conf_matrix = st.columns(2)
#     with description:
#         st.header('Final Model Performance')
#         st.write("""
#         These scores are indicative of the two major roadblocks of the project:
#         - The massive class imbalance of the dataset
#         - The model's inability to identify what constitutes as hate speech
#         """)
#     with conf_matrix:
#         st.image(Image.open('visualizations/normalized_log_reg_countvec_matrix.png'), width = 400)

with tweet_input:
    st.header('Is Your Tweet Considered Hate Speech?')
    st.write("""*Please note that this prediction is based on how the model was trained, so it may not be an accurate representation.*""")
    # user input here
    user_text = st.text_input('Enter Tweet', max_chars=280) # setting input as user_text

with model_results:    
    st.subheader('Prediction:')
    if user_text:
    # processing user_text
        # removing punctuation
        user_text = re.sub('[%s]' % re.escape(string.punctuation), '', user_text)
        # tokenizing
        stop_words = stopwords.words('english')
        tokens = nltk.word_tokenize(user_text)
        # removing stop words
        stopwords_removed = [token.lower() for token in tokens if token.lower() not in stop_words]
        # taking root word
        analyzer = SentimentIntensityAnalyzer() 
        # the object outputs the scores into a dict
        sentiment_dict = analyzer.polarity_scores(user_text)

        lemmatizer = WordNetLemmatizer() 
        lemmatized_output = ""
        for word in stopwords_removed:
            lemmatized_output+=lemmatizer.lemmatize(word)+" "

        X_test = lemmatized_output
        # instantiating count vectorizor
        tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2),max_df=0.75, min_df=5, max_features=10000)
        X_train = pickle.load(open('pickle/processed_text.pkl', 'rb'))
        
        X_train_count = tfidf_vectorizer.fit_transform(X_train)
        data = tfidf_vectorizer.transform([X_test])
        
        sentiment_analyzer = VS()



    
        def count_tags(tweet_c):
            space_pattern = '\s+'
            giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|''[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
            mention_regex = '@[\w\-]+'
            hashtag_regex = '#[\w\-]+'
            parsed_text = re.sub(space_pattern, ' ', tweet_c)
            parsed_text = re.sub(giant_url_regex, 'URLHERE', parsed_text)
            parsed_text = re.sub(mention_regex, 'MENTIONHERE', parsed_text)
            parsed_text = re.sub(hashtag_regex, 'HASHTAGHERE', parsed_text)
            return(parsed_text.count('URLHERE'),parsed_text.count('MENTIONHERE'),parsed_text.count('HASHTAGHERE'))

        def sentiment_analysis(tweet):   
            sentiment = sentiment_analyzer.polarity_scores(tweet)    
            twitter_objs = count_tags(tweet)
            features = [sentiment['neg'], sentiment['pos'], sentiment['neu'], sentiment['compound'],twitter_objs[0], twitter_objs[1],twitter_objs[2]]
    #features = pandas.DataFrame(features)
            return features

        def sentiment_analysis_array(t):
            features=[]
    
            features.append(sentiment_analysis(t))
            return np.array(features)

        final_features1 = sentiment_analysis_array(X_test)
#final_features
        print(final_features1)

        
        
# F2-Conctaenation of tf-idf scores and sentiment scores
        tfidf_a1= data.toarray()
        modelling_features1 = np.concatenate([tfidf_a1,final_features1],axis=1)
        #modelling_features1.shape
        model = pickle.load(open('pickle/lg_model.pkl', 'rb'))
        a = model.predict(modelling_features1)
        print(a)



        
        # loading in model
        
        
        # apply model to make predictions
        
        
        if a == 0 or sentiment_dict['compound']>= 0.05:
            st.subheader('**NOT Hate Speech**')
        elif sentiment_dict['compound']<= -0.05:
            st.subheader('**Hate Speech**')
        else:
            st.subheader('**NOT Hate Speech**')
        st.text('')

with sentiment_analysis1:
    if user_text:
        st.header('Sentiment Analysis with VADER')
        
        # explaining VADER
        st.write("""*VADER is a lexicon designed for scoring social media. More information can be found [here](https://github.com/cjhutto/vaderSentiment).*""")
        # spacer
        st.text('')
    
        # instantiating VADER sentiment analyzer
        analyzer = SentimentIntensityAnalyzer() 
        # the object outputs the scores into a dict
        sentiment_dict = analyzer.polarity_scores(user_text) 
        if sentiment_dict['compound'] >= 0.05 : 
            category = ("**Positive ✅**")
        elif sentiment_dict['compound'] <= - 0.05 : 
            category = ("**Negative 🚫**") 
        else : 
            category = ("**Neutral ☑️**")

        # score breakdown section with columns
        breakdown, graph = st.columns(2)
        with breakdown:
            # printing category
            st.write("Your Tweet is rated as", category) 
            # printing overall compound score
            st.write("**Compound Score**: ", sentiment_dict['compound'])
            # printing overall compound score
            st.write("**Polarity Breakdown:**") 
            st.write(sentiment_dict['neg']*100, "% Negative") 
            st.write(sentiment_dict['neu']*100, "% Neutral") 
            st.write(sentiment_dict['pos']*100, "% Positive") 
        with graph:
            sentiment_graph = pd.DataFrame.from_dict(sentiment_dict, orient='index').drop(['compound'])
            st.bar_chart(sentiment_graph) 

