# Imports
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pathlib
import textwrap
import google.generativeai as genai

def clean_data():
    df = pd.read_csv('books_data.csv', sep=',')
    # print(df.head())

    # print(df.info())

    # deleting duplicate rows
    df = df.drop_duplicates(subset='Title')
    print(df.duplicated(subset='Title').sum())

    def clean_text_author(Author_name):
        result = str(Author_name).lower()
        result.replace(' ','')
        return result

    def clean_text_genre(Genre):
        res = str(Genre).lower()
        res.replace(' ','')
        res.replace(',','')
        return res

    df['Author_name'] = df['Author_name'].apply(clean_text_author)
    df['Genre'] = df['Genre'].apply(clean_text_genre)

    df['Title'] = df['Title'].str.lower()
    df['Summary'] = df['Summary'].str.lower()

    return df

def build_model(df):
    # combine all strings:
    df2 = df.drop(['Avg_rating','Summary'],axis=1)

    df2['data'] = df2[df2.columns[1:]].apply(
        lambda x: ' '.join(x.dropna().astype(str)),
        axis=1
    )

    vectorizer = CountVectorizer()
    vectorized = vectorizer.fit_transform(df2['data'])

    similarities = cosine_similarity(vectorized)
    df3 = pd.DataFrame(similarities, columns=df['Title'], index=df['Title']).reset_index()

    return df3

def recommend(user_input, df3):
    input_book = user_input
    recommendations = pd.DataFrame(df3.nlargest(6,input_book)['Title'])
    recommendations = recommendations[recommendations['Title']!=input_book]
    print(recommendations)
    return recommendations


# sum1 = df.loc[df['Title'] == input_book, 'Summary'].item()
# sum2 = df.loc[df['Title'] == recommendations.iloc[0].Title, 'Summary'].item()

# prompt = "Your role is to provide explanations of given book recommendations based on the summaries of two given books. The first summary is of the book chosen by the user. The second summary is of the book recommended by the recommender system. Write this explanation in 5 brief points of 25 to 30 words each. Here are the two summaries respectively: Summary 1: " , sum1 , " Summary 2: " , sum2

def generate_explanation(input_book, df, recommendations):
    exp = []
    for i in range(0,5):
        sum1 = df.loc[df['Title'] == input_book, 'Summary'].item()
        sum2 = df.loc[df['Title'] == recommendations.iloc[i].Title, 'Summary'].item()
        prompt = "Your role is to provide explanations of given book recommendations based on the summaries of two given books. The first summary is of the book chosen by the user. The second summary is of the book recommended by the recommender system. Write this explanation in about 100 words. Here are the two summaries respectively: Summary 1: " , sum1 , " Summary 2: " , sum2
        genai.configure(api_key='AIzaSyAIt74KgKkNasBWpIdHh1Nutcsah-KT_wk')

        model = genai.GenerativeModel('gemini-pro')

        response = model.generate_content(prompt)

        exp.append(response.text)
    return exp

# explanations = generate_explanation()

# print(explanations)

