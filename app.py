import streamlit as st
from recos_model import clean_data, build_model, recommend, generate_explanation

def main():
    st.title("Book Recommendation System")

    # Load and clean data
    df = clean_data()

    # Build recommendation model
    df3 = build_model(df)

    # Book selection
    selected_book = st.selectbox("Select a book:", df['Title'])

    # Display book summary
    st.subheader("Book Summary:")
    st.write(df.loc[df['Title'] == selected_book, 'Summary'].item())

    # Get recommendations
    recommendations = recommend(selected_book, df3)

    # Display recommendations
    st.subheader("Book Recommendations:")
    st.write(recommendations)

    # User explanation for the recommendation
    selected_recommendation = st.selectbox("Select a recommended book to see why it's recommended:", recommendations['Title'])

    # Display explanation
    st.subheader("Why is this book recommended?")
    recommendations_list = recommendations[['Title']].copy()
    recommendations_list.index = [0, 1, 2, 3, 4]
    explanation_index = recommendations_list[recommendations_list['Title']==selected_recommendation].index.values.astype(int)[0]
    explanation = generate_explanation(selected_book, df, recommendations)
    st.write(explanation[explanation_index])
    # if not explanation_index.empty:
    #     explanation = generate_explanation(selected_book, df, recommendations)
    #     st.write(explanation[explanation_index])
    # else:
    #     st.write("Please select a book to see the explanation")

if __name__ == "__main__":
    main()