# app.py
import streamlit as st
import pandas as pd
from analise import (
    clean_text, correct_text, calculate_difference, classify_text_quality,
    predict_fake_news_probability, create_gauge_chart, train_test_split,
    TfidfVectorizer, LogisticRegression, RandomForestClassifier, SVC
)

# Function to define colors based on text quality
def get_text_quality_color(quality):
    if quality == "Poor":
        return "#FF1708"  # Red for poorly written
    elif quality == "Fair":
        return "#FF9400"  # Orange for neutral
    else:
        return "#1B8720"  # Green for well-written

# Function to define colors based on fake news probability
def get_fake_news_color(probability):
    if probability > 65:
        return "#FF1708"  # Red for high probability
    elif 30 <= probability <= 65:
        return "#FF9400"  # Orange for medium probability
    else:
        return "#1B8720"  # Green for low probability

# Function to highlight differences between texts
def highlight_differences(original, corrected):
    original_words = original.split()
    corrected_words = corrected.split()
    highlighted = []
    
    for orig, corr in zip(original_words, corrected_words):
        if orig != corr:
            highlighted.append(f"<span style='color:red'>{orig}</span> → <span style='color:green'>{corr}</span>")
        else:
            highlighted.append(orig)
    
    return " ".join(highlighted)

# Streamlit App
def main():
    st.title("Fake News Detector with Spell Checking")
    st.write("Enter the text you want to analyze and choose the machine learning model.")

    # Text input field
    text_to_analyze = st.text_area("Enter the text here:")

    # Radio button to select the model
    model_choice = st.radio(
        "Choose the model:",
        ["Logistic Regression", "Random Forest", "SVM"]
    )

    # Button to start the analysis
    if st.button("Analyze"):
        if text_to_analyze.strip() == "":
            st.warning("Please enter some text to analyze.")
        else:
            # Load data
            df = pd.read_csv('db/out_df.csv')

            # Ensure the 'clean_text' column exists
            if 'clean_text' not in df.columns:
                # Create the 'clean_text' column if it doesn't exist
                df['clean_text'] = df['Textos'].apply(clean_text).str.replace('bn bn ', '').str.replace(' bn ', '')

            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(df['clean_text'], df['check_id'], test_size=0.2, random_state=42)

            # Vectorize the training and testing data
            tfidf_vec = TfidfVectorizer()
            xtrain_tfidf = tfidf_vec.fit_transform(X_train)
            xtest_tfidf = tfidf_vec.transform(X_test)

            # Select the training model
            if model_choice == "Logistic Regression":
                model = LogisticRegression()
            elif model_choice == "Random Forest":
                model = RandomForestClassifier()
            elif model_choice == "SVM":
                model = SVC(probability=True)

            # Train the model
            model.fit(xtrain_tfidf, y_train)

            # Spell checking and correction
            corrected_text = correct_text(text_to_analyze)
            difference_percentage = calculate_difference(text_to_analyze, corrected_text)
            quality_classification = classify_text_quality(difference_percentage)

            # Predict the probability with text quality weighting
            fake_news_probability = predict_fake_news_probability(
                corrected_text, model, tfidf_vec, difference_percentage
            ) * 100  # Convert to percentage

            # Display results
            st.subheader("Results:")

            # Show debug information
            with st.expander("Debug Information"):
                st.subheader("Original Text")
                st.write(text_to_analyze)
                
                st.subheader("Corrected Text")
                st.write(corrected_text)
                
                st.subheader("Differences Highlighted")
                st.markdown(highlight_differences(text_to_analyze, corrected_text), unsafe_allow_html=True)
                
                st.subheader("Text Quality Metrics")
                st.write(f"Difference Percentage: {difference_percentage:.2f}%")
                st.write(f"Quality Classification: {quality_classification}")

            # Create two columns for the gauges
            col1, col2 = st.columns(2)

            # Gauge for text quality (in the first column)
            with col1:
                text_quality_color = get_text_quality_color(quality_classification)
                st.write("### Text Quality")
                create_gauge_chart(
                    value=difference_percentage / 100,  # Scale to 0-1 for the gauge
                    title=f"Text Quality: {quality_classification}",
                    low_color="#1B8720",  # Green for good quality
                    mid_color="#FF9400",  # Orange for fair quality
                    high_color="#FF1708",  # Red for poor quality
                    low_range=0.29,  # Low range threshold
                    mid_range=0.69,  # Mid range threshold
                    theme="White"
                )

            # Gauge for fake news probability (in the second column)
            with col2:
                fake_news_color = get_fake_news_color(fake_news_probability)
                st.write("### Fake News Probability")
                create_gauge_chart(
                    value=fake_news_probability / 100,  # Scale to 0-1 for the gauge
                    title=f"Fake News Probability: {fake_news_probability:.1f}%",
                    low_color="#1B8720",  # Green for low probability
                    mid_color="#FF9400",  # Orange for medium probability
                    high_color="#FF1708",  # Red for high probability
                    low_range=0.29,  # Low range threshold
                    mid_range=0.69,  # Mid range threshold
                    theme="White"
                )

# Run the Streamlit app
if __name__ == "__main__":
    main()