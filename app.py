import streamlit as st
from analise import (
    correct_text, calculate_difference, classify_text_quality,
    predict_fake_news_probability, load_data_and_train_model, create_gauge_chart
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
            # Load data and train model
            tfidf_vec, xtrain_tfidf, y_train = load_data_and_train_model()

            # Select the training model
            if model_choice == "Logistic Regression":
                from sklearn.linear_model import LogisticRegression
                model = LogisticRegression()
            elif model_choice == "Random Forest":
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier()
            elif model_choice == "SVM":
                from sklearn.svm import SVC
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
            )

            # Display results
            st.subheader("Results:")

            # Create two columns for the gauges
            col1, col2 = st.columns(2)

            # Gauge for text quality (in the first column)
            with col1:
                text_quality_color = get_text_quality_color(quality_classification)
                create_gauge_chart(
                    value=difference_percentage,
                    title=f"Text Quality: {quality_classification}",
                    low_color="#1B8720",  # Green 
                    mid_color="#FF9400",  # Orange
                    high_color="#FF1708",  # Red
                    low_range=0.29,  # Low range threshold
                    mid_range=0.69,  # Mid range threshold
                    theme="White"
                )

            # Gauge for fake news probability (in the second column)
            with col2:
                fake_news_color = get_fake_news_color(fake_news_probability)
                create_gauge_chart(
                    value=fake_news_probability,
                    title=f"Fake News Probability",
                    low_color="#1B8720",  # Green
                    mid_color="#FF9400",  # Orange
                    high_color="#FF1708",  # Red
                    low_range=0.29,  # Low range threshold
                    mid_range=0.69,  # Mid range threshold
                    theme="White"
                )

# Run the Streamlit app
if __name__ == "__main__":
    main()