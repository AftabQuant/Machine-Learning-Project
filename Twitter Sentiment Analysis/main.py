import gradio as gr
import joblib
import pandas as pd
import re
import warnings

warnings.filterwarnings("ignore")

try:
    model_pipeline = joblib.load('model.pkl')
    print("Model 'model.pkl' loaded successfully.")
except FileNotFoundError:
    print("ERROR: 'model.pkl' not found. Please run your training script first to generate the model file.")
    model_pipeline = None

def predict_sentiment(text):
    # Check if the model is loaded and if there is text to analyze
    if model_pipeline is None:
        return {"Error": 1.0, "Model not loaded. Please check console for details.": 0.0}
    if not isinstance(text, str) or not text.strip():
        return {"Positive": 0.0, "Negative": 0.0, "Neutral": 0.0}
    
    # Create a DataFrame from the raw text
    input_df = pd.DataFrame({'Tweet Content': [text]})

    # Perform the exact same preprocessing steps as in the training script
    input_df["lower"] = input_df["Tweet Content"].astype(str).str.lower()
    input_df["lower"] = [str(data) for data in input_df.lower]
    input_df["lower"] = input_df.lower.apply(lambda x: re.sub('[^A-Za-z0-9 ]+', ' ', x))

    # Use the pipeline to predict probabilities. The pipeline handles both the
    # CountVectorizer transformation and the Logistic Regression prediction.
    try:
        pred_probas = model_pipeline.predict_proba(input_df)[0]
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return {"Error": 1.0, "Prediction failed": 0.0}

    class_labels = model_pipeline.named_steps['clf'].classes_

    # Create a dictionary mapping each sentiment label to its predicted probability.
    output_dict = {label: prob for label, prob in zip(class_labels, pred_probas)}

    return output_dict

title = "Twitter Sentiment Analysis"
description = """
Enter a tweet or any text to analyze its sentiment.
The model will classify the text as **Positive**, **Negative**, **Neutral**, & **irrelevant** based on a Logistic Regression model trained on a Twitter dataset.
"""

# Provide some examples for the user to try
examples = [
    ["I am so excited for the new Borderlands movie!"],
    ["I'm having a really bad day, everything is going wrong."],
    ["I'm just watching the game on TV."],
    ["Playstation 5 is the best console ever made."],
    ["I can't believe they cancelled my favorite show."],
    ["This is the worst product I have ever bought. So disappointed."],
]

# Create the Gradio interface
iface = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(
        lines=5,
        placeholder="Enter your tweet or text here...",
        label="Text for Sentiment Analysis"
    ),
    outputs=gr.Label(
        num_top_classes=4,
        label="Sentiment Prediction"
    ),
    title=title,
    description=description,
    examples=examples,
    # allow_flagging="never" 
)

# --- 4. LAUNCH THE APP ---
if __name__ == "__main__":
    iface.launch()