import nltk.stem.porter
import gradio as gr
import string
import nltk
import joblib
nltk.download('punkt_tab')
nltk.download('punkt')
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer
import warnings
import pandas as pd

warnings.filterwarnings('ignore')

# --- Load Trained Model ---
# This function loads the pre-trained pipeline model from the 'model.pkl' file.
def load_model():
    try:
        model = joblib.load('model.pkl')
        return model
    except FileNotFoundError:
        print("Error: 'model.pkl' not found. Please ensure the model is trained and saved correctly.")
        return None
    
def predict_sentiment(text):
    model = load_model()
    input_data = pd.DataFrame({'Tweet Content': [text]})
    prediction = model.predict(input_data)
    
    # The prediction is returned as an array, so we extract the first element.
    return prediction[0]

def transform_text(text):
    
   
    return " ".join(y)
# --- Gradio User Interface ---
# This section sets up the web interface for the sentiment analysis tool.
demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(
        label="Enter Tweet Text",
        lines=5,
        placeholder="Write your tweet here..."
    ),
    outputs=gr.Textbox(
        label="Predicted Sentiment"
    ),
    title="Twitter Sentiment Analysis",
    description="Enter a tweet to analyze its sentiment. The model will classify it as Positive, Negative, Neutral, or Irrelevant.",
    examples=[
        ["I am so excited for the new game release!"],
        ["This is the worst customer service I have ever experienced."],
        ["The event is scheduled for tomorrow at 5 PM."],
    ]
)

# --- Launch the Application ---
# This makes the Gradio interface accessible when the script is run.
if __name__ == "__main__":
    demo.launch()
