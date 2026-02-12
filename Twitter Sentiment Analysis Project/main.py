import gradio as gr
import joblib
import numpy as np
import warnings

warnings.filterwarnings("ignore")

try:
    model = joblib.load("model_new.pkl")
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None


def analyze_sentiment(text):

    if model is None:
        return "❌ Model not loaded", None, None

    if not text.strip():
        return "⚠️ Please enter some text", None, None

    # Predict
    prediction = model.predict([text])[0]
    probabilities = model.predict_proba([text])[0]

    confidence = round(np.max(probabilities) * 100, 2)

    # Format label nicely
    result_text = f"### 🎯 Prediction: **{prediction}**"
    confidence_text = f"Confidence Score: **{confidence}%**"

    # Create probability dictionary
    prob_dict = {
        label: float(prob)
        for label, prob in zip(model.classes_, probabilities)
    }

    return result_text, confidence_text, prob_dict


with gr.Blocks(theme=gr.themes.Soft()) as demo:

    gr.Markdown(
        """
        # 🚀 Twitter Sentiment Analysis Dashboard
        ### Real-Time Sentiment Classification using Machine Learning
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                lines=5,
                placeholder="Type your tweet or text here...",
                label="📝 Input Text"
            )

            analyze_btn = gr.Button("🔍 Analyze Sentiment")

        with gr.Column(scale=1):
            prediction_output = gr.Markdown()
            confidence_output = gr.Markdown()

    gr.Markdown("## 📊 Sentiment Probability Distribution")
    probability_output = gr.Label(num_top_classes=4)

    analyze_btn.click(
        analyze_sentiment,
        inputs=text_input,
        outputs=[prediction_output, confidence_output, probability_output]
    )

    gr.Examples(
        examples=[
            ["I absolutely love this new phone!"],
            ["This is the worst experience ever."],
            ["I am watching the match tonight."],
            ["The service was average, nothing special."],
            ["Playstation 5 is amazing!"],
            ["I'm very disappointed with the product."]
        ],
        inputs=text_input
    )

if __name__ == "__main__":
    demo.launch()
