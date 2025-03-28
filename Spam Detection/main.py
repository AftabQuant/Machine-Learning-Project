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

ps = PorterStemmer()

def load_model():
    scaler = joblib.load('vectorizer.joblib')
    model = joblib.load('model.joblib')
    return scaler, model

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)

def predict(text):
    scaler , model = load_model()
    trans_text = transform_text(text) # preprocessing
    scaled_text = scaler.transform([trans_text])
    prediction = model.predict(scaled_text)[0]
    if prediction==0:  return "This message is not a spam sms", "https://media.istockphoto.com/id/1462430494/vector/yellow-paper-envelope-with-green-checkmark-on-white-background-no-viruses-cincept-3d-vector.jpg?s=1024x1024&w=is&k=20&c=q2Es9we59N7PBXfYsh2Ytq91IqNFTdmOzEQhmo6aU7Y="
    else:  return "This message is a spam sms", "https://cdn.prod.website-files.com/659fa592476e081fbd5a3335/669f84768899d32f200f7556_spamz.png"

demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(label='Enter Email Text'),
    outputs=[gr.Textbox(label='Prediction'), gr.Image(label='Spam Warning Image')],
    title='Email Spam Detection',
    description='Enter an email text to check if it is spam or not. If spam, a warning image will be displayed.',
)
if __name__ =="__main__":
    demo.launch()
