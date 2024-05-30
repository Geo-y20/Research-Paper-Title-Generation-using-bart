import streamlit as st
import torch
from transformers import BartForConditionalGeneration, BartTokenizer
from PIL import Image
import base64

# Set the title and description
st.set_page_config(page_title="Text to Title Generation", page_icon="📚")

# Custom CSS to style the page
st.markdown("""
    <style>
    .main {
        background-color: #1E1E1E;
        color: #FFFFFF;
        font-family: 'Helvetica Neue', sans-serif;
    }
    .sidebar .sidebar-content {
        background-color: #333333;
        color: white;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
        margin: 10px 0;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stTextArea textarea {
        background-color: #333333;
        color: white;
        font-size: 16px;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF;
    }
    .header-title {
        text-align: center;
        margin-top: 20px;
    }
    .header-title h1 {
        font-size: 2.5rem;
    }
    .footer {
        text-align: center;
        margin-top: 50px;
    }
    .footer a {
        color: #4CAF50;
        text-decoration: none;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.markdown("""
<div class="header-title">
    <h1>Text to Title Generation with BART 📚</h1>
</div>
""", unsafe_allow_html=True)

st.markdown("""
This application generates a title for a given abstract using a BART model fine-tuned for text generation.
Simply enter the abstract text in the box below and click on the "Generate Title" button.
""")

# Sidebar description
st.sidebar.markdown("## About this App")
st.sidebar.markdown("""
This project leverages the power of BART (Bidirectional and Auto-Regressive Transformers) for generating titles from abstracts.

**How to Use:**
1. Enter the abstract text in the provided text area.
2. Click the "Generate Title" button to get a generated title for your abstract.

**Model Information:**
- The model used is BART (Bidirectional and Auto-Regressive Transformers).
- Fine-tuned for text generation tasks.

**Developed by:**
- George Youhana
- Fares Fathy
""")

# Specify the path to the model directory
model_path = "model"  # Replace with the actual path

# Load the model and tokenizer
with st.spinner("Loading model..."):
    model = BartForConditionalGeneration.from_pretrained(model_path)
    tokenizer = BartTokenizer.from_pretrained(model_path)

st.sidebar.success("Model loaded successfully!")

# Input text box
st.markdown("### Enter the abstract text below:")
input_text = st.text_area("Abstract", "")

# Generate Title button
if st.button("Generate Title"):
    if input_text:
        with st.spinner("Generating title..."):
            inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
            summary_ids = model.generate(inputs["input_ids"], num_beams=4, max_length=150, early_stopping=True)
            predicted_title = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        st.subheader("Predicted Title:")
        st.write(predicted_title)
        
        # Download button
        b64 = base64.b64encode(predicted_title.encode()).decode()
        href = f'<a href="data:file/txt;base64,{b64}" download="title.txt">Download Title</a>'
        st.markdown(href, unsafe_allow_html=True)
        
        # Feedback section
        st.markdown("### Provide your feedback:")
        feedback = st.radio("Is the generated title satisfactory?", ("Yes", "No"))
        if feedback == "No":
            st.text_area("What can be improved?", "")
    else:
        st.warning("Please enter some text to generate a title.")

# Footer
st.markdown("""
<div class="footer">
    <p>Prepared by:</p>
    <p>
        <a href="https://www.linkedin.com/in/george-youhana-a5b756155/" target="_blank">George Youhana</a> |
        <a href="https://www.linkedin.com/in/trdaxy/" target="_blank">Fares Fathy</a>
    </p>
    <p>
        <a href="https://github.com/Geo-y20/Research-Paper-Title-Generation-using-bart.git" target="_blank">GitHub Repository</a>
    </p>
</div>
""", unsafe_allow_html=True)
