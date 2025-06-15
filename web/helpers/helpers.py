import streamlit as st
import os
from dotenv import load_dotenv
import pandas as pd
import time

# Import ntviz libraries
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ntviz import Manager, TextGenerationConfig, llm

import seaborn as sns
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from PIL import Image

# Load environment variables
load_dotenv()


def load_api_key():
    """Load Gemini API key from user input."""
    if "gemini_api_key" not in st.session_state:
        st.session_state.gemini_api_key = ""

    with st.sidebar.popover(label=" ☑️:violet[Provider Instruction]"):
        st.info("""
            #### Instructions
            This system currently only supports **Gemini**.
            - Please enter your Gemini API key below.
            - If you want to use a different provider in the future, contact the admin.
        """)

    api_key = st.sidebar.text_input(
        "Gemini API key:",
        value=st.session_state.gemini_api_key,
        type="password",
        placeholder="Enter your Gemini API key"
    )
    st.session_state.gemini_api_key = api_key

    if not api_key:
        st.warning("Please enter the Gemini API key to continue.")

    return api_key, "gemini"



def upload_file():
    """
    This function:
    - Uploads a dataset file in .csv or .json format
    - Reads input data for subsequent tasks
    """
    uploaded_file = st.file_uploader("Upload a data file in .csv format:", type=["csv"])
    if uploaded_file is not None:
        # Process the uploaded file
        if uploaded_file.name.endswith(".csv"):
            # Read CSV file
            df = pd.read_csv(uploaded_file)
            st.success(f"Successfully uploaded a CSV file with {len(df)} rows of data.")
        else:
            st.error("This format is not supported.")
        
        return df
    else:
        st.error("No data file found. Please try again.")
        return None


def clean_df(df):
    """
    Goal: Automate the data cleaning process for tasks such as replacing missing values and removing duplicates.
    
    Args:
        df (DataFrame): The dataset provided by the user

    Returns:
        cleaned_df: The cleaned dataset
    """
    df = df.copy()
    
    # Fill missing values for numerical columns with the mean value
    for col in df.select_dtypes(include=['number']).columns:
        df[col] = df[col].fillna(df[col].mean())
        
    # Remove duplicate values
    df = df.drop_duplicates()
    
    return df

# Convert base64 string to image
def base64_to_image(base64_str):
    img_data = base64.b64decode(base64_str)
    img = Image.open(BytesIO(img_data))
    return img

# Function to output, display code, explain, and evaluate the chart
def display_charts(
    ntviz, 
    chart, 
    goal, 
    library='seaborn', 
    textgen_config=None
):
    """
    Display, explain, and evaluate a chart in detail.
    """
    with st.spinner('Wait for it...'):
        time.sleep(5)
        # Check if chart exists
        if not chart:
            st.warning("No chart to display.")
            return

        # Convert chart to image
        try:
            img = base64_to_image(chart.raster)
            
        except Exception as e:
            st.error(f"Error converting chart: {e}")
            return

        # Display the chart image
        st.image(img)

        # Download button
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        href = f'<a href="data:image/png;base64,{img_str}" download="chart.png"> ˚⋆ ʕっ• ᴥ • ʔっ Download Chart ˚⋆ </a>'
        st.markdown(href, unsafe_allow_html=True)

        
        # Display VizOps
        with st.popover(label=":grey[⚙️ VizOps]"):
            code, explain, evaluations = st.tabs(["Code", "Explaination", "Evaluation"])
        
            # Display code
            with code:
                st.code(chart.code, language="python")

            # Explain the chart
            with explain:
                try:
                    explanations = ntviz.explain(
                        code=chart.code, 
                        library=library, 
                        textgen_config=textgen_config
                    )
                    for row in explanations[0]:
                        st.write(row["section"], " ** ", row["explanation"])
                except Exception as e:
                    st.error(f"Cannot explain the chart: {e}")

            # Evaluate the chart
            with evaluations:
                try:
                    eval_results = ntviz.evaluate(
                        code=chart.code,
                        image=chart.raster,
                        goal=goal,
                        textgen_config=textgen_config,
                        library=library
                    )[0]  
                    
                    for eval in eval_results:
                        aspect = eval["aspect"]
                        st.markdown(f"### {aspect.upper()} Evaluation")
                        st.markdown(f"**Average Score**: {eval['average']}/10")

                        for e in eval["evaluations"]:
                            dimension = e["dimension"]
                            score = e["score"]
                            rationale = e["rationale"]
                            st.markdown(f"- **{dimension.capitalize()}**: {score}/10")
                            st.markdown(f"  > {rationale}")

                except Exception as e:
                    st.error(f"Cannot evaluate the chart: {e}")
                    
            
        
def analyze_chart(
    ntviz,
    df,
    summary,
    chart,
    library="seaborn",
    textgen_config=None
):
    """
    Analyze the generated chart using ntviz and display insights.
    
    Args:
        ntviz: The ntviz manager object
        df (pd.DataFrame): The input data
        summary (str): Summary of the dataset or goal
        chart: The chart object generated earlier
        library (str): Visualization library used
        textgen_config (TextGenerationConfig): Configuration for LLM
    """


    try:
        # Run analysis
        analysis = ntviz.analyze(
            chart,
            df,
            summary,
            library,
            textgen_config,
        )

        # Display section
        st.subheader("📊 Chart Analysis")
        with st.expander("Analysis Report", expanded=True):
            st.write(analysis)

    except Exception as e:
        st.error(f"An error occurred during chart analysis: {e}")


