import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import time
import os

# Import NTViz library
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ntviz import Manager, TextGenerationConfig, llm

# Import helper functions from the helpers directory
from helpers.helpers import (
    upload_file,
    clean_df,
    load_api_key,
    base64_to_image,
    display_charts,
    analyze_chart
)

# Load environment variables
load_dotenv()




# Set up textgen config
def setup_model_textgen(models, provider):
    filter, requirements = st.tabs(["👩‍🏫 Filter Instruction", "🛑 Requirements"])
    # Configure text generation for Cohere
    with filter:
        with st.popover("Instruction:"):
            st.markdown(""" 
                        Set the temperature (creativity level):
                        - Lower values (e.g., 0.1–0.3): The model produces simple and predictable results.
                        - Higher values (e.g., 0.7–1.0): The model becomes more creative and complex, but the results might be less focused.
                        - Recommended range: Choose a temperature between 0.5 and 0.7 for a good balance between accuracy and creativity.
                                """)
            # if provider == "Cohere":
            #     st.markdown(""" 
            #                     Choose the model you would like to use for your dataset:
            #                     - "command-xlarge-nightly": The largest model with the most details, able to learn complex patterns and relationships in data.
            #                     - "command-large: Smaller than xlarge but still powerful, offering a good mix of performance and efficiency.
            #                     - "command-base-nightly" : The smallest and lightest model, designed for quick use and easy deployment.
            #                 """)
            if provider == "gemini":
                st.markdown(""" 
                                Choose the model you would like to use for your dataset:
                                - "Gemini 1.5 Flash:" The largerst model offers a good balance between size and performance. It can handle a wide range of tasks and is suitable for many applications.
                                - "Gemini 1.5 Flash-8B:" The smallest model in the Gemini family, designed for simpler tasks and devices with limited resources.
                                - "Gemini 1.5 Pro:" This model is optimized for specific tasks, such as code generation or technical translation.
                            """)
    with requirements:
            st.markdown(""" 
                    **NTViz works best with the datasets:**
                    - Columns: ≤ 15  
                    - Rows: ≤ 10000  
                    - File Size: ≤ 1MB  

                    **Correct CSV Format:**  
                    Your file should contain only the variable names as column headers and the corresponding values.  
                    Avoid including unrelated information such as titles, notes, or daily reports in the file.  
                    """)
                    
            with st.popover("📋 Example:"):
                    st.markdown(""" 
                        **✅ Correct Input Example:**
                        ```csv
                        Name, Age, Country, Salary
                        John, 25, USA, 50000
                        Anna, 30, Canada, 60000
                        Mark, 28, UK, 45000
                        ```

                        **❌ Incorrect Input Example:**
                        ```csv
                        ,Daily Report,,
                        Date,12/16/2024 20:38,,
                        Tên,Phiến,,
                        ,,,
                        ID,Name,Category,Date
                        1,A,BCV,12/16/2024 20:38
                        2,B,B1,12/16/2024 20:38
                        3,A,B2,12/16/2024 20:38
                        ```
                        - Extra rows like "Daily Report" or empty cells will cause errors.  
                        - Remove unnecessary information before uploading.  
                        """)
                    
    # Choose the model and the temperature
    # Model and temperature selection
    temperature = st.slider("Temperature", 
                            min_value=0.0, 
                            max_value=1.0, 
                            value=0.5, 
                            step=0.1)
    model = st.selectbox("Select Model:", models)
    textgen_config = TextGenerationConfig(
                    n=1, 
                    temperature=temperature, 
                    model=model,
                    use_cache=True,
                    provider=provider
                )
    return textgen_config
            
        
def initialize_ntviz_and_api(api_key, provider):
    """
    Initialize NTViz with default configuration.
    
    Args:
        api_key (object): The user's API key for the Cohere platform.
    """
    try:
        # Cohere provider
        # Initialize Cohere with the API key and set up with NTViz
        # if provider == "Cohere":
        #     text_gen = llm("cohere", api_key=api_key)
        #     ntviz = Manager(text_gen=text_gen)
            
        #     models =  ["command-xlarge-nightly", 
        #                "command-large", 
        #                "command-base-nightly"]
        #     config = setup_model_textgen(models, provider=provider)   
        #     # textgen_config = config
        
        #     st.sidebar.success("Successfully connected to Cohere!")
     
        
        if provider == "gemini":
            text_gen = llm("gemini", api_key = api_key)
            ntviz = Manager(text_gen=text_gen)
            models =["gemini-1.5-flash", 
                    "gemini-1.5-flash-8b",
                    "gemini-1.5-pro"]
                
            config = setup_model_textgen(models=models, provider=provider)   
            # textgen_config = config
            st.sidebar.success("Successfully connected to Gemini!")
            
        
        textgen_config = config
           
        return ntviz, textgen_config
    
    except Exception as e:
        st.sidebar.error(f"Unable to connect to {provider}: {e}")
        return None, None

def process_data_summary(df, ntviz, textgen_config):
    """
    Perform data summarization and goal setting.
    
    Args:
        df (DataFrame): Input dataset.
        ntviz: NTViz systemp to handle tasks.
        textgen_config: TextGeneration configuration for NTViz.
    
    Returns:
        Summary and goals based on user requirements.
    """
    st.dataframe(df.head())
    
    # Check and clean data
    null_values = df.isnull().sum()
    dup_values = df.duplicated().sum()
    
    if null_values.any() > 0 or dup_values > 0:  
        df = clean_df(df)
        st.success("Data cleaned!")
    else:
        st.success("No missing or duplicate values found in the data.")
    with st.spinner('Wait for it...'):
        time.sleep(5)
        summary = ntviz.summarize(df, summary_method="default", textgen_config=textgen_config)
        goals = ntviz.goals(summary, n=5, textgen_config=textgen_config)

    return summary, goals



def generate_visualizations(ntviz, summary, goals, df, textgen_config):
    if "generated_charts" not in st.session_state:
        st.session_state.generated_charts = []

    library = "seaborn"
    n = 5

    if st.button("Generate Charts"):
        st.session_state.generated_charts = []  
        for i in range(n):
            try:
                charts = ntviz.visualize(summary=summary, goal=goals[i], library=library)
                for chart in charts:
                    st.session_state.generated_charts.append({
                        "goal": goals[i],
                        "chart": chart
                    })
            except Exception as e:
                st.error(f"Error generating chart for Goal {i+1}: {e}")

    for i, item in enumerate(st.session_state.generated_charts):
        goal = item["goal"]
        chart = item["chart"]

        st.subheader(f"✷ :grey[Insight {i+1}:]")
        st.write(goal)

        display_charts(
            ntviz,
            chart,
            goal,
            library,
            textgen_config
        )

        # Analyze Chart Button
        if st.button(f"🔍 Analyze Chart {i+1}", key=f"analyze_{i+1}"):
            executed_viz = ntviz.execute(
                code_specs=[chart.code],
                data=df,
                summary=summary,
                library=library
            )
            analyze_chart(
                ntviz,
                df,
                summary,
                executed_viz[0],
                textgen_config = textgen_config
            )

        st.divider()




def process_user_query_graphs(df, ntviz, textgen_config, provider):
    """
    Generate visualizations based on user queries.
    """
    if "generated_query" not in st.session_state:
        st.session_state.generated_query = []

    user_query = st.text_area(
        label="Type Your Question Here:",
        placeholder="e.g., What is the correlation between height and weight? Or, Who are the best opponents in this dataset?"
    )

    k = 1
    st.info("Gemini can only generate one chart at a time because the chat functionality does not support a candidate_count greater than 1.")
    
    if st.button("Generate Charts"):
        try:
            textgen_config.n = k
            summary = ntviz.summarize(df, summary_method="default", textgen_config=textgen_config)
            query_charts = ntviz.visualize(summary=summary, goal=user_query, textgen_config=textgen_config)
            
            # Lưu vào session_state để truy cập lại sau
            st.session_state.generated_query = [
                {"goal": user_query, "chart": chart, "summary": summary}
                for chart in query_charts
            ]

            if len(query_charts) < k:
                st.error(f"Sorry, we could only suggest {len(query_charts)} charts at the moment.")
                st.warning("We suggest you to change your temperature and try again. Thank you.")

        except Exception as e:
            st.error(f"Error initializing NTViz or Gemini: {e}")
    
    # Hiển thị và phân tích từng biểu đồ đã sinh
    for i, item in enumerate(st.session_state.generated_query):
        chart = item["chart"]
        goal = item["goal"]
        summary = item["summary"]

        display_charts(
            ntviz,
            chart,
            goal,
            "seaborn",
            textgen_config
        )

        if st.button(f"🔍 Analyze Chart {i+1}", key=f"analyze_{i+1}"):
            try:
                executed_viz = ntviz.execute(
                    code_specs=[chart.code],
                    data=df,
                    summary=summary,
                    library="seaborn"
                )
                analyze_chart(
                    ntviz,
                    df,
                    summary,
                    executed_viz[0],
                    textgen_config=textgen_config
                )
            except Exception as e:
                st.error(f"Error during chart analysis: {e}")


def process_viz_recommend(df, ntviz, textgen_config):
    """
    Recommend charts based on data summary.
    
    Returns:
        Recommended visualizations based on user requests.
    """
    k = st.number_input(label="Number of Charts:", min_value=1, max_value=5, step=1)
    
    if st.button(label="Generate Charts"):
        try:
            
            with st.spinner('Wait for it...'):
                time.sleep(3)
                summary = ntviz.summarize(df, 
                                         summary_method="default", 
                                         textgen_config=textgen_config)
                goals = ntviz.goals(summary, 
                                   n=1, 
                                   textgen_config=textgen_config)
                
                charts = ntviz.visualize(summary=summary, 
                                        goal=goals[0], 
                                        library="seaborn")
                
                if charts:
                    recommended_charts = ntviz.recommend(
                        code=charts[0].code, 
                        summary=summary, 
                        n=k,  
                        textgen_config=textgen_config
                    )
                    
                    st.write(f"Suggested number of charts: {len(recommended_charts)}")
                    for chart in recommended_charts:
                        display_charts(
                            ntviz, 
                            chart, 
                            goals[0], 
                            "seaborn", 
                            textgen_config
                        )
                    if len(recommended_charts) < k:
                        st.error(f"Sorry, we could only suggest {len(recommended_charts)} charts at the moment.")
                        st.warning("We suggest you to change your temperature and try again. Thank you.")
                else:
                    st.error("No charts were generated.")
                    st.warning("We suggest you to change your temperature and try again. Thank you.")
        
        except Exception as e:
            st.error(f"Error during chart recommendation: {e}")




# Additional helper function to get the current provider's API key
def get_current_api_key():
    """Get the API key for the currently selected provider."""
    provider = st.session_state.get("provider", "gemini")
    if provider == "gemini":
        return st.session_state.gemini_api_key
    

def show_task():
    """
    Main function to coordinate tasks in the app.
    """
    # Load API key
    api_key, provider = load_api_key()
            
    # Initialize NTViz and Provider
    ntviz, textgen_config = initialize_ntviz_and_api(api_key, provider) if api_key else (None, None)

    with st.sidebar.container():
        st.header("Tasks:")
        task = st.selectbox("Functions:", ["VizRecommend", 
                                           "UserQuery based graphs",
                                           "ExtraViz"                                           ])

    # Task-specific content
    df = upload_file()
    
    if df is not None:  
        if task == "VizRecommend" and ntviz:
            summary, goals = process_data_summary(df, ntviz, textgen_config)
            generate_visualizations(ntviz, summary, goals, df, textgen_config)
        
        elif task == "UserQuery based graphs" and ntviz:
            df = clean_df(df)
            process_user_query_graphs(df, ntviz, textgen_config, provider)
        
        elif task == "ExtraViz" and ntviz:
            df = clean_df(df)
            process_viz_recommend(df, ntviz, textgen_config)
  
        else:
            st.error("Please select a valid task or ensure NTViz is initialized correctly.")

show_task()
