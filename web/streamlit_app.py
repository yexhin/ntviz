
import seaborn as sns
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from PIL import Image
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from st_pages import add_page_title, get_nav_from_toml


import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "key/gen-lang-client-0317182817-0db658957f9a.json"

def main():
    # Load environment variables
    load_dotenv()
    # DESIGN implement changes to the standard streamlit UI/UX
    
    st.set_page_config(page_title="NTViz", 
                       page_icon="./web/material/outlook/logo.jpg", 
                       layout="wide", 
                       initial_sidebar_state="expanded")
    sections = st.sidebar.toggle("Sections", 
                                 value=True, 
                                 key="use_sections")

    nav = get_nav_from_toml(
        "./web/.streamlit/pages_sections.toml" if sections else "./web/.streamlit/pages.toml"
    )
    # Decor
    st.logo(image="./web/material/outlook/logo.jpg", size="large",icon_image=None)
    pg = st.navigation(nav)

    add_page_title(pg)

    pg.run()
    
if __name__ == "__main__":
    main()