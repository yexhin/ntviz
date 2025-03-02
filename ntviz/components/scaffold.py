from dataclasses import asdict

from ntviz.datamodel import Goal


# if len(plt.xticks()[0])) > 20 assuming plot is made with plt or
# len(ax.get_xticks()) > 20 assuming plot is made with ax, set a max of 20
# ticks on x axis, ticker.MaxNLocator(20)

class ChartScaffold(object):
    """Return code scaffold for charts in multiple visualization libraries"""

    def __init__(
        self,
    ) -> None:

        pass

    def get_template(self, goal: Goal, library: str):

        # general_instructions = f"""
        # If the solution requires a single value (e.g. max, min, median, first, last etc), 
        # ALWAYS add a line (axvline or axhline) to the chart, 
        # ALWAYS with a legend containing the single value (formatted with 0.2F). 
        # If using a <field> where semantic_type=date, 
        # YOU MUST APPLY the following transform before using that column 
        # i) convert date fields to date types using data[''] = pd.to_datetime(data[<field>], errors='coerce'), 
        # ALWAYS use  errors='coerce' 
        # ii) drop the rows with NaT values data = data[pd.notna(data[<field>])] 
        # iii) convert field to right time format for plotting.  
        # ALWAYS make sure the x-axis labels are legible (e.g., rotate when needed). 
        # Solve the task  carefully by completing ONLY the <imports> AND <stub> section. 
        # Given the dataset summary, the plot(data) method should generate a {library} chart ({goal.visualization}) that addresses this goal: {goal.question}. DO NOT WRITE ANY CODE TO LOAD THE DATA. The data is already loaded and available in the variable data.
        # """

    
        
        # general_instructions = """
        # You are a helpful and professional assistant with expertise in generating optimized code templates for data visualizations.
        
        # # Code Scaffold Guidelines:
        # ### 1. Constraints: 
        # ### If the solution requires a single value (e.g. mean, max, min, median, first, last etc):
        #  i) ALWAYS add a reference line (axvline or axhline) to the chart, 
        #  ii) ALWAYS with a legend containing the single value (formatted with 0.2F). 
         
        # ### 2. DATA Handling:
        # ### 2.1 Data field:
        # - If using a <field> where 'semantic_type = date', YOU MUST APPLY the following transformation before using:
        #     i) Convert date fields to date types using:
        # ```python
        #  data['field'] = pd.to_datetime(data['<field>'], errors='coerce')
        # ```    
        
        #     ii) Drop the rows with NaT values data = data[pd.notna(data[<field>])] 
            
        #     iii) Ensure the field is formatted correctly for plotting:
        # ```python
        # data['<field>'] = data['<field>'].dt.strftime('%Y-%m-%d')  # Adjust as needed
        # ```
        
        # ### 2.2 **Numerical Fields**:
        # - Fill missing values with median (`data['<field>'].fillna(data['<field>'].median(), inplace=True)`).
        
        # ### 3. Expected Output:
        # i) ALWAYS make sure the x-axis, y-axis labels are legible (e.g., rotate x-axis when needed). 
        # ii) Solve the task carefully.
        # iii) ONLY modify the <imports> AND <stub> section when completing the task. 
        # iv) Given the dataset summary, Ensure plot(data) method generates a {library} chart ({goal.visualization}) that directly addresses this goal: {goal.question}.
        # v) NEVER WRITE ANY CODE TO LOAD THE DATA. The data is already loaded and available."
        # """
        
        
        general_instructions = """
        You are a helpful and professional assistant with expertise in generating optimized code templates for data visualizations.
        1. **Analyze the dataset:**
        - What are the column names and data types?
        - Identify numerical vs categorical fields.
        - Check for missing values.

        2. **Determine the visualization best practice approach:**
        - If time-series → Line Chart.
        - If category comparison → Bar Chart.
        - If distribution → Histogram and the overlaid density estimates.

        3. **Generate the code accordingly.**
        
        4. Expected Output:
        i) ALWAYS make sure the x-axis, y-axis labels are legible (e.g., rotate x-axis when needed). 
        ii) Solve the task carefully.
        iii) ONLY modify the <imports> AND <stub> section when completing the task. 
        iv) Given the dataset summary, Ensure plot(data) method generates a {library} chart ({goal.visualization}) that directly addresses this goal: {goal.question}.
        v) NEVER WRITE ANY CODE TO LOAD THE DATA. The data is already loaded and available."
        """
        
        matplotlib_instructions = f" {general_instructions} DO NOT include plt.show(). The plot method must return a matplotlib object (plt). Think step by step. \n"

        if library == "matplotlib":
            instructions = {
                "role": "assistant",
                "content": f"  {matplotlib_instructions}. Use Cartopy for charts that require a map. "}
            template = \
                f"""
import matplotlib.pyplot as plt
import pandas as pd
<imports>
# plan -
def plot(data: pd.DataFrame):
    <stub> # only modify this section
    plt.title('{goal.question}', wrap=True)
    return plt

chart = plot(data) # data already contains the data to be plotted. Always include this line. No additional code beyond this line."""
        elif library == "seaborn":
            instructions = {
                "role": "assistant",
                "content": f"{matplotlib_instructions}. Use Cartopy for charts that require a map. "}

            template = \
                f"""
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
<imports>
# solution plan
# i.  ..
def plot(data: pd.DataFrame):

    <stub> # only modify this section
    plt.title('{goal.question}', wrap=True)
    return plt

chart = plot(data) # data already contains the data to be plotted. Always include this line. No additional code beyond this line."""

        elif library == "ggplot":
            instructions = {
                "role": "assistant",
                "content": f"{general_instructions}. The plot method must return a ggplot object (chart)`. Think step by step.p. \n",
            }

            template = \
                f"""
import plotnine as p9
<imports>
def plot(data: pd.DataFrame):
    chart = <stub>

    return chart

chart = plot(data) # data already contains the data to be plotted. Always include this line. No additional code beyond this line.. """

        elif library == "altair":
            instructions = {
                "role": "system",
                "content": f"{general_instructions}. Always add a type that is BASED on semantic_type to each field such as :Q, :O, :N, :T, :G. Use :T if semantic_type is year or date. The plot method must return an altair object (chart)`. Think step by step. \n",
            }
            template = \
                """
import altair as alt
<imports>
def plot(data: pd.DataFrame):
    <stub> # only modify this section
    return chart
chart = plot(data) # data already contains the data to be plotted.  Always include this line. No additional code beyond this line..
"""

        elif library == "plotly":
            instructions = {
                "role": "system",
                "content": f"{general_instructions} If calculating metrics such as mean, median, mode, etc. ALWAYS use the option 'numeric_only=True' when applicable and available, AVOID visualizations that require nbformat library. DO NOT inlcude fig.show(). The plot method must return an plotly figure object (fig)`. Think step by step. \n.",
            }
            template = \
                """
import plotly.express as px
<imports>
def plot(data: pd.DataFrame):
    fig = <stub> # only modify this section

    return chart
chart = plot(data) # variable data already contains the data to be plotted and should not be loaded again.  Always include this line. No additional code beyond this line..
"""

        else:
            raise ValueError(
                "Unsupported library. Choose from 'matplotlib', 'seaborn', 'plotly', 'bokeh', 'ggplot', 'altair'."
            )

        return template, instructions
