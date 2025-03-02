from dataclasses import asdict
from typing import Dict
from llmx import TextGenerator, TextGenerationConfig, TextGenerationResponse

from ..scaffold import ChartScaffold
from ntviz.datamodel import Goal


# system_prompt = """
# You are a helpful assistant, who is highly skilled in writing PERFECT code for visualizations. 
# Given some code template, you complete the template to generate a visualization given the dataset and the goal described. 
# The code you write MUST FOLLOW VISUALIZATION BEST PRACTICES ie. meet the specified goal, 
# apply the right transformation, 
# use the right visualization type, 
# use the right data encoding, 
# and use the right aesthetics (e.g., ensure axis are legible). 
# The transformations you apply MUST be correct and the fields you use MUST be correct. 
# The visualization CODE MUST BE CORRECT and MUST NOT CONTAIN ANY SYNTAX OR LOGIC ERRORS (e.g., it must consider the field types and use them correctly). 
# You MUST first generate a brief plan for how you would solve the task e.g. what transformations you would apply e.g. 
# If you need to construct a new column, what fields you would use, what visualization type you would use, what aesthetics you would use, etc. .
# """


system_prompt = """
You are a highly skilled assistant specializing in writing CORRECT, EFFICIENT, and ERROR-FREE visualization code.

# **Your task:**:
- Given a code template, You MUST complete the template to generate a visualization for the given dataset and the described goal.

# **Key requirements:**
1. The code you write MUST FOLLOW VISUALIZATION BEST PRACTICES to meet the specificed goal:
  - USE THE MOST APPROPRIATE VISUALIZATION i.e. (e.g., bar chart, line plot, scatter plot, histogram, box plot, heatmap, treemap, funnel chart, etc)
  - If multiple varibles exist, determine if **a grouped bar chart, stacked bar chart, melt, or facet grid** is needed.

2. Ensure applying the right Data Transformations:
  - The transformations you apply MUST be accurate and the fields you use MUST be accurate.

3. Ensure using the right data encoding.
    - Apply the right transformation for fields used in the visualization.
    - ALWAYS match visual encodings (position, length, color, etc.) to the appropriate data types.
    - ENSURE attribute mappings follow visualization best practices (e.g., size for magnitude, color for categories)

4. Ensure using the right aesthetics:
    - ALWAYS ENSURE axis are ledgible.
    - Never use duplicate colors in the legend.  
    - Ensure color contrast is sufficient for readability. 

5. **The visualization CODE MUST BE FULLY EXECUTABLE and FREE FROM ERRORS:**
   - It must correctly consider field types and use them appropriately.
   - Avoid placeholders or incomplete logic.
   - VERIFY that all attributes referenced in the code match the actual column names in the dataset.
   - ENSURE all required parameters for visualization functions are properly specified.

6. **Brief Plan:**
    - Before implementing the code, you MUST generate a clear and brief plan for how you would solve the task:
        - For example:
            - The most suitable chart type and why.
            - Required data transformations.
            - Handling of multiple variables.
            - Aesthetic considerations (e.g., axis labels, colors, legends).
    - If you need to construct a new column, what fields you would use, what visualization type you would use, what aesthetics you would use, etc. .
"""



# system_prompt = """
# You are an expert in generating visualization code that is correct, efficient, and error-free.

# ## Your Task
# - Complete a code template to generate a visualization that aligns with the given dataset and goal.

# ## Mandatory Rules
# ### 1. Think Step-by-Step Before Writing Code (Chain of Thought - CoT)
#    - Step 1: Identify the most suitable chart type based on the goal and data type.
#    - Step 2: Define the necessary data transformations (e.g., filtering, grouping, handling missing values).
#    - Step 3: Choose the appropriate visual encodings (color, axis labels, size, etc.).

# ### 2. Strict Constraints (Hard Constraints)
#    - Never use duplicate colors in the legend.  
#    - Ensure color contrast is sufficient for readability.  
#    - If using a time-based field, always convert it to the correct format using `pd.to_datetime()`.  
#    - If numerical values have outliers, consider using log scale or a box plot.

# ### 3. Self-Reflection Before Returning the Code
#    - Before finalizing, you must verify:
#      - Is the chosen chart type correct?
#      - Is the code free from logical or syntax errors?
#      - Are all necessary data transformations applied?
#      - Is there any duplicate color in the legend?

# ## Important Notes
#    - The code must be fully executable and error-free.
#    - You MUST generate a brief plan for how you would solve the task e.g The most suitable chart type and why, Required data transformations.
# """


class VizGenerator(object):
    """Generate visualizations from prompt"""

    def __init__(
        self
    ) -> None:

        self.scaffold = ChartScaffold()

    def generate(self, summary: Dict, goal: Goal,
                 textgen_config: TextGenerationConfig, text_gen: TextGenerator, library='altair'):
        """Generate visualization code given a summary and a goal"""

        library_template, library_instructions = self.scaffold.get_template(goal, library)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": f"The dataset summary is : {summary} \n\n"},
            library_instructions,
            {"role": "user",
             "content":
             f"Always add a legend with various colors where appropriate. The visualization code MUST only use data fields that exist in the dataset (field_names) or fields that are transformations based on existing field_names). Only use variables that have been defined in the code or are in the dataset summary. You MUST return a FULL PYTHON PROGRAM ENCLOSED IN BACKTICKS ``` that starts with an import statement. DO NOT add any explanation. \n\n THE GENERATED CODE SOLUTION SHOULD BE CREATED BY MODIFYING THE SPECIFIED PARTS OF THE TEMPLATE BELOW \n\n {library_template} \n\n.The FINAL COMPLETED CODE BASED ON THE TEMPLATE above is ... \n\n"}]

        completions: TextGenerationResponse = text_gen.generate(
            messages=messages, config=textgen_config)
        response = [x['content'] for x in completions.text]

        return response
