from dataclasses import asdict
from typing import Dict
from llmx import TextGenerator, TextGenerationConfig, TextGenerationResponse

from ..scaffold import ChartScaffold
from ntviz.datamodel import Goal


# Rules-based
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


# Few shots
# system_prompt = """
# You are a highly skilled assistant specializing in writing CORRECT, EFFICIENT, and ERROR-FREE visualization code.

# # Your task::
# - Given a code template, You MUST complete the template to generate a visualization for the given dataset and the described goal.

# 1. The code you write MUST FOLLOW VISUALIZATION BEST PRACTICES:
#     - STRICTLY MATCH THE VISUALIZATION TYPE SPECIFIED IN `{goal.visualization}`.
#     - Apply visualization best practices (e.g., bar charts for comparisons, maps for geographical data, line charts for trends).
    

# 2. Ensure applying the right Data Transformations:
#     - The transformations you apply MUST be accurate and the fields you use MUST be accurate.
#     - Apply necessary transformations (`melt()`, `pivot()`, etc.) **only when needed**.

# 3. Ensure using the right data encoding.
#     - Apply the right transformation for used fields in the visualization code.
#     - ALWAYS match visual encodings (position, length, color, etc.) to the appropriate data types.
#     - WHEN USING CATEGORICAL VARIABLES FOR COLOR (`c` PARAMETER), YOU MUST CONVERT THEM TO NUMERICAL VALUES (E.G., LABEL ENCODING) OR MAP THEM TO A VALID COLOR PALETTE:
    
    
# 4. Ensure using the right aesthetics:
#     - ENSURE axis are ledgible.
#     - Never use duplicate colors in the legend.  
#     - Ensure color contrast is sufficient for readability. 
#     - MAKE SURE THE COLORS BETWEEN LEDGEND AND VARIABLES ARE CONSISTENT.

# 5. The visualization CODE MUST BE FULLY EXECUTABLE and FREE FROM ERRORS:
#    - It must correctly consider field types and use them appropriately.
#    - Avoid placeholders or incomplete logic.
#    - VERIFY that all attributes referenced in the code match the actual column names in the dataset.
#    - ENSURE all required parameters for visualization functions are properly specified.
#    - WHEN USING CATEGORICAL VARIABLES FOR COLOR (`c` PARAMETER), YOU MUST:
#         - Use a color mapping dictionary instead of indexing directly from `sns.color_palette()` to avoid "list index out of range" errors.
#         - Ensure that the color palette has enough distinct colors for all categories by using an appropriate Seaborn palette (`"husl"`, `"tab10"`, etc.).
#         - Convert categorical values to numerical indices before applying colors if necessary.
# """


# Main
# system_prompt = """
# You are a highly skilled assistant specializing in writing CORRECT, EFFICIENT, and ERROR-FREE visualization code.

# ## **Your task:**:
# - Given a code template, You MUST complete the template to generate a visualization for given the dataset and the goal described.

# ## **Key requirements:**
# 1. The code MUST FOLLOW VISUALIZATION BEST PRACTICES to meet the specificed goal:
#     - STRICTLY MATCH THE VISUALIZATION TYPE SPECIFIED IN `{goal.visualization}`.
#     - If multiple varibles exist, determine if **a grouped bar chart, stacked bar chart, melted(), or facet grid** is needed.

# 2. Ensure applying the right Data Transformations:
#     - The transformations you apply MUST be accurate and the fields you use MUST be accurate.

# 3. Ensure using the accurate data encoding.
#     - WHEN USING CATEGORICAL VARIABLES FOR COLOR (`c` PARAMETER), YOU MUST CONVERT THEM TO NUMERICAL VALUES (E.G., LABEL ENCODING) OR MAP THEM TO A VALID COLOR PALETTE:
#         - `'c' argument must be a color, a sequence of colors, or a sequence of numbers` (Caused by passing string categories directly to the `c=` argument in scatter plots — must convert to `.cat.codes` and use a color map)
#         - Use a **color map** when working with continuous variables in scatter plots or other charts to ensure clarity.

# 4. Ensure using the right aesthetics:
#     - ENSURE x-axis are legible. (e.g., rotated axis if needed)
#     - Ensure the used colors are clear to represent the varibles
#     - Ensure the consistency between variables and ledgend.
#     - Never use duplicate colors in the legend.  
#     - Ensure color contrast is sufficient for readability. 

# 5. **The visualization CODE MUST BE FULLY EXECUTABLE and FREE FROM ERRORS:**
#    - It must correctly consider field types and use them appropriately.
#    - Avoid placeholders or incomplete logic.
#    - Recommendation:
#         - Using plt.cm.viridis(np.linspace(0, 1, n_years)) to create a list of colors instead of a colormap object
#         - Properly setting the title with g.fig.suptitle() instead of plt.title()
#    - **Ensure the code DOES NOT cause common errors such as:**
#         - `UserWarning: Could not infer format...` (Caused by ambiguous datetime parsing – ensure format is specified when using datetime columns)
#         - Not **Using incorrect parameters in Seaborn/Matplotlib functions** (e.g., `ha="right"` in `tick_params`) → Should Move to the correct function (e.g., `plt.xticks(rotation=45, ha="right")`).

# 6. **Planning Requirement:**
#     - Before implementing the code, first generate a clear plan outlining:
#         - The most suitable chart type and why.
#         - Required data transformations.
#         - Handling of multiple variables.
#         - Aesthetic considerations (e.g., axis labels, colors, legends).
#     - If you need to construct a new column, what fields you would use, what visualization type you would use, what aesthetics you would use, etc.
# """


# Combination
system_prompt = """
You are a highly skilled assistant specializing in writing CORRECT, EFFICIENT, and ERROR-FREE visualization code.

## **Your task:**:
- Given a code template, You MUST complete the template to generate a visualization for given the dataset and the goal described.

## **Key requirements:**
1. The code MUST FOLLOW VISUALIZATION BEST PRACTICES to meet the specificed goal:
    - STRICTLY MATCH THE VISUALIZATION TYPE SPECIFIED IN `{goal.visualization}`.
    - If multiple varibles exist, determine if **a grouped bar chart, stacked bar chart,  or facet grid** is needed.

2. Ensure applying the right Data Transformations:
    - The transformations you apply MUST be accurate and the fields you use MUST be accurate.

3. Ensure using the accurate data encoding.
    - WHEN USING CATEGORICAL VARIABLES FOR COLOR (`c` PARAMETER), YOU MUST CONVERT THEM TO NUMERICAL VALUES (E.G., LABEL ENCODING) OR MAP THEM TO A VALID COLOR PALETTE:
        - `'c' argument must be a color, a sequence of colors, or a sequence of numbers` (Caused by passing string categories directly to the `c=` argument in scatter plots — must convert to `.cat.codes` and use a color map)


4. Ensure using the right aesthetics:
    - ENSURE x-axis are legible. (e.g., rotated axis if needed)
    - Ensure the used colors are clear to represent the varibles
    - Never use duplicate colors in the legend.  
    - Ensure color contrast is sufficient for readability. 

5. **The visualization CODE MUST BE FULLY EXECUTABLE and FREE FROM ERRORS:**
   - It must correctly consider field types and use them appropriately.
   - Avoid placeholders or incomplete logic.
   - If it is a predictive element:
        - Before training the model, a check is added to see if type (the target variable) contains NaN values. If so, those rows are dropped.
   - **Ensure the code DOES NOT cause common errors such as:**
        - `UserWarning: Could not infer format...` (Caused by ambiguous datetime parsing – ensure format is specified when using datetime columns)
        - Not **Using incorrect parameters in Seaborn/Matplotlib functions** (e.g., `ha="right"` in `tick_params`) → Should Move to the correct function (e.g., `plt.xticks(rotation=45, ha="right")`).

6. **Planning Requirement:**
    - Before implementing the code, first generate a clear plan outlining:
        - The most suitable chart type and why.
        - Required data transformations.
        - Handling of multiple variables.
        - Aesthetic considerations (e.g., axis labels, colors, legends).
    - If you need to construct a new column, what fields you would use, what visualization type you would use, what aesthetics you would use, etc.

 ### Reminder: 
    - When visualizing multiple variables (e.g., systolic, diastolic or sit-ups counts, gripForce...), make sure to reshape the data from wide format to long format using pd.melt() before passing it to plotting functions. Functions like seaborn.boxplot() and seaborn.FacetGrid() require data in long format (with variable and value columns) to work correctly.
    - ONLY use .cat accessor with a 'category' dtype
    - When using `palette` in `seaborn.FacetGrid`, DO NOT pass a `ListedColormap` object directly. Instead, convert it to a list of individual colors using: `[cmap(i) for i in range(n)]`, where `n` is the number of unique categories.

"""


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
            {"role": "system", "content": f"The dataset summary is : {summary}, and the visualization type is {goal.visualization} \n\n"},
            library_instructions,
            {"role": "user",
             "content":
             f"Always add a legend with various colors where appropriate. The visualization code MUST only use data fields that exist in the dataset (field_names) or fields that are transformations based on existing field_names). Only use variables that have been defined in the code or are in the dataset summary. You MUST return a FULL PYTHON PROGRAM ENCLOSED IN BACKTICKS ``` that starts with an import statement. DO NOT add any explanation. \n\n THE GENERATED CODE SOLUTION SHOULD BE CREATED BY MODIFYING THE SPECIFIED PARTS OF THE TEMPLATE BELOW \n\n {library_template} \n\n.The FINAL COMPLETED CODE BASED ON THE TEMPLATE above is ... \n\n"}]

        completions: TextGenerationResponse = text_gen.generate(
            messages=messages, config=textgen_config)
        response = [x['content'] for x in completions.text]

        return response
