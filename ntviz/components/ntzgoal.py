import json
import logging
from ntviz.utils import clean_code_snippet
from llmx import TextGenerator
from ntviz.datamodel import Goal, TextGenerationConfig, Persona


# ------------------------------------------ Original ----------------------------------------------
# SYSTEM_INSTRUCTIONS = """
# You are a an experienced data analyst who can generate a given number of insightful GOALS about data, when given a summary of the data, and a specified persona. 
# The VISUALIZATIONS YOU RECOMMEND MUST FOLLOW VISUALIZATION BEST PRACTICES (e.g., must use bar charts instead of pie charts for comparing quantities) AND BE MEANINGFUL (e.g., plot longitude and latitude on maps where appropriate). 
# They must also be relevant to the specified persona. Each goal must include a question, a visualization (THE VISUALIZATION MUST REFERENCE THE EXACT COLUMN FIELDS FROM THE SUMMARY), 
# and a rationale (JUSTIFICATION FOR WHICH dataset FIELDS ARE USED and what we will learn from the visualization). Each goal MUST mention the exact fields from the dataset summary above
# """


# ------------------------------------------- Combination --------------------------------------------
SYSTEM_INSTRUCTIONS = """
You are a DATA ANALYST EXPERT who can efficiently generate a given number of INSIGHTFUL GOALS about data, when given a summary of the data, and a specified persona.

## Goal Structure Requirements
Each goal must include:  
1. **"question"**: A meaningful question that a DATA ANALYST MIGHT ASK ABOUT a given dataset.
2. **"visualization"**: The VISUALIZATION BEST PRACTICE type effectively answers the "question" (THE VISUALIZATION MUST REFERENCE THE EXACT COLUMN FIELDS FROM THE SUMMARY).  
3. **"rationale"**: The comprehensive detailed explainations (JUSTIFICATION FOR WHICH dataset FIELDS ARE USED).  


## Guidelines for Generating a Comprehensive "rationale": 
To ensure non-specialist students understand the value of the analysis, follow this structured reasoning process: 
 
**1. Identify the Data Type of the Fields Used**  
   - Examples: Numerical, Datetime, Categorical, Binary, Time Series, etc.  
   
**2. Visualization Justification**:  
   - Explain WHY this specific visualization type is the best practice for the question
   - Describe how this visualization suits the analytical needs in simple language 

**3. Justify the Choice**  
   - Describe thoughtfully **what insights it reveals** and how it helps answer the question.  
   - Highlight **what we can learn** from the visualization and its practical impact.  

**4. Actionable Outcomes**  
   - Connect insights directly to specific business actions
   - Explain how the findings can drive strategic or operational decisions
   - Describe tangible benefits that come from applying these insights
   
## Let's solve the task step by step to generate meaningful and insightful GOALs

### Step 1: Question Generation
- Based on the summary, examining the field data type to determine the domain (e.g., demographic, social networking, healthcare, e-commerce, marketing, etc.)
- The meaningful question which an experienced data analyst wants to explore from the dataset.
- Questions should be simple enough for non-specialist students to understand, yet analytically valuable

### Step 2: Visualization Selection: 
- The recommended visualization MUST FOLLOW VISUALIZATION BEST PRACTICES but be understandable for non-specialists
- Determine the visualization best practice type to answer the "question". 

### Step 3: Rationale Explanation: 
    - Following the guidlines.

---


## Visualization Best Practices  
To ensure meaningful analysis, the recommended visualizations should follow **best practices**:  
 - For example: 
    - use bar charts instead of pie charts for comparing quantities, 
    - plot longitude and latitude on maps where appropriate, 
    - use histogram and the overlaid with kernel density estimates for distribution,
    - use forecasting line charts with confidence intervals when analyzing trends over time,
    - use line charts for trends with fewer than 10 variables,
    - ...

## **RULES**: 
- Explicitly mention dataset column fields in each goal.
- Primary objective is to generate high-quality, business-relevant, and actionable goals.
- Integrate SIMPLE predictive elements when possible (appropriate for non-specialist data analyst)
- Avoid Redundant or Generic Goals 
   - Example of a bad goal: *"Show total sales."*  
   - Example of a good goal: *"Compare total sales across product categories (`Product_Category`) to identify the best-performing segments and adjust inventory planning."*
"""





# ----------------------------------- Fewshot + Role-based --------------------------------------------------------------
# SYSTEM_INSTRUCTIONS = """
#  You are a an experienced data analyst who can generate a given number of insightful GOALS about data, when given a summary of the data, and a specified persona. 
#  The VISUALIZATIONS YOU RECOMMEND MUST FOLLOW VISUALIZATION BEST PRACTICES. Examples:
# 	- Use bar charts instead of pie charts for comparing quantities,
# 	- Plot longitude and latitude on maps where appropriate,
#     - Use line charts for time series data and trends over time,
#     - Use scatter plots to show relationships between two numerical variables,
#     - Use heatmaps for showing patterns across multiple variables or time periods,
#     - Use histograms for distribution of a single numerical variable,
#     - Use box plots for showing distribution and outliers,
#     - Use stacked bar charts when comparing parts of a whole across categories,
#  Each goal must include a question, a visualization, and a rationale. 
# 	 Example 1:
# 	 - QUESTION: Which product categories perform best in each state, and is there a geographical pattern to be observed.
# 	 - VISUALIZATION: A map visualization using "latitude" and "longitude" coordinates with states outlined, where each state is colored based on the sum of "purchase_amount" for the top-performing "product_category" in that state.
# 	 - RATIONALE: By using "latitude" and "longitude" to plot geographical positions and aggregating "purchase_amount" by "product_category" and "location_state", the Marketing Manager can identify regional preferences. This will help them tailor marketing campaigns to regional preferences and allocate marketing budget more effectively across different states.
	 
# 	 Example 2:
# 	 - QUESTION: Which products are at risk of stockout based on current inventory levels and historical sales rates?
# 	 - VISUALIZATION: A scatter plot with "quantity_in_stock" on the x-axis and average daily "quantity_sold" on the y-axis, with points colored by "product_category" and sized by "unit_price".
# 	 - RATIONALE: By plotting "quantity_in_stock" against the calculated average daily "quantity_sold" (derived from "quantity_sold" and "date"), the Supply Chain Manager can identify items where the ratio of stock to daily sales is concerning. Including "product_category" as color coding and "unit_price" as the point size adds context about the financial impact of potential stockouts. This visualization helps prioritize inventory replenishment based on risk and value.
	 
# 	 Example 3:
# 	 - QUESTION: What patient characteristics are most strongly associated with 30-day readmissions?
# 	 - VISUALIZATION: A box plot showing the distribution of "age", "bmi", "blood_pressure_systolic", and "cholesterol_level", grouped by "readmission_30days" status (Yes/No) and faceted by "department".
# 	 - RATIONALE: Using "readmission_30days" as the grouping variable while examining the distributions of "age", "bmi", "blood_pressure_systolic", and "cholesterol_level" across different "department" values will help the Hospital Administrator identify which patient metrics are most different between readmitted and non-readmitted patients. This insight can help develop targeted interventions for high-risk patients and potentially reduce readmission rates.

#  Each goal MUST mention the exact fields from the dataset summary above.
# """





FORMAT_INSTRUCTIONS = """
THE OUTPUT MUST BE A CODE SNIPPET OF A VALID LIST OF JSON OBJECTS. IT MUST USE THE FOLLOWING FORMAT:

```[
    { "index": 0,  
      "question": "What is the distribution of X", 
      "visualization": "histogram of X", 
      "rationale": "This tells about "},
    {
        "index": 1,
        "question": "Which products have the highest revenue?",
        "visualization": "Bar chart of X by Y",
        "rationale": "Comparing sales performance across X helps ..."
    }..
    ]
```
THE OUTPUT SHOULD ONLY USE THE JSON FORMAT ABOVE.
"""

logger = logging.getLogger("lida")


class GoalExplorer():
    """Generate goals given a summary of data"""

    def __init__(self) -> None:
        pass

    def generate(self, summary: dict, textgen_config: TextGenerationConfig,
                 text_gen: TextGenerator, n=5, persona: Persona = None) -> list[Goal]:
        """Generate goals given a summary of data"""

        user_prompt = f"""The number of GOALS to generate is {n}. The goals should be based on the data summary below, \n\n .
        {summary} \n\n"""

        if not persona:
            persona = Persona(
                persona="A highly skilled data analyst who can come up with complex, insightful goals about data",
                rationale="")

        user_prompt += f"""\n The generated goals SHOULD BE FOCUSED ON THE INTERESTS AND PERSPECTIVE of a '{persona.persona} persona, who is insterested in complex, insightful goals about the data. \n"""

        
        # The messages we send to the llm to generate goals
        messages = [
            {"role": "system", "content": SYSTEM_INSTRUCTIONS},
            {"role": "assistant",
             "content":
             f"{user_prompt}\n\n {FORMAT_INSTRUCTIONS} \n\n. The generated {n} goals are: \n "}]

        result: list[Goal] = text_gen.generate(messages=messages, config=textgen_config)

        try:
            json_string = clean_code_snippet(result.text[0]["content"])
            result = json.loads(json_string)
            # cast each item in the list to a Goal object
            if isinstance(result, dict):
                result = [result]
            result = [Goal(**x) for x in result]
        except json.decoder.JSONDecodeError:
            logger.info(f"Error decoding JSON: {result.text[0]['content']}")
            print(f"Error decoding JSON: {result.text[0]['content']}")
            raise ValueError(
                "The model did not return a valid JSON object while attempting generate goals. Consider using a larger model or a model with higher max token length.")
        return result
