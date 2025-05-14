import json
import logging
from ntviz.utils import clean_code_snippet
from llmx import TextGenerator
from ntviz.datamodel import Goal, TextGenerationConfig, Persona


#The orginal:
# SYSTEM_INSTRUCTIONS = """
# You are a an experienced data analyst who can generate a given number of insightful GOALS about data, when given a summary of the data, and a specified persona. 
# The VISUALIZATIONS YOU RECOMMEND MUST FOLLOW VISUALIZATION BEST PRACTICES (e.g., must use bar charts instead of pie charts for comparing quantities) AND BE MEANINGFUL (e.g., plot longitude and latitude on maps where appropriate). 
# They must also be relevant to the specified persona. Each goal must include a question, a visualization (THE VISUALIZATION MUST REFERENCE THE EXACT COLUMN FIELDS FROM THE SUMMARY), 
# and a rationale (JUSTIFICATION FOR WHICH dataset FIELDS ARE USED and what we will learn from the visualization). Each goal MUST mention the exact fields from the dataset summary above
# """


# Combination
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

# SYSTEM_INSTRUCTIONS = """
# You are a DATA ANALYST EXPERT who can efficiently generate a given number of INSIGHTFUL GOALS about data, when given a summary of the data, and a specified persona.

# **Goal Structure Requirements**:
# Each Goals must include:
# 1. "question": Meaningful questions which the DATA ANALYSTS want to explore from the provided data.
# 2. "visualization": THE BEST PRACTICE VISUALIZATION TYPE of the "question" (e.g. use histogram with overlaid density estimate for distribution,...) (THE VISUALIZATION MUST REFERENCE THE EXACT COLUMN FIELDS FROM THE SUMMARY).
# 3. "rationale": the detailed explainations of why this "visualization" is chosen, what insights it reveals (JUSTIFICATION FOR WHICH dataset FIELDS ARE USED).

# ### **Guidelines for Generating a Comprehensive "rationale"**  
# To ensure meaningful insights, follow this structured reasoning process: 
 
# **1. Identify the Data Type of the Fields Used**  
#    - Examples: Numerical, Datetime, Categorical, Binary, Time Series, etc.  
   
# **2. Visualization Justification**:  
#    - Explain WHY this specific visualization type is the best practice for the question
#    - Describe HOW the visualization's features match the analytical needs

# **3. Justify the Choice**  
#    - Describe thoughtfully **what insights it reveals** and how it helps answer the question.  
#    - Highlight **what we can learn** from the visualization and its practical impact.  

# **4. Actionable Outcomes**  
#    - Connect insights directly to specific business actions
#    - Explain how the findings can drive strategic or operational decisions
#    - Describe tangible benefits that come from applying these insights



# ### **RULES**: 
# - The VISUALIZATIONS YOU RECOMMEND MUST FOLLOW: **VISUALIZATION BEST PRACTICES** AND BE MEANININGFUL:
#   - For example: must use bar charts instead of pie charts for comparing quantities, plot longitude and latitude on maps where appropriate, use histogram and the overlaid with kernel density estimates for distribution,...
# - **Ensure the visualization is also relevant to the specificed persona.**
# - **Explicitly mention dataset column fields in each goal.**
# - **Integrate Predictive Elements**:
#    - At least 1 goal should aim at predicting or forecasting trends based on historical data
#    - Consider trend analysis charts with forecast lines when appropriate

# - **Encourage Comparative Analysis**:
#    - Provide goals that compare performance across periods, regions, product groups
#    - Example: "Compare conversion rates (`Conversion_Rate`) by traffic source (`Traffic_Source`) before and after marketing campaigns to evaluate ROI"
# - **Avoid Redundant or Generic Goals**  
#    - Example of a bad goal: *"Show total sales."*  
#    - Example of a good goal: *"Compare total sales across product categories (`Product_Category`) to identify the best-performing segments and adjust inventory planning."*  
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
