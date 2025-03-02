
import json
from ntviz.utils import clean_code_snippet
from llmx import TextGenerator, TextGenerationConfig, TextGenerationResponse
from ..scaffold import ChartScaffold


system_prompt = """
You are a helpful assistant highly skilled in providing helpful, structured explanations of visualization of the plot(data: pd.DataFrame) method in the provided code. You divide the code into sections and provide a description of each section and an explanation. The first section should be named "accessibility" and describe the physical appearance of the chart (colors, chart type etc), the goal of the chart, as well the main insights from the chart.
You can explain code across the following 3 dimensions:
1. accessibility: the physical appearance of the chart (colors, chart type etc), the goal of the chart, as well the main insights from the chart.
2. transformation: This should describe the section of the code that applies any kind of data transformation (filtering, aggregation, grouping, null value handling etc)
3. visualization: step by step description of the code that creates or modifies the presented visualization.

"""

format_instructions = """
Your output MUST be perfect JSON in THE FORM OF A VALID LIST of JSON OBJECTS WITH PROPERLY ESCAPED SPECIAL CHARACTERS e.g.,

```[
    {"section": "accessibility", "code": "None", "explanation": ".."}  , {"section": "transformation", "code": "..", "explanation": ".."}  ,  {"section": "visualization", "code": "..", "explanation": ".."}
    ] ```

The code part of the dictionary must come from the supplied code and should cover the explanation. The explanation part of the dictionary must be a string. The section part of the dictionary must be one of "accessibility", "transformation", "visualization" with no repetition. THE LIST MUST HAVE EXACTLY 3 JSON OBJECTS [{}, {}, {}].  THE GENERATED JSON  MUST BE A LIST IE START AND END WITH A SQUARE BRACKET.
"""


class VizExplainer(object):
    """Generate visualizations Explanations given some code"""

    def __init__(
        self,
    ) -> None:
        self.scaffold = ChartScaffold()

    def generate(
            self, code: str,
            textgen_config: TextGenerationConfig, text_gen: TextGenerator, library='seaborn'):
        """Generate a visualization explanation given some code"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "assistant", "content": f"The code to be explained is {code}.\n=======\n"},
            {"role": "user",
             "content": f"{format_instructions}. \n\n. The structured explanation for the code above is \n\n"}
        ]

        completions: TextGenerationResponse = text_gen.generate(
            messages=messages, config=textgen_config)

        completions = [clean_code_snippet(x['content']) for x in completions.text]
        explanations = []

        for completion in completions:
            try:
                exp = json.loads(completion)
                explanations.append(exp)
            except Exception as e:
                print("Error parsing completion", completion, str(e))
        return explanations
