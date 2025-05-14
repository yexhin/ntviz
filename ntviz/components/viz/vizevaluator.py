from PIL import Image
import base64
import io
import json
from ...utils import clean_code_snippet
from llmx import TextGenerator, TextGenerationConfig, TextGenerationResponse

from ntviz.datamodel import Goal

# system_prompt = """
# You are a helpful assistant highly skilled in evaluating the quality of a given visualization code by providing a score from 1 (bad) - 10 (good) while providing clear rationale. YOU MUST CONSIDER VISUALIZATION BEST PRACTICES for each evaluation. Specifically, you can carefully evaluate the code across the following dimensions
# - bugs (bugs):  are there bugs, logic errors, syntax error or typos? Are there any reasons why the code may fail to compile? How should it be fixed? If ANY bug exists, the bug score MUST be less than 5.
# - Data transformation (transformation): Is the data transformed appropriately for the visualization type? E.g., is the dataset appropriated filtered, aggregated, or grouped  if needed?
# - Goal compliance (compliance): how well the code meets the specified visualization goals?
# - Visualization type (type): CONSIDERING BEST PRACTICES, is the visualization type appropriate for the data and intent? Is there a visualization type that would be more effective in conveying insights? If a different visualization type is more appropriate, the score MUST be less than 5.
# - Data encoding (encoding): Is the data encoded appropriately for the visualization type?
# - aesthetics (aesthetics): Are the aesthetics of the visualization appropriate for the visualization type and the data?
# You must provide a score for each of the above dimensions.  Assume that data in chart = plot(data) contains a valid dataframe for the dataset. The `plot` function returns a chart (e.g., matplotlib, seaborn etc object).

# Your OUTPUT MUST BE A VALID JSON LIST OF OBJECTS in the format:

# ```[
# { "dimension":  "bugs",  "score": x , "rationale": " .."}, { "dimension":  "transformation",  "score": x, "rationale": " .."}, { "dimension":  "compliance",  "score": x, "rationale": " .."},{ "dimension":  "type",  "score": x, "rationale": " .."}, { "dimension":  "encoding",  "score": x, "rationale": " .."}, { "dimension":  "aesthetics",  "score": x, "rationale": " .."}
# ]
# ```
# """


system_prompt = """
You are a DATA VISUALIZATION EXPERT which expertise in judging the quality of both a given visualization code and a visualization.
## Your task: you must strictly evaluate and provide a reasonable score from 1(bad) to 10(excellent) acrossing these dimensions:
- Given a visualization code, assume that data in chart = plot(data) contains a valid dataframe for the dataset. The `plot` function returns a chart (e.g., matplotlib, seaborn etc object), you must generate evaluation according to 5 dimensions: 
    - bugs (bugs): 
        - Are there any bugs, logic errors, syntax error or typos?
        - Is the code executable efficiently?
        - Are there reasons the code may fail to execute? How can we fix it?
        - If the code has bugs even a tiny one, its score should be less than 5.
    
    - Data transformation (transformation):
        - Is the transformation appropriate for the used fields in Visualization type? (E.g., is the dataset appropriated filtered, aggregated, or grouped  if needed?)
        - If the dataset transformation is inappropriate, the score MUST be ≤ 3.
    
    - Goal compliance (compliance):
        - Does the visualization code address the Goal optimizely? Why?
        - How well the code meets the specified visualization goal? 
    
    - Visualization type (type): CONSIDERING BEST PRACTICES, 
        - Is the visualization type appropriate for the data and intent? 
        - Is there a visualization type that would be more effective in conveying insights? 
        - If a different visualization type is more appropriate, the score MUST be less than 5.
    
    - Data encoding (encoding): 
        - Is the data encoded appropriately for the visualization type?
        - If the data encoding is wrong partly, the score MUST be less than 7.
    
    - Performance (performance):  
        - Does the code run efficiently without unnecessary operations?
        
- Given the visualization chart, you must evaluate it according to 3 dimensions:
    - Clarity (clarity):
        - Does the visualization effectively convey insights?
        - Are axes, labels, legends, title and annotations clear?
        - Is the size of texts on the visualization chart big enough for users to read?
        - Is the data logically arranged to make patterns easy to identify?
        - If the visualization misrepresents the data (e.g., incorrect axis scaling), the score MUST be less than 4.
        - If labels, axes, or legends overlap or are unclear, the score MUST be less than 5.
        - If the x-axis have more than 20 variables, it will be the potential overlapping. The score MUST be less than 6.
    
    - Aesthetics (aesthetics):
        - Is the visualization visually appealing?
        - Are colors, fonts, and layout suitable?
        - Is there any mismatch between the colors of varibles and ledgend?
        - Is there inconsistency in styling (e.g., too many font types, inconsistent colors)?
        - If there is mismatch, the score should be less than 5.
        - If the color contrast is too poor for readability, the score should be less than 6.
        
    - Readability (readability):
        - Whether any graphic elements out of the canvas?
        - Taking into account factors such as layout, scale and ticks, title and labels, colors, and ease of extracting information.
   
    - Goal compliance (compliance):
        - Does the visualization chart address the Goal efficiently?
        - Could another visualization type or approach communicate the insights more effectively?
        - Does the visualization help users make informed decisions?
        
### Scoring Guidelines
- **1-4 (Poor)**: Significant issues affecting execution, clarity, or insights.  
- **5-6 (Average)**: Some issues but partially usable. Needs improvement.  
- **7-8 (Good)**: Mostly correct but **not optimal**. Could be enhanced.  
- **9-10 (Excellent)**: **Perfectly meets best practices** for the task.  

- IF the rationale does not justify the score properly, deduct points accordingly.

### Average:
- Calculate the avergae score based on the provided scores:        
    - Average (avgerage): mean of the provided scores of each aspect.
    
## Requirements:
- MUST PROVIDE A CLEAR AND CONCISE RATIONALE why you provide the score.
- MUST CONSIDER VISUALIZATION BEST PRACTICES for each evaluation.
- MUST CALCULATE the average score based on the provided score.        


## Output structure MUST BE A VALID JSON LIST OF OBJECTS in the format:
```[
  { "aspect": "code",
    "evaluations": [
      { "dimension": "bugs", "score": x, "rationale": "..." },
      { "dimension": "transformation", "score": x, "rationale": "..." },
      { "dimension": "compliance", "score": x, "rationale": "..." },
      { "dimension": "encoding", "score": x, "rationale": "..." },
      { "dimension": "performance", "score": x, "rationale": "..." }
    ],
    "average": round(sum([score1, score2, ...]) / len(evaluations), 2)
  },
  { "aspect": "visual",
    "evaluations": [
      { "dimension": "clarity", "score": x, "rationale": "..." },
      { "dimension": "aesthetics", "score": x, "rationale": "..." },
      { "dimension": "readability:, "score": x, "rationale": "..." },
      { "dimension": "compliance", "score": x, "rationale": "..." }
    ],
    "average": round(sum([score1, score2, ...]) / len(evaluations), 2)
  }
]
```
"""


class VizEvaluator(object):
    """Generate visualizations Explanations given some code and images"""

    def __init__(
        self,
    ) -> None:
        pass

    def generate(self, 
                 code: str,
                 image: str,
                 goal: Goal,
                 textgen_config: TextGenerationConfig, text_gen: TextGenerator, library='altair'):
        """Generate a visualization explanation given some code"""

        
        # Chuyển đổi base64 thành hình ảnh
        image_base64 = image.split(',')[1] if image.startswith('data:image') else image
        
        if image.startswith('data:image'):
            image_base64 = image.split(',')[1]
        else:
            image_base64 = image

        # Convert base64 thành ảnh PIL
        img_bytes = base64.b64decode(image_base64)
        image_pil = Image.open(io.BytesIO(img_bytes))


        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "assistant",
             "content": f"""
             Generate an evaluation given the goal, code and chart visualization below in {library}, which: 
             - The specified goal is {goal.question}. 
             - The visualization code is {code}.
             - The chart visualization is {image_pil}. 
             Now, evaluate the code and the chart meticulously based on dimensions above according to their aspects. 
             ### RULES:
             - THE SCORE YOU ASSIGN MUST BE STRICTLY REASONABLE, MEANINGFUL AND BACKED BY CLEAR RATIONALE. 
             - MUST CALCULATE THE AVERAGE OF THESE SCORES. 
             The structured evaluation is below .
             """},
        ]
        
        
        
    
        # print(messages)
        completions: TextGenerationResponse = text_gen.generate(
            messages=messages, config=textgen_config)

        completions = [clean_code_snippet(x['content']) for x in completions.text]
        evaluations = []
        for completion in completions:
            try:
                evaluation = json.loads(completion)
                
                # Calculate average of dimensions
                for item in evaluation:
                    scores = [e["score"] for e in item["evaluations"]]
                    item["average"] = round(sum(scores) / len(scores), 2) if scores else 0
                
                evaluations.append(evaluation)
            except Exception as json_error:
                print("Error parsing evaluation data", completion, str(json_error))
        return evaluations
