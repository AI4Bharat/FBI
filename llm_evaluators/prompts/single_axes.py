system_prompt = "You are a helpful assistant."
general_schema = """##Task Description
"Open Prompt": Given a short user provided prompt and its response, your task is to evaluate the completion 
with respect to the starting prompt and listed set of metrics. For the metric, you must always
return a score and a justification of the score. Note that, both the starting prompt and its completion are given in English.
##Metrics
{metric}
##Prompt
{prompt}
##Response
{response}
##Format Instructions
{format_instruction}
"""
hallucination = """{
    "name": "hallucinations",
    "description": "Hallucinations assess the extent to which a model's output remains anchored to, and consistent with, the input content provided.
        Text with hallucinations while linguistically fluent, are factually baseless or counterfactual in relation to the input. These hallucinations can
        manifest as additions, omissions, or distortions, and might lead to outputs that are misleading or factually incorrect. This metric serves as
        a check against unwarranted deviations from the ground truth provided in the input. Give a score of 1 if the completion is factually correct and
        grounded in the input, and 0 if the completion contains hallucinations or is factually incorrect.",

}
"""
task_quality = """{
    "name": "task quality",
    "description": "Task Quality gauges the degree to which a model adheres to and executes the specific directives given in the prompt. This metric
        zeroes in exclusively on the fidelity of the model's response to the prompt's instructions. An ideal response not only recognizes the overt commands
        of the prompt but also respects its nuance and subtleties. Give a score between 0 and 2 based on the quality of the completion, with 0 being the
        lowest and 2 being the highest.",
}
"""
content_quality = """{
    "name": "output content quality",
    "description": "Output Content Quality measures the overall caliber of the content generated, factoring in its relevance, clarity, originality,
        and linguistic fluency. High-quality output should not only be grammatically sound but should also convey information in an articulate, coherent,
        and engaging manner without any evidence of plagiarism, redundancy, or artificiality. This metric ensures that the produced content meets the
        expectations of originality, clarity, and contextual relevance in addition to linguistic fluency. Give a score between 0 and 2 based on the quality
        of the completion, with 0 being the lowest and 2 being the highest.",
}
"""

reasoning_accuracy = """{
    "name": "reasoning_accuracy",
       "description": "Reasoning Accuracy assesses the extent to which the model's outputs are logically sound and mathematically correct.
       This metric focuses on the model's ability to apply logical reasoning, solve mathematical problems accurately, and present conclusions that are consistent with
       the principles of deductive and inductive reasoning. It is particularly crucial in tasks that require numerical calculations, data interpretation, or the application
       of mathematical formulas and algorithms. This metric is key in ensuring that outputs not only appear numerically or logically plausible but are factually and
       computationally correct. Give a score of 0 if the completion shows flawed reasoning or incorrect calculations, 1 for partially correct but flawed or incomplete reasoning or calculations,
       and 2 for fully correct and logically sound reasoning and computations."
}
"""
