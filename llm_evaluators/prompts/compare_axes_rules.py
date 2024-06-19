system_prompt = "You are a helpful assistant."
general_schema = """You are a helpful assistant in evaluating the quality of the outputs for a given instruction along a given metric.
Your goal is to select the best output for the given instruction across the metric. You will be presented with two outputs along with rules of evaluation across a metric and your task is to evaluate which output is better.
The two outputs are generated by two different AI assistants A and B respectively. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. 
Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants.
Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: "[[A]]" if
assistant A is better, "[[B]]" if assistant B is better, "[[C]]" for a tie, and "[[D]]" if both the answers are unacceptable.
##Metric
{metric}
User Question]
{question}
[The Start of Assistant A's Answer]
{answer_a}
[The End of Assistant A's Answer]
[The Start of Assistant B's Answer]
{answer_b}
[The End of Assistant B's Answer]\n\n
{format_instruction}\n\n
"""

hallucination = """{
    "name": "hallucinations",
    "description": "This metric guides evaluators to choose the best response out of two by comparing their adherence to the input content and absence of hallucinations. The response that best reflects the input without adding, omitting, or distorting facts should be selected. The evaluators should compare the responses based on the following rules:",
    "rules": {
        "1": "Select the response that more accurately reflects the information provided in the input, closely adhering to the original content without introducing new, unrelated information.",
        "2": "Choose the response that contains the least number of factual inaccuracies. If one response introduces facts not present in the input or contradicts the input content, prefer the other response.",
        "3": "Evaluate whether either response adds, omits, or distorts facts that were not part of the original input. Select the response that is free from such hallucinations.",
        "4": "Determine which response maintains the integrity of the original input better. This includes avoiding unwarranted extrapolations or deviations that are not supported by the input.",
        "5": "Consider which response integrates the information more coherently and logically, making it a more plausible continuation or conclusion based on the input provided."
    }
}

"""
task_quality = """{
    "name": "task_quality",
    "description": "This metric guides evaluators to choose the best response out of two by comparing their adherence to the specific directives given in the prompt. The response that best executes the instructions of the prompt, recognizing both its explicit commands and implicit subtleties, should be selected. Evaluators should compare the responses based on the following rules:",
    "rules": {
        "1": "Select the response that demonstrates a closer alignment with the prompt's instructions, showing a clear understanding of the task requirements.",
        "2": "Choose the response that more accurately and fully addresses the main themes and elements of the prompt, including finer details and nuances that may be implied rather than explicitly stated.",
        "3": "Evaluate each response for its relevance to the prompt, preferring the one that avoids extraneous details not asked for and does not omit any requested specifics.",
        "4": "Determine which response maintains a direct and clear correlation between the user's instructions and the output, ensuring that all aspects of the directive are comprehensively addressed.",
        "5": "Consider which response integrates the instructions more coherently and logically, making it a more effective and precise execution of the prompt."
    }
}
"""
content_quality = """{
    "name": "content_quality",
    "description": "This metric guides evaluators to choose the best response out of two by comparing their overall content quality. Content quality is assessed based on relevance, clarity, originality, and linguistic fluency. Evaluators should compare the responses based on the following rules:",
    "rules": {
        "1": "Select the response that is more relevant to the given context or prompt, ensuring that the content directly addresses the requested information or topic.",
        "2": "Choose the response that demonstrates greater clarity and coherence, making it easy for the reader to understand and follow the argument or narrative.",
        "3": "Evaluate each response for its originality, preferring the one that provides unique insights or perspectives without copying or closely mimicking existing content.",
        "4": "Determine which response exhibits better linguistic fluency, including grammatical accuracy, appropriate use of vocabulary, and natural sentence structure.",
        "5": "Assess the engaging quality of each response, selecting the one that maintains the reader's interest more effectively and communicates the message in an appealing manner."
    }
}
"""

reasoning_accuracy = """{

    "name": "reasoning_accuracy",
    "description": "This metric guides evaluators to choose the best response out of two by comparing their reasoning accuracy. Evaluators should assess which response more accurately applies logical reasoning, solves mathematical problems, and presents conclusions consistent with deductive and inductive reasoning principles. The response that is logically sound, mathematically correct, and factually accurate should be selected. Evaluators should compare the responses based on the following rules:",
    "rules": {
        "1": "Select the response that demonstrates more accurate and precise application of logical reasoning, avoiding logical fallacies and maintaining consistency throughout the argument.",
        "2": "Choose the response that solves mathematical problems more accurately, with correct use of formulas, computations, and deductions, especially in contexts requiring numerical analysis.",
        "3": "Evaluate each response for its ability to draw conclusions that are consistent with the principles of both deductive and inductive reasoning, preferring the one that better supports its conclusions with evidence and logical structure.",
        "4": "Determine which response maintains a higher level of factual and computational correctness, particularly in the application of mathematical algorithms and problem-solving techniques.",
        "5": "Consider which response integrates data interpretation more effectively, providing a clear, logical, and mathematically sound analysis of given data or scenarios."
    }
}
"""