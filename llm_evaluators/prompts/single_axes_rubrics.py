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
        a check against unwarranted deviations from the ground truth provided in the input. The scoring rubric is described below, with a few possible
        reasons (which might not be exhaustive) for a given score.",
    "scoring": {
        "1": {
            "(a)": "The model's output is strictly aligned with and grounded in the information provided in the input.",
            "(b)": "No evidence of added, omitted, or distorted facts that weren't part of the original content.",
            "(c)": "Maintains the integrity of the original information without any unwarranted extrapolations."
        },
        "0": {
            "(a)": "The output introduces statements, claims, or details that weren't present or implied in the input.",
            "(b)": "Contains counterfactual information that directly conflicts with the input content.",
            "(c)": "Demonstrates unexplained deviations, extrapolations, or interpretations not grounded in the provided data."
        }
    }
}
"""

task_quality = """{
    "name": "task quality",
    "description": "Task Quality gauges the degree to which a model adheres to and executes the specific directives given in the prompt. This metric
        zeroes in exclusively on the fidelity of the model's response to the prompt's instructions. An ideal response not only recognizes the overt commands
        of the prompt but also respects its nuance and subtleties. The scoring rubric is described below, with a few possible reasons (which might not be
        exhaustive) for a given score."
    "scoring": {
        "0": {
            "(a)": "The model disregards the instructions entirely.",
            "(b)": "The output is entirely irrelevant to the prompt.",
            "(c)": "There is a clear disconnect between the user's request and the model's response."
        },
        "1": {
            "(a)": "The model grasps and addresses the main theme or element of the instruction but may miss out on finer details or nuances.",
            "(b)": "There is partial alignment with the prompt, indicating some elements of relevance, but not a complete match.",
            "(c)": "The response might include extraneous details not asked for, or it might omit some requested specifics."
        },
        "2": {
            "(a)": "The model demonstrates a precise understanding and adherence to the prompt's instructions.",
            "(b)": "The output holistically satisfies all aspects of the given directive without any deviation.",
            "(c)": "There's a clear and direct correlation between the user's instruction and the model's response, with no aspect of the
            instruction left unaddressed."
        }
    }
}
"""
content_quality = """{
    "name": "output content quality",
    "description": "Output Content Quality measures the overall caliber of the content generated, factoring in its relevance, clarity, originality,
        and linguistic fluency. High-quality output should not only be grammatically sound but should also convey information in an articulate, coherent,
        and engaging manner without any evidence of plagiarism, redundancy, or artificiality. This metric ensures that the produced content meets the
        expectations of originality, clarity, and contextual relevance in addition to linguistic fluency. The scoring rubric is described below, with a
        few possible reasons (which might not be exhaustive) for a given score.",
    "scoring": {
        "0": {
            "(a)": "The output is in a language different from the intended/requested one.",
            "(b)": "Content appears scraped from the web, giving a plagiarized feel.",
            "(c)": "The output is repetitive or overly redundant.",
            "(d)": "Displays artifacts of poor machine translation."
        },
        "1": {
            "(a)": "The content is generally accurate in terms of grammar and word choice.",
            "(b)": "Sounds unnatural or awkward in the language, lacking smoothness.",
            "(c)": "May have minor discrepancies in content clarity or relevance.",
            "(d)": "Shows traces of generative patterns or repetitiveness, albeit less pronounced than level 0."
        },
        "2": {
            "(a)": "The text shows a high level of originality and authenticity.",
            "(b)": "Demonstrates clear, coherent, and contextually appropriate content.",
            "(c)": "Engages the reader with natural linguistic flow and rhythm.",
            "(d)": "Absence of any noticeable generative artifacts or awkward."
        }
    }
}
"""

reasoning_accuracy = """{
    "name": "reasoning_accuracy",
    "description": "Reasoning Accuracy assesses the extent to which the model's outputs are logically sound and mathematically correct. This metric focuses on the model's ability to apply logical reasoning, 
    solve mathematical problems accurately, and present conclusions that are consistent with the principles of deductive and inductive reasoning. It is particularly crucial in tasks that require numerical 
    calculations, data interpretation, or the application of mathematical formulas and algorithms. This metric is key in ensuring that outputs not only appear numerically or logically plausible but are factually and computationally correct. The scoring rubric is described below, with a few possible reasons (which might not be exhaustive) for a given score.",
    "scoring": {
        "0": {
            "(a)": "The output includes logical fallacies or incorrect reasoning that undermines the argument or conclusion.",
            "(b)": "Mathematical calculations are incorrect, leading to erroneous results.",
            "(c)": "Failure to apply basic principles of logic or mathematics appropriately."
        },
        "1": {
            "(a)": "The reasoning is mostly correct but may contain minor logical inconsistencies or imprecisions in mathematical calculations.",
            "(b)": "Demonstrates a basic understanding of the problem but may overlook deeper or more complex logical implications.",
            "(c)": "Calculations are generally correct but may contain minor errors that do not drastically alter the conclusion."
        },
        "2": {
            "(a)": "Outputs demonstrate clear, correct, and coherent reasoning throughout.",
            "(b)": "Mathematical operations and calculations are accurate and correctly applied to the problem at hand.",
            "(c)": "Effectively synthesizes information and applies logical principles to reach correct conclusions."
        }
    }
}
"""