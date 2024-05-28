system_prompt = "You are a helpful assistant."
general_schema = """[System]
Please act as an impartial judge and evaluate the quality of the responses provided by two
AI assistants to the user question displayed below. Your evaluation should consider
correctness and helpfulness. You will be given assistant A's answer, and assistant B's
answer. Your job is to evaluate which assistant's answer is better. You should
independently solve the user question step-by-step first. Then compare both assistants'
answers with your answer. Identify and correct any mistakes. Avoid any position biases and
ensure that the order in which the responses were presented does not influence your
decision. Do not allow the length of the responses to influence your evaluation. Do not
favor certain names of the assistants. Be as objective as possible. After providing your
explanation, output your final verdict by strictly following this format: "[[A]]" if
assistant A is better, "[[B]]" if assistant B is better, and "[[C]]" for a tie.
[User Question]
{question}
[The Start of Assistant A's Answer]
{correct_answer}
[The End of Assistant A's Answer]
[The Start of Assistant B's Answer]
{perturbed_answer}
[The End of Assistant B's Answer]\n\n
{format_instruction}\n\n"""