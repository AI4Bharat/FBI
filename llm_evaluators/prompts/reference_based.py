system_prompt = "You are a helpful assistant."
general_schema = """[System]
Please act as an impartial judge and evaluate the quality of the responses provided by the
AI assistant to the user question displayed below. Your evaluation should consider factors 
such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of
the response. You will be given a reference answer and an Assistant's answer. 
Your job is to evaluate how good the Assistant's answer is with respect to the reference answer. Identify and correct any mistakes. 
Do not allow the length of the responses to influence your evaluation. 
Begin your evaluation by providing a short explanation. Be as objective as
possible. After providing your explanation, please rate the response on a scale of 1 to 10.\n\n
[User Question]\n
{question}\n\n
[Reference Answer]\n
{correct_answer}\n
[The Start of Assistant's Answer]\n
{response}\n
[The End of Assistant's Answer]\n\n
{format_instruction}\n\n"""