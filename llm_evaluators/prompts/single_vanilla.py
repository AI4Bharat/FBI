system_prompt = "You are a helpful assistant."
general_schema = """[System]\nPlease act as an impartial judge and evaluate the quality of the response provided by an 
    AI assistant to the user question displayed below. Your evaluation should consider factors 
    such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of
    the response. Be as objective as possible. Please rate the response on a scale of 1 to 10.\n\n
    [User Question]\n
    {question}\n\n
    [The Start of Assistant's Answer]\n
    {correct_answer}\n
    [The End of Assistant's Answer]\n\n
    {format_instruction}\n\n"""