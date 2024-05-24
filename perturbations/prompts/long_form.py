import argparse
import pandas as pd

from langchain_core.output_parsers import JsonOutputParser

from perturbations.utils import create_jsonl, dump_jsonl
from perturbations.parsers import DirectError


def grammar_perturbations(args: argparse.Namespace, testset: pd.DataFrame) -> None:
    """Given a gold answer, introduce grammatical errors.
    
    Args:
        args (argparse.Namespace): Arguments
        testset (pd.DataFrame): Testset
        
    Returns:
        None
    """
    jsons = []
    for _, row in testset.iterrows():
        parser = JsonOutputParser(pydantic_object=DirectError)
        PROMPT =  (
            "Given the following Gold Answer, introduce a grammatical error in the answer.\n"
            "Here are a list of possible errors you can introduce:\n"
            "1. Subject-Verb Agreement Errors\n"
            "2. Pronoun-Antecedent Agreement Errors\n"
            "3. Misplaced Modifiers\n"
            "4. Dangling Modifiers\n"
            "5. Run-On Sentences\n"
            "6. Comma Splices\n"
            "7. Incorrect Use of Apostrophes\n"
            "8. Confusion Between 'Its' and 'It's'\n"
            "9. Incorrect Verb Tense\n"
            "10. Fragmented Sentences\n"
            "11. Double Negatives\n"
            "12. Incorrect Word Usage\n"
            "13. Confusion Between 'Your' and 'You're'\n"
            "14. Confusion Between 'Then' and 'Than'\n"
            "15. Preposition at the End of a Sentence\n"
            "16. Incorrect Pluralization\n\n"
            "Pick one of the errors from the above list and introduce it in the answer.\n"
            "And make this error consistent throughout the answer.\n"
            f"{parser.get_format_instructions()}\n\n"
            "Gold Answer:\n"
            f"{row['answer']}\n\n"
            "Rewrite the full Gold Answer with the introduced error.\n"
        )
        dict_ = create_jsonl(row['cdx'], args.model, PROMPT, args.max_tokens, args.temperature, args.top_p, args.frequency_penalty, args.presence_penalty)
        jsons.append(dict_)
    
    dump_jsonl(args, jsons, f'{args.data_dir}/grammar-errors-temp{args.temperature}.jsonl')
    return


def spelling_perturbations(args: argparse.Namespace, testset: pd.DataFrame) -> None:
    """Given a gold answer, introduce spelling errors.
    
    Args:
        args (argparse.Namespace): Arguments
        testset (pd.DataFrame): Testset

    Returns:
        None
    """
    jsons = []
    for _, row in testset.iterrows():
        parser = JsonOutputParser(pydantic_object=DirectError)
        PROMPT = (
            "Given the following Gold Answer, introduce a spelling error in the answer.\n"
            "Make sure that the introduced error is not immediately obvious and requires some level of knowledge to identify.\n"
            "Please make sure that the introduced error is consistent throughout the answer.\n"
            "Provide an explanation of the introduced error.\n"
            f"{parser.get_format_instructions()}\n\n"
            "Gold Answer:\n"
            f"{row['answer']}\n\n"
            "Rewrite the full Gold Answer with the introduced error.\n"
        )
        dict_ = create_jsonl(row['cdx'], args.model, PROMPT, args.max_tokens, args.temperature, args.top_p, args.frequency_penalty, args.presence_penalty)
        jsons.append(dict_)

    dump_jsonl(args, jsons, f'{args.data_dir}/spelling-errors-temp{args.temperature}.jsonl')
    return


def chronological_perturbations(args: argparse.Namespace, testset: pd.DataFrame) -> None:
    """Given a gold answer, introduce chronological errors.
    
    Args:
        args (argparse.Namespace): Arguments
        testset (pd.DataFrame): Testset
        
    Returns:
        None
    """
    jsons = []
    for _, row in testset.iterrows():
        parser = JsonOutputParser(pydantic_object=DirectError)
        PROMPT = (
            "Given the following Gold Answer, introduce chronological errors in the answer.\n"
            "Mix up the sequence of events, use incorrect time references, switch between past, present, and future tenses inappropriately, and introduce conflicting timelines to create inconsistencies in the order of actions.\n"
            "Provide an explanation of the introduced error.\n"
            f"{parser.get_format_instructions()}\n\n"
            "Gold Answer:\n"
            f"{row['answer']}\n\n"
            "Rewrite the full Gold Answer with the introduced error.\n"
        )
        dict_ = create_jsonl(row['cdx'], args.model, PROMPT, args.max_tokens, args.temperature, args.top_p, args.frequency_penalty, args.presence_penalty)
        jsons.append(dict_)
    
    dump_jsonl(args, jsons, f'{args.data_dir}/chronological-errors-temp{args.temperature}.jsonl')
    return

def chronological_perturbations_v2(args: argparse.Namespace, testset: pd.DataFrame) -> None:
    """Given a gold answer, introduce chronological errors.
    
    Args:
        args (argparse.Namespace): Arguments
        testset (pd.DataFrame): Testset
        
    Returns:
        None
    """
    jsons = []
    for _, row in testset.iterrows():
        parser = JsonOutputParser(pydantic_object=DirectError)
        PROMPT = (
            "Given the following Gold Answer, introduce a sequencing error in the answer.\n"
            "Here are a list of possible sequencing errors you can introduce:\n"
            "1. Shuffling Steps: Rearrange the steps of a process so they are out of order.\n"
            "2. Reverse Causality: Swap the cause and effect statements.\n"
            "3. Misordered Conditional Statements: Incorrectly order 'if-then' statements.\n"
            "4. Misplaced Modifiers: Place modifiers in such a way that they mislead or change the meaning.\n"
            "5. Parallel Structure Error: Mix items that should be in parallel structure.\n"
            "6. Chronological Inconsistency: Mix up the sequence of historical or time-based events.\n"
            "7. Stepwise Process Error: Present steps in an order that is impossible to follow.\n"
            "8. Spatial Disorientation: Misplace spatial instructions or descriptions.\n\n"
            "Pick one of the errors from the above list, or any other similar sequencing error relevant to the given gold answer, and introduce it in the answer.\n"
            "Provide an explanation of the introduced error.\n"
            f"{parser.get_format_instructions()}\n\n"
            "Gold Answer:\n"
            f"{row['answer']}\n\n"
            "Rewrite the full Gold Answer with the introduced error.\n"
        )
        dict_ = create_jsonl(row['cdx'], args.model, PROMPT, args.max_tokens, args.temperature, args.top_p, args.frequency_penalty, args.presence_penalty)
        jsons.append(dict_)
    
    dump_jsonl(args, jsons, f'{args.data_dir}/seq-errors-temp{args.temperature}.jsonl')
    return

def consistency_perturbations(args: argparse.Namespace, testset: pd.DataFrame) -> None:
    """Given a gold answer, introduce inconsistencies in the answer.
    
    Args:
        args (argparse.Namespace): Arguments
        testset (pd.DataFrame): Testset
    
    Returns:
        None
    """
    jsons = []
    for _, row in testset.iterrows():
        parser = JsonOutputParser(pydantic_object=DirectError)
        # removed: conflicting information, contradictory facts, and
        PROMPT = (
            "Given the following Gold Answer, introduce inconsistencies in the answer.\n"
            "Introduce logical inconsistencies to create errors in the answer.\n"
            "Provide an explanation of the introduced error.\n"
            f"{parser.get_format_instructions()}\n\n"
            "Gold Answer:\n"
            f"{row['answer']}\n\n"
            "Rewrite the full Gold Answer with the introduced error.\n"
        )
        dict_ = create_jsonl(row['cdx'], args.model, PROMPT, args.max_tokens, args.temperature, args.top_p, args.frequency_penalty, args.presence_penalty)
        jsons.append(dict_)

    dump_jsonl(args, jsons, f'{args.data_dir}/consistency-errors-temp{args.temperature}.jsonl')
    return

def consistency_perturbations_v2(args: argparse.Namespace, testset: pd.DataFrame) -> None:
    """Given a gold answer, introduce inconsistencies in the answer.
    
    Args:
        args (argparse.Namespace): Arguments
        testset (pd.DataFrame): Testset
    
    Returns:
        None
    """
    jsons = []
    for _, row in testset.iterrows():
        parser = JsonOutputParser(pydantic_object=DirectError)
        # removed: conflicting information, contradictory facts, and
        PROMPT = (
            "Given the following Gold Answer, introduce a consistency error in the answer.\n"
            "Here are a list of possible consistency errors you can introduce:\n"
            "1. Tone Inconsistency: Start with a formal or polite tone and end with an informal or rude tone.\n"
            "2. Contradictory Statements: Make a statement at the beginning and then contradict it later.\n"
            "   Example: 'Climate change is a critical issue we must address. Climate change is a hoax.'\n"
            "3. Terminology Inconsistency: Use different terms for the same concept or entity within the same answer.\n"
            "   Example: 'The project manager will lead the team. The team leader will assign tasks.'\n"
            "4. Perspective Inconsistency: Switch from one perspective to another (e.g., first person to third person).\n"
            "5. Inconsistent Level of Detail: Provide detailed information at some parts and be overly vague at others.\n"
            "   Example: 'To solve the equation, first, isolate the variable. Then do the math.'\n"
            "6. Inconsistent Naming: Refer to the same person or thing by different names.\n"
            "   Example: 'Dr. Smith presented the findings. Later, John elaborated on the results.'\n"
            "7. Inconsistent Recommendations: Give a recommendation and later suggest the opposite.\n"
            "   Example: 'You should invest in stocks. Avoid investing in the stock market.'\n"
            "8. Inconsistent Stance: Take a position on an issue and then take the opposite stance.\n"
            "   Example: 'It's important to conserve water. Excessive water use is not a big issue.'\n"
            "9. Inconsistent Use of Units: Switch between different units of measurement.\n"
            "10. Inconsistent Logic: Present logical arguments that do not align or follow a coherent line of reasoning.\n"
            "    Example: 'Exercise is essential for health. People who exercise are not necessarily healthy.'\n\n"
            "Pick one of the errors that suit best from the above list and introduce it in the answer.\n"
            "And make this error consistent throughout the answer.\n\n"
            "Provide an explanation of the introduced error.\n"
            f"{parser.get_format_instructions()}\n\n"
            "Gold Answer:\n"
            f"{row['answer']}\n\n"
            "Rewrite the full Gold Answer with the introduced error.\n"
        )
        dict_ = create_jsonl(row['cdx'], args.model, PROMPT, args.max_tokens, args.temperature, args.top_p, args.frequency_penalty, args.presence_penalty)
        jsons.append(dict_)

    dump_jsonl(args, jsons, f'{args.data_dir}/consistency-errors-temp{args.temperature}.jsonl')
    return


def coherence_perturbations(args: argparse.Namespace, testset: pd.DataFrame) -> None:
    """Given a gold answer, introduce coherence errors.
    
    Args:
        args (argparse.Namespace): Arguments
        testset (pd.DataFrame): Testset
    
    Returns:
        None
    """
    jsons = []
    for _, row in testset.iterrows():
        parser = JsonOutputParser(pydantic_object=DirectError)
        # todo: blend the errors into the answer
        PROMPT = (
            "Given the following Gold Answer, introduce coherence errors in the answer.\n"
            "Introduce disjointed ideas, disconnected thoughts, and fragmented information to create errors in the answer.\n"
            "Provide an explanation of the introduced error.\n"
            f"{parser.get_format_instructions()}\n\n"
            "Gold Answer:\n"
            f"{row['answer']}\n\n"
            "Rewrite the full Gold Answer with the introduced error.\n"
        )
        dict_ = create_jsonl(row['cdx'], args.model, PROMPT, args.max_tokens, args.temperature, args.top_p, args.frequency_penalty, args.presence_penalty)
        jsons.append(dict_)

    dump_jsonl(args, jsons, f'{args.data_dir}/coherence-errors-temp{args.temperature}.jsonl')
    return


def formatting_perturbations(args: argparse.Namespace, testset: pd.DataFrame) -> None:
    """Given a gold answer, remove the formatting in the answers. This is a score invariant perturbation.
    
    Args:
        args (argparse.Namespace): Arguments
        testset (pd.DataFrame): Testset
    
    Returns:
        None
    """
    jsons = []
    for _, row in testset.iterrows():
        parser = JsonOutputParser(pydantic_object=DirectError)
        PROMPT = (
            "Given the following Gold Answer, remove the formatting in the answer.\n"
            "Except newline characters remove any bold, italic, underline, or any other formatting in the answer.\n"
            "Do not change any logical or factual information in the answer. Only change the formatting.\n"
            "Provide an explanation of the introduced error.\n"
            f"{parser.get_format_instructions()}\n\n"
            "Gold Answer:\n"
            f"{row['answer']}\n\n"
            "Rewrite the full Gold Answer with the introduced error.\n"
        )
        dict_ = create_jsonl(row['cdx'], args.model, PROMPT, args.max_tokens, args.temperature, args.top_p, args.frequency_penalty, args.presence_penalty)
        jsons.append(dict_)

    dump_jsonl(args, jsons, f'{args.data_dir}/formatting-errors-temp{args.temperature}.jsonl')
    return


def comprehensiveness_perturbations(args: argparse.Namespace, testset: pd.DataFrame) -> None:
    """Given a Gold Answer, introduce errors that make the answer less comprehensive.
    
    Args:
        args (argparse.Namespace): Arguments
        testset (pd.DataFrame): Testset
    
    Returns:
        None
    """
    jsons = []
    for _, row in testset.iterrows():
        parser = JsonOutputParser(pydantic_object=DirectError)
        # TODO: errors are too short answers
        PROMPT = (
            "Given the following Question and Gold Answer, introduce one of the below errors.\n"
            "1. Lack of Detail\n"
            "2. Vagueness\n"
            "3. Irrelevance\n"
            "4. Incomplete Information\n"
            "5. Poor Structure\n"
            "6. Lack of Examples\n"
            "7. Inaccuracies\n"
            "8. Lack of Context\n"
            "9. Unclear Language\n"
            "10. Absence of Concluding Remarks\n"
            "11. Failure to Address All Parts of the Question\n\n"
            "Pick one of the errors from the above list and introduce it in the answer.\n"
            "Do not drastically change the answer, only introduce an error that makes the answer less comprehensive with respect to the given question.\n"
            "Provide an explanation of the introduced error.\n"
            f"{parser.get_format_instructions()}\n\n"
            "Question:\n"
            f"{row['question']}\n\n"
            "Gold Answer:\n"
            f"{row['answer']}\n\n"
            "Rewrite the full Gold Answer with the introduced error.\n"
        )
        dict_ = create_jsonl(row['cdx'], args.model, PROMPT, args.max_tokens, args.temperature, args.top_p, args.frequency_penalty, args.presence_penalty)
        jsons.append(dict_)

    dump_jsonl(args, jsons, f'{args.data_dir}/comprehensiveness-errors-temp{args.temperature}.jsonl')
    return


def superficial_perturbations(args: argparse.Namespace, testset: pd.DataFrame) -> None:
    """Given a Gold Answer, introduce superficial cues to make it look like LLM generated answers.
    
    Args:
        args (argparse.Namespace): Arguments
        testset (pd.DataFrame): Testset
    
    Returns:
        None
    """
    jsons = []
    for _, row in testset.iterrows():
        parser = JsonOutputParser(pydantic_object=DirectError)
        PROMPT = (
            "Given the following Gold Answer, introduce a factual error in the answers and also introduce superficial cues to make it look like a Language Model generated answer.\n"
            "These could include any of the below errors that do not add any value to the answer.\n"
            "1. Redundancy\n"
            "2. Filler Words and Phrases\n"
            "3. Overly Long Sentences\n"
            "4. Unrelated Information\n"
            "5. Excessive Qualifiers\n"
            "6. Repetitive Phrasing\n"
            "7. Verbose Introductions and Conclusions\n"
            "8. Unnecessary Examples\n"
            "9. Generalizations\n"
            "10. Abstract Language\n"
            "11. Parenthetical Statements\n"
            "12. Unnecessary Citations or References\n\n"
            "Pick one of the errors from the above list and introduce it in the answer.\n"
            "Provide an explanation of the introduced error.\n"
            f"{parser.get_format_instructions()}\n\n"
            "Gold Answer:\n"
            f"{row['answer']}\n\n"
            "Rewrite the full Gold Answer with the introduced error.\n"
        )
        dict_ = create_jsonl(row['cdx'], args.model, PROMPT, args.max_tokens, args.temperature, args.top_p, args.frequency_penalty, args.presence_penalty)
        jsons.append(dict_)

    dump_jsonl(args, jsons, f'{args.data_dir}/superficial-errors-temp{args.temperature}.jsonl')
    return