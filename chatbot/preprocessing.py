import re

def normalize_string(s):
    """
    Normalize a string by converting to lowercase, adding spaces around punctuation,
    and removing non-letter characters except basic punctuation.
    
    Args:
        s (str): Input string to normalize
        
    Returns:
        str: Normalized string with only letters and basic punctuation
    """
    s = s.lower().strip()
    s = re.sub(r"([.!?])", r" \1", s)  # Add spaces around punctuation
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)  # Remove non-letter chars except punctuation
    return s

def prepare_data(pairs):
    """
    Prepare vocabulary lists from conversation pairs.
    
    Args:
        pairs (list): List of (input, output) conversation pairs
        
    Returns:
        tuple: (input_lang, output_lang) containing unique words for each
    """
    # Create word lists
    input_lang = []
    output_lang = []
    
    # Extract words from each conversation pair
    for pair in pairs:
        input_lang.extend(pair[0].split())
        output_lang.extend(pair[1].split())
    
    # Convert to unique word sets
    input_lang = list(set(input_lang))
    output_lang = list(set(output_lang))
    
    return input_lang, output_lang 