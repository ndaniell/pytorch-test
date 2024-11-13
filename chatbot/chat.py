import torch
from preprocessing import normalize_string, prepare_data
from train import create_word_mappings, load_model, read_pairs_from_file, model_file, pairs_file

def evaluate(input_sentence, word_mappings, encoder, decoder):
    """
    Evaluate a single input sentence and generate a response using the trained model.
    
    Args:
        input_sentence (str): Input text from user
        word_mappings (tuple): Contains mappings between words and indices for input/output vocabularies
        encoder (EncoderRNN): Trained encoder model
        decoder (DecoderRNN): Trained decoder model
        
    Returns:
        str: Generated response text
    """
    input_word2idx, output_word2idx, input_idx2word, output_idx2word = word_mappings
    with torch.no_grad():
        input_words = normalize_string(input_sentence).split()
        
        # Check if all words exist in vocabulary
        unknown_words = [word for word in input_words if word not in input_word2idx]
        if unknown_words:
            return f"Sorry, I don't understand the word(s): {', '.join(unknown_words)}"
            
        # Convert input words to tensor indices
        input_tensor = torch.tensor([[input_word2idx[word]] for word in input_words])
        
        # Initialize encoder hidden state
        encoder_hidden = encoder.init_hidden()
        
        # Run input sequence through encoder
        for ei in range(len(input_words)):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        
        # Start decoder with SOS token
        decoder_input = torch.tensor([[0]])  # SOS token
        decoder_hidden = encoder_hidden
        
        decoded_words = []
        
        # Generate response one word at a time
        for di in range(20):  # Maximum length of response
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            decoded_words.append(output_idx2word[topi.item()])
            
            # Stop if EOS token is generated
            if topi.item() == 1:  # EOS token
                break
                
            decoder_input = topi
        
        return ' '.join(decoded_words)

def chat():
    """
    Main chat loop that:
    1. Loads the trained model and vocabulary
    2. Takes user input
    3. Generates responses using the evaluate() function
    4. Continues until user types 'quit'
    """
    # Load conversation pairs from file, falling back to defaults if file not found
    pairs = read_pairs_from_file(pairs_file)

    # Prepare data by normalizing strings
    pairs = [(normalize_string(p[0]), normalize_string(p[1])) for p in pairs]
    input_lang, output_lang = prepare_data(pairs)

    # Get vocabulary sizes
    input_size = len(input_lang)
    output_size = len(output_lang)

    # Try to load existing model, otherwise initialize new one
    loaded_model = load_model(model_file, input_size, output_size)

    if loaded_model:
        encoder, decoder, input_lang, output_lang = loaded_model
        print("Loaded existing model")
    else:
        raise Exception("No trained model found. Please train the model first by running train.py")

    # Create word to index mappings (needed for inference)
    word_mappings = create_word_mappings(input_lang, output_lang)

    # Main chat loop
    while True:
        user_input = input('You: ')
        if user_input.lower() == 'quit':
            break
            
        response = evaluate(user_input, word_mappings, encoder, decoder)
        print('Bot:', response)


if __name__ == '__main__':
    chat() 