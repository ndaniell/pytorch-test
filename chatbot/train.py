import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from model import EncoderRNN, DecoderRNN
from preprocessing import normalize_string, prepare_data


# Try to load existing model, otherwise initialize new one
model_file = "chatbot_model.pt"  # File path for saving/loading model
pairs_file = "conversations.yaml" # YAML file containing conversation pairs

# Training parameters
hidden_size = 256      # Size of hidden layer in encoder/decoder
learning_rate = 0.01   # Learning rate for optimizers
n_epochs = 1000        # Number of training epochs

def read_pairs_from_file(filename):
    """
    Read conversation pairs from a YAML file.
    The YAML file should contain a list of dictionaries with 'input' and 'output' keys.
    Returns a list of (input, output) tuples.
    """
    print(f"Current working directory: {os.getcwd()}")
    try:
        # Try to read and parse YAML file
        with open(filename, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            pairs = []
            # Extract input/output pairs if file has correct format
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and 'input' in item and 'output' in item:
                        pairs.append((item['input'], item['output']))
            return pairs
    except (FileNotFoundError, yaml.YAMLError) as e:
        print(f"Warning: Error reading YAML file {filename}: {str(e)}")
        # Return default pairs if file not found or invalid
        return [
            ("hello", "hi there"),
            ("how are you", "i am good"),
            ("what's your name", "i am a chatbot"), 
            ("goodbye", "bye bye"),
        ]

def save_model(encoder, decoder, input_lang, output_lang, filename):
    """
    Save model state and vocabulary to file.
    Saves encoder/decoder states and input/output vocabularies.
    """
    torch.save({
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'input_lang': input_lang,
        'output_lang': output_lang
    }, filename)

def load_model(filename, input_size, output_size):
    """
    Load model state and vocabulary from file.
    Returns encoder, decoder and vocabularies if file exists, None otherwise.
    """
    if os.path.exists(filename):
        checkpoint = torch.load(filename, weights_only=True)
        encoder = EncoderRNN(input_size, hidden_size)
        decoder = DecoderRNN(hidden_size, output_size)
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        input_lang = checkpoint['input_lang']
        output_lang = checkpoint['output_lang']
        return encoder, decoder, input_lang, output_lang
    return None

def create_word_mappings(input_lang, output_lang):
    """
    Create word-to-index and index-to-word mappings for input and output languages.
    Used for converting between words and tensor indices during training/inference.
    """
    input_word2idx = {word: idx for idx, word in enumerate(input_lang)}
    output_word2idx = {word: idx for idx, word in enumerate(output_lang)}
    input_idx2word = {idx: word for word, idx in input_word2idx.items()}
    output_idx2word = {idx: word for word, idx in output_word2idx.items()}
    return input_word2idx, output_word2idx, input_idx2word, output_idx2word

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    # Initialize hidden state
    encoder_hidden = encoder.init_hidden()
    
    # Zero gradients
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    
    # Get sequence lengths
    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    
    loss = 0
    
    # Encoder - process input sequence
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
    
    # Decoder - generate output sequence
    decoder_input = torch.tensor([[0]])  # Start with SOS token
    decoder_hidden = encoder_hidden
    
    # Teacher forcing - use actual target outputs as next input
    for di in range(target_length):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        loss += criterion(decoder_output, target_tensor[di])
        decoder_input = target_tensor[di]
    
    # Backpropagate loss
    loss.backward()
    
    # Update parameters
    encoder_optimizer.step()
    decoder_optimizer.step()
    
    # Return average loss
    return loss.item() / target_length


def train_model():      
    # Load conversation pairs from file, falling back to defaults if file not found
    pairs = read_pairs_from_file(pairs_file)

    # Normalize and prepare training data
    pairs = [(normalize_string(p[0]), normalize_string(p[1])) for p in pairs]
    input_lang, output_lang = prepare_data(pairs)

    # Get vocabulary sizes
    input_size = len(input_lang)
    output_size = len(output_lang)

    # Initialize encoder and decoder models
    encoder = EncoderRNN(input_size, hidden_size)
    decoder = DecoderRNN(hidden_size, output_size)
    print("Initialized new model")

    # Create word<->index mappings for input/output vocabularies
    input_word2idx, output_word2idx, input_idx2word, output_idx2word = create_word_mappings(input_lang, output_lang)

    # Define loss function and optimizers
    criterion = nn.NLLLoss()
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(n_epochs):
        total_loss = 0
        # Train on each conversation pair
        for pair in pairs:
            input_sentence = pair[0].split()
            target_sentence = pair[1].split()
            
            # Convert words to tensor indices
            input_tensor = torch.tensor([[input_word2idx[word]] for word in input_sentence])
            target_tensor = torch.tensor([[output_word2idx[word]] for word in target_sentence])
            
            # Train on this pair
            loss = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
            total_loss += loss
        
        # Print progress and save checkpoint periodically
        if epoch % 100 == 0:
            print(f'Epoch {epoch} Loss: {total_loss / len(pairs)}')
            save_model(encoder, decoder, input_lang, output_lang, model_file)



if __name__ == '__main__':
    train_model() 

