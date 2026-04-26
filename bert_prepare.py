# Method for processing sentences and extracting tokenized embeddings using BERT.
#
# https://github.com/Johan-Verschoor/CL-XD-ABSA/
#
# Originally developed by Lorenz Kremer (2024) for his master's thesis on a similar topic.
# Modified and extended by Johan Verschoor (2025) for the thesis:
# "Enhancing Cross-Domain Aspect-Based Sentiment Analysis with Contrastive Learning."
#
# This script tokenizes input sentences using BERT, replaces aspect terms with "$T$",
# assigns unique indexed tokens, and extracts contextual embeddings for further analysis.

from transformers import BertTokenizer, BertModel
import torch


# Processes a given sentence by tokenizing it with BERT, extracting embeddings,
# and replacing the target term with "$T$". It also assigns unique indices to repeated tokens.
def process_sentence(sentence, target_term, tokenizer, model, token_global_index, max_length=150):
    """
    Tokenizes a sentence using BERT, extracts contextual embeddings,
    replaces the aspect term with "$T$", and assigns unique indices to repeated tokens.

    :param sentence: str
        Input sentence containing "$T$" as a placeholder for the target term.
    :param target_term: str
        The aspect term that will be replaced with "$T$".
    :param tokenizer: BertTokenizer
        Pretrained BERT tokenizer for tokenizing the sentence.
    :param model: BertModel
        Pretrained BERT model for extracting word embeddings.
    :param token_global_index: dict
        Dictionary to track unique occurrences of tokens for indexing.
    :param max_length: int
        Maximum token length for BERT processing (default=150).
    :return: Tuple
        - str: The modified sentence with "$T$".
        - list: Token embeddings as (indexed token, embedding vector).
        - list: Indexed target term occurrences.
    """

    # Tokenize the sentence, adding special tokens and truncating if needed
    input_ids = tokenizer.encode(sentence, add_special_tokens=True, max_length=max_length, truncation=True)

    # Convert input tokens into tensor format for BERT processing
    input_tensor = torch.tensor([input_ids])

    # Run the sentence through BERT without gradient tracking
    with torch.no_grad():
        outputs = model(input_tensor)

    # Extract hidden states (contextual embeddings)
    hidden_states = outputs.last_hidden_state

    # Convert token IDs back to actual words
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    # Tokenize the target term separately for comparison
    target_tokens = tokenizer.tokenize(target_term)
    target_length = len(target_tokens)

    modified_sentence = []
    target_token_indices = []
    token_embeddings = []

    i = 0
    while i < len(tokens):
        token = tokens[i]

        # Skip special tokens
        if token in ['[CLS]', '[SEP]']:
            i += 1
            continue

        # If a match for the target term is found, replace with "$T$"
        if tokens[i:i + target_length] == target_tokens:
            for target_token in target_tokens:
                if target_token in token_global_index:
                    token_global_index[target_token] += 1
                else:
                    token_global_index[target_token] = 0

                indexed_target_token = f"{target_token}_{token_global_index[target_token]}"
                target_token_indices.append(indexed_target_token)

                # Extract corresponding word embedding
                word_embedding = hidden_states[0, i, :].numpy()
                embedding_str = ' '.join(map(str, word_embedding))
                token_embeddings.append((indexed_target_token, embedding_str))

                i += 1  # Move to the next token

            modified_sentence.append("$T$")  # Replace target term with placeholder
        else:
            if token in token_global_index:
                token_global_index[token] += 1
            else:
                token_global_index[token] = 0

            indexed_token = f"{token}_{token_global_index[token]}"
            modified_sentence.append(indexed_token)

            # Store the corresponding word embedding
            word_embedding = hidden_states[0, i, :].numpy()
            embedding_str = ' '.join(map(str, word_embedding))
            token_embeddings.append((indexed_token, embedding_str))

            i += 1

    # Convert the modified sentence list back into a string
    modified_sentence_str = ' '.join(modified_sentence)

    return modified_sentence_str, token_embeddings, target_token_indices


# Reads an input file, processes each sentence, and saves tokenized sentences and embeddings
def process_file(input_filename, output_sentence_filename, output_embedding_filename, max_length=150):
    """
    Reads an input file and processes sentences using BERT to extract tokenized versions and embeddings.
    The processed sentences and their embeddings are then saved to separate files.

    :param input_filename: str
        Path to the input file containing sentences, targets, and sentiment labels.
    :param output_sentence_filename: str
        Path to store processed sentences.
    :param output_embedding_filename: str
        Path to store extracted token embeddings.
    :param max_length: int
        Maximum token length for BERT input (default=150).
    :return: None
    """

    # Load BERT tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    combined_sentences = []
    combined_embeddings = []
    token_global_index = {}

    # Read the input file and process lines in groups of three (sentence, target, sentiment)
    with open(input_filename, 'r', encoding='utf-8') as f:
        lines = f.read().strip().split('\n')

        i = 0
        while i < len(lines):
            sentence = lines[i].strip()
            target_term = lines[i + 1].strip()
            sentiment = lines[i + 2].strip()

            # Process sentence and extract embeddings
            modified_sentence, token_embeddings, target_token_indices = process_sentence(
                sentence.replace("$T$", target_term), target_term, tokenizer, model, token_global_index, max_length
            )

            # Save formatted sentence with indexed tokens
            combined_sentences.append(f"{modified_sentence}\n{' '.join(target_token_indices)}\n{sentiment}")

            # Store token embeddings
            combined_embeddings.extend(token_embeddings)

            i += 3  # Move to the next sentence group

    # Write processed sentences to output file
    with open(output_sentence_filename, 'w') as f:
        f.write('\n'.join(combined_sentences))

    # Write token embeddings to output file
    with open(output_embedding_filename, 'w') as f:
        for token, embedding in combined_embeddings:
            f.write(f"{token} {embedding}\n")


# Process the dataset and save output
input_filename = 'data/programGeneratedData/BERT/book/raw_data_book_2019.txt'
output_sentence_filename = 'data/programGeneratedData/BERT/book/temp/output_sentences.txt'
output_embedding_filename = 'data/programGeneratedData/BERT/book/temp/output_embeddings.txt'

# Process the input file to generate tokenized sentences and embeddings
process_file(input_filename, output_sentence_filename, output_embedding_filename)
