import logging
import torch
from transformers import AutoTokenizer, BertTokenizer
from sklearn.model_selection import train_test_split
import numpy as np 
import pandas as pd 
from src.data.dataset import DisasterTweets

# TODO: handle logging correctly
logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M')

logger=logging.getLogger(__name__)

def main():
    # TODO: MOVE THIS CONFIG!
    # Split test, train
    test_size = 0.2
    # seed
    random_state = 42
    # Set the maximum sequence length for padding
    MAX_LEN = 128

    # load dataset
    ds = DisasterTweets()

    # split train into sentences and labels
    sentences = ds.train.input.values
    labels = ds.train.target.values

    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids = []

    # For every sentence...
    for sent in sentences:
        # `encode` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        encoded_sent = tokenizer.encode(
                            sent,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            max_length = MAX_LEN,          # Truncate all sentences.
                            truncation =False
                            #return_tensors = 'pt',     # Return pytorch tensors.
                    )
        # Add the encoded sentence to the list.
        input_ids.append(encoded_sent)

    # Print sentence 0, now as a list of IDs.
    logger.info(f'Original: {sentences[0]}')
    logger.info(f'Token IDs: {input_ids[0]}')
    logger.info(f'Max sentence length: {max([len(sen) for sen in input_ids])}')

    # Pad our input tokens with value 0 (trailing).
    for tweet in input_ids:
        no_pads = MAX_LEN - len(tweet)  # Number of pads to add to the tweet
        pad_list = no_pads * [0]
        tweet += pad_list

    # Create attention masks
    attention_masks = []

    # For each sentence...
    for sent in input_ids:
        # Create the attention mask.
        #   - If a token ID is 0, then it's padding, set the mask to 0.
        #   - If a token ID is > 0, then it's a real token, set the mask to 1.
        att_mask = [int(token_id > 0) for token_id in sent]
        
        # Store the attention mask for this sentence.
        attention_masks.append(att_mask)

    # Split into train, val
    train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels, 
                                                                random_state=random_state, test_size=test_size)
    # Do the same for the masks.
    train_masks, validation_masks, _, _ = train_test_split(attention_masks, labels,
                                                random_state=random_state, test_size=test_size)

    # to tensors
    train_inputs = torch.tensor(train_inputs)
    validation_inputs = torch.tensor(validation_inputs)

    train_labels = torch.tensor(train_labels)
    validation_labels = torch.tensor(validation_labels)

    train_masks = torch.tensor(train_masks)
    validation_masks = torch.tensor(validation_masks)

    # save to processed
    torch.save(train_inputs, 'data/processed/train_inputs.pt')
    torch.save(validation_inputs, 'data/processed/validation_inputs.pt')

    torch.save(train_labels, 'data/processed/train_labels.pt')
    torch.save(validation_labels, 'data/processed/validation_labels.pt')

    torch.save(train_masks, 'data/processed/train_masks.pt')
    torch.save(validation_masks, 'data/processed/validation_masks.pt')

if __name__=="__main__":
    main()