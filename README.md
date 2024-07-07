# TRANSFORMER IMPLEMENTATION FROM SCRATCH

# File Structure
    - `transformer.py`: Consists of the encoder, decoder, and other improvisation methods.
    - `main.py`: Contains the training loops, datasets, and other functions to run the code.
    - `utilities.py`: Contains the Utilities class for running sanity checks and visualizing attention maps.
    - `tokenizer.py`: Contains the SimpleTokenizer class for building vocabulary and encoding/decoding text.
    - `dataset.py`: Contains dataset classes for text classification and language modeling tasks.
    - `speechesdataset`: This folder consists of the required text and tsv files for the model .




# Usage

In the main.py file, when the code is run, There are three options as part1 , part2, part3. Enter the required option to run the specific code.

1. Part 1: Encoder (Classifier Training)
   
   Run part1 to train a classifier using the Transformer encoder.
   The classifier is trained on a fixed number of epochs using the provided training data.
   During training, the script prints the loss and test accuracy every epoch.
   Additionally, it provides a sanity check and attention map for a sample sentence.

2. Part 2: Decoder (Language Modeling)

   Run part2 to train a language model using the Transformer decoder.
   The language model is trained on a fixed number of iterations due to computational constraints.
   The script prints the training perplexity for every 100 iterations for the language model on the training set and separate test sets (Obama, Hbush, Wbush).
   Additionally, it provides a sanity check and attention map for a sample sentence.
   

3. Part 3: Model Exploration for Improvement

   This part includes experimentation with different model architectures like alibi positional encoding, disentangled multi head etc and various hyperparameter tuning to enhance the 
   classifier performance. The script prints the loss and test accuracy during training of one of the models.



