This script contains the instruction code for training a Transformer-based model for various tasks including classification and language modeling. Below are the details of each part:

The transformer.py file consists of the encoder, decoder and the exploration part codes.

The main.py file consists of the training loops, data sets, other functions to run the code. It imports the required classes from utilities.py, dataset.py and transformer.py files. 

In the main.py file, when the code is run, There are three options for part as part1 , part2, part3. Enter the required option to run the specific code.

Part 1: Encoder (Classifier Training)

Run part1 to train a classifier using the Transformer encoder.
The classifier is trained on a fixed number of epochs using the provided training data.
During training, the script prints the loss and test accuracy every epoch.
Additionally, it provides a sanity check and attention map for a sample sentence.


Part 2: Decoder (Language Modeling)

Run part2 to train a language model using the Transformer decoder.
The language model is trained on a fixed number of iterations due to computational constraints.
The script prints the training perplexity for every 100 iterations for the language model on the training set and separate test sets (Obama, Hbush, Wbush).
Additionally, it provides a sanity check and attention map for a sample sentence.
Part 3: Model Exploration for Improvement

Part3 : Exploration

This part includes experimentation with different model architectures like alibi positional encoding, disentangled multi head etc and various hyperparameter tuning to enhance the 
classifier performance. The script prints the loss and test accuracy during training of one of the models.



Important Note:

Before running any part, ensure you have the necessary datasets in the specified directory and on correct device. 


