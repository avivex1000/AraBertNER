# Sequence classification for arabic text guide


## Pre-Setup

### Model:
Download the arabert model from hugging face:
https://huggingface.co/aubmindlab/bert-base-arabert

** You can manually download the files or clone the repo locally.

### Training Dataset:
Create a folder named `data` and in it create two csv files named 'train.csv' & 'validation.csv'.
The csvs should both have the following structure:

`text`: The string of text to be classified

`label`: The label of the text

Example csv:
```csv
text,label
I am a text,0
I am another text,1
```


### Installation:

In the project, I've used huggingface transforms, datasets and eval libraries which are supported on python ^3.6.
Specifically, I've tested this out on a version supported on python ^3.7 on my macbook but according to the docs, 
everything should be supported on windows as well.

You can filter the requirements.txt to your needs but notice not to remove the `accelerate` & `scickit-learn` libraries
specifically as they are used in some libraries at runtime but not referenced directly.


## Training the model

In general, the steps to train a classification model on top of the base bert model are as follows:
1. Load the local model to memory.
2. Load the dataset to memory.
3. Configure the training arguments and run the training.

After the training has completed, it will be stored locally to a separate dir which can be loaded and used for inference.

### Prediction

Load the newly trained model and pass it the desired text to classify.
That's it!

