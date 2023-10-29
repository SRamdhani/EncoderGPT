# eGPT

### Contents 
This repository came from a thought to combine the training of GPT and classification after seeing the https://huggingface.co/datasets/SetFit/amazon_polarity Amazon Polarity dataset in HuggingFace Hub. The GPT will train on the user descriptions to then provide a generated text from the title to be fed to a classifier. Hence the eGPT which is encoder GPT or the GPT provides context for the encoder to encode for classification.

### Methodology
The way I have set this up is as follows: a decoder block of transformers represents the GPT that is to be trained on user descriptions. Next a encoder block of transformers is fed with the GPT's generated text using the title as a start point. The loss for the GPT will be user description to predict description while the encoder loss will classification loss (i.e., binary crossentropy).

### Setting up the environment
The package management is done by poetry. 
```
conda create -n egpt
conda activate egpt
conda install pip
pip install poetry
poetry install
# pip install tensorflow # This might be needed depending on your system.
python run.py
```

### Additional Pending Work
- So far the training is sequential. A parallel training paradigm is equally possible (i.e., train the GPT on descriptions and the classifier on the titles).

### Output and viewing the results
PENDING.

### Contact 
If you would like to collaborate or need any help with the code you can reach me at satesh.ramdhani@gmail.com. 