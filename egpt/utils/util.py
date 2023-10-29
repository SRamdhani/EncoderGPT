from dataclasses import dataclass, field
from datasets import load_dataset, Dataset
from tqdm import tqdm
import transformers
import os

@dataclass(frozen=False, unsafe_hash=True)
class Utility:
    seq_len: int = field(init=True, default=int, repr=False, compare=False)
    tokenizer: transformers.tokenization_utils_base.BatchEncoding =\
        field(init=True, default=transformers.tokenization_utils_base.BatchEncoding, repr=False, compare=False)
    data_collator: transformers.data.data_collator.DataCollatorWithPadding =\
        field(init=True, default=transformers.data.data_collator.DataCollatorWithPadding, repr=False, compare=False)

    def preprocess(self, examples: slice) -> object:
        model_inputs = self.tokenizer(examples["text"], max_length=self.seq_len, padding=True, truncation=True)
        labels_class = self.tokenizer(examples["title"], max_length=self.seq_len, padding=True, truncation=True)
        model_inputs["label"] = examples["label"]
        model_inputs["labels_class"] = labels_class['input_ids']
        return model_inputs

    @staticmethod
    def customDataCollatorWithPadding(tokenized_dataset: Dataset,
                                      data_collator: transformers.data.data_collator.DataCollatorWithPadding) -> Dataset:
        template = {
            'input_ids': [],
            'label': [],
            'labels_gpt': [],
            'labels_class': []

        }

        for x in tqdm(tokenized_dataset):
            template['label'].append(x['label'])
            template['input_ids'].append(data_collator(x)['input_ids'])
            template['labels_gpt'].append(data_collator(x)['input_ids'])
            x['input_ids'] = x['labels_class']
            template['labels_class'].append(data_collator(x)['input_ids'])

        collated_dataset = Dataset.from_dict(template)
        return collated_dataset

    def loadDataFromHubOrDisk(self, dataset: str) -> tuple:

        if os.path.exists('./train') and os.path.exists('./test'):
            training_collated = Dataset.load_from_disk('./train')
            testing_collated = Dataset.load_from_disk('./test')

        else:
            dataset  = load_dataset(dataset)
            training = dataset['train']
            testing  = dataset['test']

            # For debugging purposes
            # training = load_dataset("SetFit/amazon_polarity", split='train[0:200]')
            tokenized_training = training.map(self.preprocess, batched=True)
            training_collated = Utility.customDataCollatorWithPadding(tokenized_training,
                                                                      data_collator=self.data_collator)

            training_collated.save_to_disk(dataset_path='./train')

            # For debugging purposes.
            # testing = load_dataset("SetFit/amazon_polarity", split='train[0:200]')
            tokenized_testing = testing.map(self.preprocess, batched=True)
            testing_collated = Utility.customDataCollatorWithPadding(tokenized_testing,
                                                                     data_collator=self.data_collator)
            testing_collated.save_to_disk(dataset_path='./test')

        return training_collated, testing_collated