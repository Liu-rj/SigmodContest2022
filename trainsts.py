"""
This example loads the pre-trained SentenceTransformer model 'nli-distilroberta-base-v2' from the server.
It then fine-tunes this model for some epochs on the STS benchmark dataset.

Note: In this example, you must specify a SentenceTransformer model.
If you want to fine-tune a huggingface/transformers model like bert-base-uncased, see training_nli.py and training_stsbenchmark.py
"""
from torch.utils.data import DataLoader
import math
from sentence_transformers import SentenceTransformer, LoggingHandler, losses, util, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import logging
from datetime import datetime
import csv

# Just some code to print debug information to stdout
# logging.basicConfig(format='%(asctime)s - %(message)s',
#                     datefmt='%Y-%m-%d %H:%M:%S',
#                     level=logging.INFO,
#                     handlers=[LoggingHandler()])
# /print debug information to stdout

# Check if dataset exists. If not, download and extract  it
# sts_dataset_path = 'data/stsbenchmark.tsv.gz'

# if not os.path.exists(sts_dataset_path):
#     util.http_get('https://sbert.net/datasets/stsbenchmark.tsv.gz', sts_dataset_path)

# Read the dataset
# model_name = 'all-MiniLM-L6-v2'
# model_name = 'paraphrase-MiniLM-L3-v2'
# model_name = 'prajjwal1/bert-tiny'
model_name = 'model/mix_base'
print('model:\t', model_name)
train_batch_size = 128
num_epochs = 200
model_save_path = 'model/' + 'sts' + '-' + model_name + '-' + datetime.now().strftime(
    "%Y-%m-%d_%H-%M-%S")

# Load a pre-trained sentence transformer model
model = SentenceTransformer(model_name)

# Convert the dataset to a DataLoader ready for training
logging.info("Read X2 train dataset")

train_samples = []
dev_samples = []
test_samples = []
with open('data/sts_train_x2.csv', 'r') as fin:
    reader = csv.DictReader(fin, delimiter=',')
    for row in reader:
        inp_example = InputExample(
            texts=[row['left'], row['right']], label=float(row['score']))
        train_samples.append(inp_example)
with open('data/sts_dev_x2.csv', 'r') as fin:
    reader = csv.DictReader(fin, delimiter=',')
    for row in reader:
        inp_example = InputExample(
            texts=[row['left'], row['right']], label=float(row['score']))
        dev_samples.append(inp_example)
with open('data/sts_test_x2.csv', 'r') as fin:
    reader = csv.DictReader(fin, delimiter=',')
    for row in reader:
        inp_example = InputExample(
            texts=[row['left'], row['right']], label=float(row['score']))
        test_samples.append(inp_example)

train_dataloader = DataLoader(
    train_samples, shuffle=True, batch_size=train_batch_size)
train_loss = losses.CosineSimilarityLoss(model=model)

# Development set: Measure correlation between cosine score and gold labels
logging.info("Read X2 dev dataset")
evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
    dev_samples, name='sts-dev')

# Configure the training. We skip evaluation in this example
# 10% of train data for warm-up
warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)
logging.info("Warmup-steps: {}".format(warmup_steps))

# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=1000,
          warmup_steps=warmup_steps,
          output_path=model_save_path,
          checkpoint_path='model/checkpoints/sts-mix_base',
          checkpoint_save_steps=2000
          )

##############################################################################
#
# Load the stored model and evaluate its performance on STS benchmark dataset
#
##############################################################################

model = SentenceTransformer(model_save_path)
test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
    test_samples, name='sts-test')
test_evaluator(model, output_path=model_save_path)
