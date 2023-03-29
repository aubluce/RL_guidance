#  necessary functions to run pre-trained BERT prediction for RL reward

import os
import copy
from pathlib import Path
import logging
import pandas as pd
import numpy as np
import pickle
import shutil

from transformers import AutoTokenizer
from transformers import (
    AdamW,
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
)

from pytorch_lamb import Lamb
from sklearn.metrics import roc_curve, auc, hamming_loss, accuracy_score

import torch
from torch import Tensor
from torch import nn
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import (
    Dataset,
    TensorDataset,
    DataLoader,
    RandomSampler,
    SequentialSampler,
)
from torch.optim.lr_scheduler import _LRScheduler, Optimizer


import matplotlib.pyplot as plt
from fastprogress.fastprogress import master_bar, progress_bar
from tensorboardX import SummaryWriter

from transformers import (
    WEIGHTS_NAME,
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer)
from packaging import version


class BertForMultiLabelSequenceClassification(BertForSequenceClassification):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.
    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_labels`: the number of classes for the classifier. Default = 2.
    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_labels].
    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, num_labels].

    """

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
        position_ids=None,
        head_mask=None,
    ):

        outputs = self.bert(
            input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
        )
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        # add hidden states and attention if they are here
        outputs = (logits,) + outputs[2:]
        if labels is not None:
            loss_fct = BCEWithLogitsLoss()

            loss = loss_fct(
                logits.view(-1, self.num_labels), labels.view(-1, self.num_labels)
            )
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


PYTORCH_VERSION = version.parse(torch.__version__)
CLASSIFICATION_THRESHOLD: float = 0.5
MODEL_CLASSES = {
    "bert": (
        BertConfig,
        (BertForSequenceClassification, BertForMultiLabelSequenceClassification),
        BertTokenizer,
    )}

try:
    from apex import amp
    IS_AMP_AVAILABLE = True
except ImportError:
    IS_AMP_AVAILABLE = False


class Learner(object):
    def __init__(
        self,
        data,
        model,
        pretrained_model_path,
        output_dir,
        device,
        logger,
        multi_gpu=True,
        is_fp16=True,
        warmup_steps=0,
        fp16_opt_level="O1",
        grad_accumulation_steps=1,
        max_grad_norm=1.0,
        adam_epsilon=1e-8,
        logging_steps=100,
    ):

        if isinstance(output_dir, str):
            output_dir = Path(output_dir)

        self.data = data
        self.model = model
        self.pretrained_model_path = pretrained_model_path
        self.multi_gpu = multi_gpu
        self.is_fp16 = is_fp16
        self.fp16_opt_level = fp16_opt_level
        self.adam_epsilon = adam_epsilon
        self.warmup_steps = warmup_steps
        self.grad_accumulation_steps = grad_accumulation_steps
        self.device = device
        self.logger = logger
        self.layer_groups = None
        self.optimizer = None
        self.n_gpu = 0
        self.max_grad_norm = max_grad_norm
        self.logging_steps = logging_steps
        self.max_steps = -1
        self.weight_decay = 0.0
        self.model_type = data.model_type

        self.output_dir = output_dir

        if self.multi_gpu:
            self.n_gpu = torch.cuda.device_count()

    # Get the optimiser object
    def get_optimizer(self, lr, optimizer_type="lamb"):

        # Prepare optimiser and schedule
        no_decay = ["bias", "LayerNorm.weight"]

        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        if optimizer_type == "lamb":
            optimizer = Lamb(optimizer_grouped_parameters, lr=lr, eps=self.adam_epsilon)
        elif optimizer_type == "adamw":
            optimizer = AdamW(
                optimizer_grouped_parameters, lr=lr, eps=self.adam_epsilon
            )

        return optimizer

    # Get learning rate scheduler
    def get_scheduler(self, optimizer, t_total, schedule_type="warmup_cosine"):

        SCHEDULES = {
            None: get_constant_schedule,
            "none": get_constant_schedule,
            "warmup_cosine": get_cosine_schedule_with_warmup,
            "warmup_constant": get_constant_schedule_with_warmup,
            "warmup_linear": get_linear_schedule_with_warmup,
            "warmup_cosine_hard_restarts": get_cosine_with_hard_restarts_schedule_with_warmup,
        }

        if schedule_type == None or schedule_type == "none":
            return SCHEDULES[schedule_type](optimizer)

        elif schedule_type == "warmup_constant":
            return SCHEDULES[schedule_type](
                optimizer, num_warmup_steps=self.warmup_steps
            )

        else:
            return SCHEDULES[schedule_type](
                optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=t_total,
            )

    def save_model(self, path=None):

        if not path:
            path = self.output_dir / "model_out"

        path.mkdir(exist_ok=True)

        # Convert path to str for save_pretrained calls
        path = str(path)

        torch.cuda.empty_cache()
        # Save a trained model
        model_to_save = (
            self.model.module if hasattr(self.model, "module") else self.model
        )  # Only save the model it-self
        model_to_save.save_pretrained(path)

        # save the tokenizer
        self.data.tokenizer.save_pretrained(path)


class BertClassificationPredictor(object):
    def __init__(
        self,
        model_path,
        label_path,
        multi_label=False,
        model_type="bert",
        use_fast_tokenizer=True,
        do_lower_case=True,
        device=None,
    ):
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.model_path = model_path
        self.label_path = label_path
        self.multi_label = multi_label
        self.model_type = model_type
        self.do_lower_case = do_lower_case
        self.device = device

        # Use auto-tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, use_fast=use_fast_tokenizer
        )

        self.learner = self.get_learner()

    def get_learner(self):
        databunch = BertDataBunch(
            self.label_path,
            self.label_path,
            self.tokenizer,
            train_file=None,
            val_file=None,
            batch_size_per_gpu=32,
            max_seq_length=512,
            multi_gpu=False,
            multi_label=self.multi_label,
            model_type=self.model_type,
            no_cache=True,
        )

        learner = BertLearner.from_pretrained_model(
            databunch,
            self.model_path,
            metrics=[],
            device=self.device,
            logger=None,
            output_dir=None,
            warmup_steps=0,
            multi_gpu=False,
            is_fp16=False,
            multi_label=self.multi_label,
            logging_steps=0,
        )

        return learner

    def predict_batch(self, texts):
        return self.learner.predict_batch(texts)

    def predict(self, text):
        predictions = self.predict_batch([text])[0]
        return predictions


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            labels: (Optional) [string]. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        if isinstance(label, list):
            self.label = label
        elif label:
            self.label = str(label)
        else:
            self.label = None


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def convert_examples_to_features(
    examples,
    label_list,
    max_seq_length,
    tokenizer,
    output_mode="classification",
    cls_token_at_end=False,
    pad_on_left=False,
    cls_token="[CLS]",
    sep_token="[SEP]",
    pad_token=0,
    sequence_a_segment_id=0,
    sequence_b_segment_id=1,
    cls_token_segment_id=1,
    pad_token_segment_id=0,
    mask_padding_with_zero=True,
    logger=None,
):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            if logger:
                logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(str(example.text_a))

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(str(example.text_b))
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[: (max_seq_length - 2)]

        tokens = tokens_a + [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if tokens_b:
            tokens += tokens_b + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = (
                [0 if mask_padding_with_zero else 1] * padding_length
            ) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + (
                [0 if mask_padding_with_zero else 1] * padding_length
            )
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if isinstance(example.label, list):
            label_id = []
            for label in example.label:
                label_id.append(float(label))
        else:
            if example.label is not None:
                label_id = label_map[example.label]
            else:
                label_id = ""

        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_id=label_id,
            )
        )
    return features


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, filename, size=-1):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, filename, size=-1):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, filename, size=-1):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()


class TextProcessor(DataProcessor):
    def __init__(self, data_dir, label_dir):
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.labels = None

    def get_train_examples(
        self, filename="train.csv", text_col="text", label_col="label", size=-1
    ):
        if size == -1:
            data_df = pd.read_csv(os.path.join(self.data_dir, filename))

            return self._create_examples(
                data_df, "train", text_col=text_col, label_col=label_col
            )
        else:
            data_df = pd.read_csv(os.path.join(self.data_dir, filename))
            #             data_df['comment_text'] = data_df['comment_text'].apply(cleanHtml)
            return self._create_examples(
                data_df.sample(size), "train", text_col=text_col, label_col=label_col
            )

    def get_dev_examples(
        self, filename="val.csv", text_col="text", label_col="label", size=-1
    ):

        if size == -1:
            data_df = pd.read_csv(os.path.join(self.data_dir, filename))
            return self._create_examples(
                data_df, "dev", text_col=text_col, label_col=label_col
            )
        else:
            data_df = pd.read_csv(os.path.join(self.data_dir, filename))
            return self._create_examples(
                data_df.sample(size), "dev", text_col=text_col, label_col=label_col
            )

    def get_test_examples(
        self, filename="val.csv", text_col="text", label_col="label", size=-1
    ):
        data_df = pd.read_csv(os.path.join(self.data_dir, filename))
        #         data_df['comment_text'] = data_df['comment_text'].apply(cleanHtml)
        if size == -1:
            return self._create_examples(
                data_df, "test", text_col=text_col, label_col=None
            )
        else:
            return self._create_examples(
                data_df.sample(size), "test", text_col=text_col, label_col=None
            )

    def get_labels(self, filename="labels.csv"):
        """See base class."""
        if self.labels is None:
            self.labels = list(
                pd.read_csv(os.path.join(self.label_dir, filename), header=None)[0]
                .astype("str")
                .values
            )
        return self.labels

    def _create_examples(self, df, set_type, text_col, label_col):
        """Creates examples for the training and dev sets."""
        if label_col is None:
            return list(
                df.apply(
                    lambda row: InputExample(
                        guid=row.index, text_a=str(row[text_col]), label=None
                    ),
                    axis=1,
                )
            )
        else:
            return list(
                df.apply(
                    lambda row: InputExample(
                        guid=row.index,
                        text_a=str(row[text_col]),
                        label=str(row[label_col]),
                    ),
                    axis=1,
                )
            )


class MultiLabelTextProcessor(TextProcessor):
    def _create_examples(self, df, set_type, text_col, label_col):
        def _get_labels(row, label_col):
            if isinstance(label_col, list):
                return list(row[label_col])
            else:
                # create one hot vector of labels
                label_list = self.get_labels()
                labels = [0] * len(label_list)
                # cast with string in case labels are integers
                labels[label_list.index(str(row[label_col]))] = 1
                return labels

        """Creates examples for the training and dev sets."""
        if label_col is None:
            return list(
                df.apply(
                    lambda row: InputExample(
                        guid=row.index, text_a=row[text_col], label=[]
                    ),
                    axis=1,
                )
            )
        else:
            return list(
                df.apply(
                    lambda row: InputExample(
                        guid=row.index,
                        text_a=row[text_col],
                        label=_get_labels(row, label_col),
                    ),
                    axis=1,
                )
            )


class BertDataBunch(object):
    def __init__(
        self,
        data_dir,
        label_dir,
        tokenizer,
        train_file="train.csv",
        val_file="val.csv",
        test_data=None,
        label_file="labels.csv",
        text_col="text",
        label_col="label",
        batch_size_per_gpu=16,
        max_seq_length=512,
        multi_gpu=True,
        multi_label=False,
        backend="nccl",
        model_type="bert",
        logger=None,
        clear_cache=False,
        no_cache=False,
        custom_sampler=None,
    ):

        # just in case someone passes string instead of Path
        if isinstance(data_dir, str):
            data_dir = Path(data_dir)

        if isinstance(label_dir, str):
            label_dir = Path(label_dir)

        if isinstance(tokenizer, str):
            # instantiate the new tokeniser object using the tokeniser name
            tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=True)

        self.tokenizer = tokenizer
        self.data_dir = data_dir
        self.train_file = train_file
        self.val_file = val_file
        self.test_data = test_data
        self.cache_dir = data_dir / "cache"
        self.max_seq_length = max_seq_length
        self.batch_size_per_gpu = batch_size_per_gpu
        self.train_dl = None
        self.val_dl = None
        self.test_dl = None
        self.multi_label = multi_label
        self.n_gpu = 1
        self.no_cache = no_cache
        self.model_type = model_type
        self.output_mode = "classification"
        self.custom_sampler = custom_sampler
        if logger is None:
            logger = logging.getLogger()
        self.logger = logger
        if multi_gpu:
            self.n_gpu = torch.cuda.device_count()

        if clear_cache:
            shutil.rmtree(self.cache_dir, ignore_errors=True)

        if multi_label:
            processor = MultiLabelTextProcessor(data_dir, label_dir)
        else:
            processor = TextProcessor(data_dir, label_dir)

        self.labels = processor.get_labels(label_file)

        if train_file:
            # Train DataLoader
            train_examples = None
            cached_features_file = os.path.join(
                self.cache_dir,
                "cached_{}_{}_{}_{}_{}".format(
                    self.model_type.replace("/", "-"),
                    "train",
                    "multi_label" if self.multi_label else "multi_class",
                    str(self.max_seq_length),
                    os.path.basename(train_file),
                ),
            )

            if os.path.exists(cached_features_file) is False or self.no_cache is True:
                train_examples = processor.get_train_examples(
                    train_file, text_col=text_col, label_col=label_col
                )
            train_dataset = self.get_dataset_from_examples(
                train_examples, "train", no_cache=self.no_cache
            )

            self.train_batch_size = self.batch_size_per_gpu * max(1, self.n_gpu)

            if self.custom_sampler is not None:
                train_sampler = self.custom_sampler
            else:
                train_sampler = RandomSampler(train_dataset)

            self.train_dl = DataLoader(
                train_dataset, sampler=train_sampler, batch_size=self.train_batch_size
            )

        if val_file:
            # Validation DataLoader
            val_examples = None
            cached_features_file = os.path.join(
                self.cache_dir,
                "cached_{}_{}_{}_{}_{}".format(
                    self.model_type.replace("/", "-"),
                    "dev",
                    "multi_label" if self.multi_label else "multi_class",
                    str(self.max_seq_length),
                    os.path.basename(val_file),
                ),
            )

            if os.path.exists(cached_features_file) is False:
                val_examples = processor.get_dev_examples(
                    val_file, text_col=text_col, label_col=label_col
                )

            val_dataset = self.get_dataset_from_examples(
                val_examples, "dev", no_cache=self.no_cache
            )

            # no grads necessary, hence double val batch size
            self.val_batch_size = self.batch_size_per_gpu * 2 * max(1, self.n_gpu)
            val_sampler = SequentialSampler(val_dataset)
            self.val_dl = DataLoader(
                val_dataset, sampler=val_sampler, batch_size=self.val_batch_size
            )

        if test_data:
            # Test set loader for predictions
            test_examples = []
            input_data = []

            for index, text in enumerate(test_data):
                test_examples.append(InputExample(index, text))
                input_data.append({"id": index, "text": text})

            test_dataset = self.get_dataset_from_examples(
                test_examples, "test", is_test=True, no_cache=self.no_cache
            )

            self.test_batch_size = self.batch_size_per_gpu * max(1, self.n_gpu)
            test_sampler = SequentialSampler(test_dataset)
            self.test_dl = DataLoader(
                test_dataset, sampler=test_sampler, batch_size=self.test_batch_size
            )

    def get_dl_from_texts(self, texts):

        test_examples = []
        input_data = []

        for index, text in enumerate(texts):
            test_examples.append(InputExample(index, text, label=None))
            input_data.append({"id": index, "text": text})

        test_dataset = self.get_dataset_from_examples(
            test_examples, "test", is_test=True, no_cache=True
        )

        test_sampler = SequentialSampler(test_dataset)
        return DataLoader(
            test_dataset, sampler=test_sampler, batch_size=self.batch_size_per_gpu
        )

    def save(self, filename="databunch.pkl"):
        tmp_path = self.data_dir / "tmp"
        tmp_path.mkdir(exist_ok=True)
        with open(str(tmp_path / filename), "wb") as f:
            pickle.dump(self, f)

    def get_dataset_from_examples(
        self, examples, set_type="train", is_test=False, no_cache=False
    ):

        if set_type == "train":
            file_name = self.train_file
        elif set_type == "dev":
            file_name = self.val_file
        elif set_type == "test":
            file_name = (
                "test"  # test is not supposed to be a file - just a list of texts
            )

        cached_features_file = os.path.join(
            self.cache_dir,
            "cached_{}_{}_{}_{}_{}".format(
                self.model_type.replace("/", "-"),
                set_type,
                "multi_label" if self.multi_label else "multi_class",
                str(self.max_seq_length),
                os.path.basename(file_name),
            ),
        )

        if os.path.exists(cached_features_file) and no_cache is False:
            self.logger.info(
                "Loading features from cached file %s", cached_features_file
            )
            features = torch.load(cached_features_file)
        else:
            # Create tokenized and numericalized features
            features = convert_examples_to_features(
                examples,
                label_list=self.labels,
                max_seq_length=self.max_seq_length,
                tokenizer=self.tokenizer,
                output_mode=self.output_mode,
                # xlnet has a cls token at the end
                cls_token_at_end=bool(self.model_type in ["xlnet"]),
                cls_token=self.tokenizer.cls_token,
                sep_token=self.tokenizer.sep_token,
                cls_token_segment_id=2 if self.model_type in ["xlnet"] else 0,
                # pad on the left for xlnet
                pad_on_left=bool(self.model_type in ["xlnet"]),
                pad_token_segment_id=4 if self.model_type in ["xlnet"] else 0,
                logger=self.logger,
            )

            # Create folder if it doesn't exist
            if no_cache is False:
                self.cache_dir.mkdir(exist_ok=True)
                self.logger.info(
                    "Saving features into cached file %s", cached_features_file
                )
                torch.save(features, cached_features_file)

        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor(
            [f.input_mask for f in features], dtype=torch.long
        )
        all_segment_ids = torch.tensor(
            [f.segment_ids for f in features], dtype=torch.long
        )

        if is_test is False:  # labels not available for test set
            if self.multi_label:
                all_label_ids = torch.tensor(
                    [f.label_id for f in features], dtype=torch.float
                )
            else:
                all_label_ids = torch.tensor(
                    [f.label_id for f in features], dtype=torch.long
                )

            dataset = TensorDataset(
                all_input_ids, all_input_mask, all_segment_ids, all_label_ids
            )
        else:
            all_label_ids = []
            dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)

        return dataset


def accuracy_multilabel(y_pred: Tensor, y_true: Tensor, sigmoid: bool = True):
    if sigmoid:
        y_pred = y_pred.sigmoid()
    y_pred = y_pred.cpu()
    y_true = y_true.cpu()
    outputs = np.argmax(y_pred, axis=1)
    real_vals = np.argmax(y_true, axis=1)
    return np.mean(outputs.numpy() == real_vals.numpy())


def roc_auc(y_pred: Tensor, y_true: Tensor):
    # ROC-AUC calcualation
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    return roc_auc["micro"]


def load_model(dataBunch, pretrained_path, finetuned_wgts_path, device, multi_label):

    model_type = dataBunch.model_type
    model_state_dict = None

    if torch.cuda.is_available():
        map_location = lambda storage, loc: storage.cuda()
    else:
        map_location = "cpu"

    if finetuned_wgts_path:
        model_state_dict = torch.load(finetuned_wgts_path, map_location=map_location)
    else:
        model_state_dict = None

    if multi_label is True:
        config_class, model_class, _ = MODEL_CLASSES[model_type]

        config = config_class.from_pretrained(
            str(pretrained_path), num_labels=len(dataBunch.labels)
        )

        model = model_class[1].from_pretrained(
            str(pretrained_path), config=config, state_dict=model_state_dict
        )

    return model.to(device)


class BertLearner(Learner):
    @staticmethod
    def from_pretrained_model(
        dataBunch,
        pretrained_path,
        output_dir,
        metrics,
        device,
        logger,
        finetuned_wgts_path=None,
        multi_gpu=True,
        is_fp16=True,
        loss_scale=0,
        warmup_steps=0,
        fp16_opt_level="O1",
        grad_accumulation_steps=1,
        multi_label=False,
        max_grad_norm=1.0,
        adam_epsilon=1e-8,
        logging_steps=100,
        freeze_transformer_layers=False,
    ):
        if is_fp16 and (IS_AMP_AVAILABLE is False):
            logger.debug("Apex not installed. switching off FP16 training")
            is_fp16 = False

        model = load_model(
            dataBunch, pretrained_path, finetuned_wgts_path, device, multi_label
        )

        return BertLearner(
            dataBunch,
            model,
            str(pretrained_path),
            output_dir,
            metrics,
            device,
            logger,
            multi_gpu,
            is_fp16,
            loss_scale,
            warmup_steps,
            fp16_opt_level,
            grad_accumulation_steps,
            multi_label,
            max_grad_norm,
            adam_epsilon,
            logging_steps,
            freeze_transformer_layers,
        )

    def __init__(
        self,
        data: BertDataBunch,
        model: nn.Module,
        pretrained_model_path,
        output_dir,
        metrics,
        device,
        logger,
        multi_gpu=True,
        is_fp16=True,
        loss_scale=0,
        warmup_steps=0,
        fp16_opt_level="O1",
        grad_accumulation_steps=1,
        multi_label=False,
        max_grad_norm=1.0,
        adam_epsilon=1e-8,
        logging_steps=100,
        freeze_transformer_layers=False,
    ):

        super(BertLearner, self).__init__(
            data,
            model,
            pretrained_model_path,
            output_dir,
            device,
            logger,
            multi_gpu,
            is_fp16,
            warmup_steps,
            fp16_opt_level,
            grad_accumulation_steps,
            max_grad_norm,
            adam_epsilon,
            logging_steps,
        )

        # Classification specific attributes
        self.multi_label = multi_label
        self.metrics = metrics

        # LR Finder
        self.history = {"lr": [], "loss": []}
        self.best_loss = None
        self.state_cacher = None

        # Freezing transformer model layers
        if freeze_transformer_layers:
            for name, param in self.model.named_parameters():
                if name.startswith(data.model_type):
                    param.requires_grad = False

    ### Train the model ###
    def fit(
        self,
        epochs,
        lr,
        validate=True,
        return_results=False,
        schedule_type="warmup_cosine",
        optimizer_type="lamb",
    ):
        results_val = []
        tensorboard_dir = self.output_dir / "tensorboard"
        tensorboard_dir.mkdir(exist_ok=True)

        # Train the model
        tb_writer = SummaryWriter(tensorboard_dir)

        train_dataloader = self.data.train_dl
        if self.max_steps > 0:
            t_total = self.max_steps
            self.epochs = (
                self.max_steps // len(train_dataloader) // self.grad_accumulation_steps
                + 1
            )
        else:
            t_total = len(train_dataloader) // self.grad_accumulation_steps * epochs

        # Prepare optimiser
        optimizer = self.get_optimizer(lr, optimizer_type=optimizer_type)

        # get the base model if its already wrapped around DataParallel
        if hasattr(self.model, "module"):
            self.model = self.model.module

        if self.is_fp16:
            self.model, optimizer = amp.initialize(
                self.model, optimizer, opt_level=self.fp16_opt_level
            )

        # Get scheduler
        scheduler = self.get_scheduler(
            optimizer, t_total=t_total, schedule_type=schedule_type
        )

        # Parallelize the model architecture
        if self.multi_gpu is True:
            self.model = torch.nn.DataParallel(self.model)

        # Start Training
        self.logger.info("***** Running training *****")
        self.logger.info("  Num examples = %d", len(train_dataloader.dataset))
        self.logger.info("  Num Epochs = %d", epochs)
        self.logger.info(
            "  Total train batch size (w. parallel, distributed & accumulation) = %d",
            self.data.train_batch_size * self.grad_accumulation_steps,
        )
        self.logger.info(
            "  Gradient Accumulation steps = %d", self.grad_accumulation_steps
        )
        self.logger.info("  Total optimization steps = %d", t_total)

        global_step = 0
        epoch_step = 0
        tr_loss, logging_loss, epoch_loss = 0.0, 0.0, 0.0
        self.model.zero_grad()
        pbar = master_bar(range(epochs))

        for epoch in pbar:
            epoch_step = 0
            epoch_loss = 0.0
            for step, batch in enumerate(progress_bar(train_dataloader, parent=pbar)):
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "labels": batch[3],
                }

                if self.model_type in ["bert", "xlnet"]:
                    inputs["token_type_ids"] = batch[2]

                outputs = self.model(**inputs)
                loss = outputs[
                    0
                ]  # model outputs are always tuple in pytorch-transformers (see doc)

                if self.n_gpu > 1:
                    loss = (
                        loss.mean()
                    )  # mean() to average on multi-gpu parallel training
                if self.grad_accumulation_steps > 1:
                    loss = loss / self.grad_accumulation_steps

                if self.is_fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        amp.master_params(optimizer), self.max_grad_norm
                    )
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )

                tr_loss += loss.item()
                epoch_loss += loss.item()
                if (step + 1) % self.grad_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()

                    self.model.zero_grad()
                    global_step += 1
                    epoch_step += 1

                    if self.logging_steps > 0 and global_step % self.logging_steps == 0:
                        if validate:
                            # evaluate model
                            results = self.validate()
                            for key, value in results.items():
                                tb_writer.add_scalar(
                                    "eval_{}".format(key), value, global_step
                                )
                                self.logger.info(
                                    "eval_{} after step {}: {}: ".format(
                                        key, global_step, value
                                    )
                                )

                        # Log metrics
                        self.logger.info(
                            "lr after step {}: {}".format(
                                global_step, scheduler.get_lr()[0]
                            )
                        )
                        self.logger.info(
                            "train_loss after step {}: {}".format(
                                global_step,
                                (tr_loss - logging_loss) / self.logging_steps,
                            )
                        )
                        tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                        tb_writer.add_scalar(
                            "loss",
                            (tr_loss - logging_loss) / self.logging_steps,
                            global_step,
                        )

                        logging_loss = tr_loss

            # Evaluate the model against validation set after every epoch
            if validate:
                results = self.validate()
                for key, value in results.items():
                    self.logger.info(
                        "eval_{} after epoch {}: {}: ".format(key, (epoch + 1), value)
                    )
                results_val.append(results)

            # Log metrics
            self.logger.info(
                "lr after epoch {}: {}".format((epoch + 1), scheduler.get_lr()[0])
            )
            self.logger.info(
                "train_loss after epoch {}: {}".format(
                    (epoch + 1), epoch_loss / epoch_step
                )
            )
            self.logger.info("\n")

        tb_writer.close()

        if return_results:
            return global_step, tr_loss / global_step, results_val
        else:
            return global_step, tr_loss / global_step

    ### Evaluate the model
    def validate(self, quiet=False, loss_only=False):
        if quiet is False:
            self.logger.info("Running evaluation")
            self.logger.info("  Num examples = %d", len(self.data.val_dl.dataset))
            self.logger.info("  Batch size = %d", self.data.val_batch_size)

        all_logits = None
        all_labels = None

        eval_loss = 0
        nb_eval_steps, nb_eval_examples = 0, 0

        preds = None
        out_label_ids = None

        validation_scores = {metric["name"]: 0.0 for metric in self.metrics}

        iterator = self.data.val_dl if quiet else progress_bar(self.data.val_dl)

        for step, batch in enumerate(iterator):
            self.model.eval()
            batch = tuple(t.to(self.device) for t in batch)

            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "labels": batch[3],
                }

                if self.model_type in ["bert", "xlnet"]:
                    inputs["token_type_ids"] = batch[2]

                outputs = self.model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()

            nb_eval_steps += 1
            nb_eval_examples += inputs["input_ids"].size(0)

            if all_logits is None:
                all_logits = logits
            else:
                all_logits = torch.cat((all_logits, logits), 0)

            if all_labels is None:
                all_labels = inputs["labels"]
            else:
                all_labels = torch.cat((all_labels, inputs["labels"]), 0)

            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(
                    out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0
                )

        eval_loss = eval_loss / nb_eval_steps

        results = {"loss": eval_loss}

        if loss_only is False:
            # Evaluation metrics
            for metric in self.metrics:
                validation_scores[metric["name"]] = metric["function"](
                    all_logits, all_labels
                )
            results.update(validation_scores)

        return results


    ### Return Predictions ###
    def predict_batch(self, texts=None):

        if texts:
            dl = self.data.get_dl_from_texts(texts)
        elif self.data.test_dl:
            dl = self.data.test_dl
        else:
            dl = self.data.val_dl

        all_logits = None

        self.model.eval()
        for step, batch in enumerate(dl):
            batch = tuple(t.to(self.device) for t in batch)

            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": None}

            if self.model_type in ["bert", "xlnet"]:
                inputs["token_type_ids"] = batch[2]

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs[0]
                if self.multi_label:
                    logits = logits.sigmoid()
                # elif len(self.data.labels) == 2:
                #     logits = logits.sigmoid()
                else:
                    logits = logits.softmax(dim=1)

            if all_logits is None:
                all_logits = logits.detach().cpu().numpy()
            else:
                all_logits = np.concatenate(
                    (all_logits, logits.detach().cpu().numpy()), axis=0
                )

        result_df = pd.DataFrame(all_logits, columns=self.data.labels)
        results = result_df.to_dict(orient="records")

        return [sorted(x.items(), key=lambda kv: kv[1], reverse=True) for x in results]

    def _train_batch(self, train_iter):
        self.model.train()
        total_loss = None  # for late initialization

        self.optimizer.zero_grad()
        for i in range(self.grad_accumulation_steps):
            batch = next(train_iter)

            batch = tuple(t.to(self.device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[3],
            }

            if self.model_type in ["bert", "xlnet"]:
                inputs["token_type_ids"] = batch[2]

            outputs = self.model(**inputs)
            loss = outputs[
                0
            ]  # model outputs are always tuple in pytorch-transformers (see doc)

            if self.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            loss /= self.grad_accumulation_steps

            if self.is_fp16:
                # For minor performance optimization, see also:
                # https://nvidia.github.io/apex/advanced.html#gradient-accumulation-across-iterations
                delay_unscale = ((i + 1) % self.grad_accumulation_steps) != 0

                with amp.scale_loss(
                    loss, self.optimizer, delay_unscale=delay_unscale
                ) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if total_loss is None:
                total_loss = loss
            else:
                total_loss += loss

        self.optimizer.step()

        return total_loss.item()

    def _validate(self, val_iter):
        # Set model to evaluation mode and disable gradient computation
        running_loss = 0
        self.model.eval()
        with torch.no_grad():
            for batch in val_iter:
                batch = tuple(t.to(self.device) for t in batch)

                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "labels": batch[3],
                }

                if self.model_type in ["bert", "xlnet"]:
                    inputs["token_type_ids"] = batch[2]

                batch_size = batch[0].size(0)

                loss = self.model(**inputs)[0]

                running_loss += loss.item() * batch_size

        return running_loss / len(val_iter.dataset)

    def _set_learning_rate(self, new_lrs):
        if not isinstance(new_lrs, list):
            new_lrs = [new_lrs] * len(self.optimizer.param_groups)
        if len(new_lrs) != len(self.optimizer.param_groups):
            raise ValueError(
                "Length of `new_lrs` is not equal to the number of parameter groups "
                + "in the given optimizer"
            )

        for param_group, new_lr in zip(self.optimizer.param_groups, new_lrs):
            param_group["lr"] = new_lr

    def _check_for_scheduler(self):
        for param_group in self.optimizer.param_groups:
            if "initial_lr" in param_group:
                raise RuntimeError("Optimizer already has a scheduler attached to it")

    def plot(self, skip_start=10, skip_end=5, log_lr=True, show_lr=None, ax=None):
        """Plots the learning rate range test.
        Arguments:
            skip_start (int, optional): number of batches to trim from the start.
                Default: 10.
            skip_end (int, optional): number of batches to trim from the start.
                Default: 5.
            log_lr (bool, optional): True to plot the learning rate in a logarithmic
                scale; otherwise, plotted in a linear scale. Default: True.
            show_lr (float, optional): if set, adds a vertical line to visualize the
                specified learning rate. Default: None.
            ax (matplotlib.axes.Axes, optional): the plot is created in the specified
                matplotlib axes object and the figure is not be shown. If `None`, then
                the figure and axes object are created in this method and the figure is
                shown . Default: None.
        Returns:
            The matplotlib.axes.Axes object that contains the plot.
        """

        if skip_start < 0:
            raise ValueError("skip_start cannot be negative")
        if skip_end < 0:
            raise ValueError("skip_end cannot be negative")
        if show_lr is not None and not isinstance(show_lr, float):
            raise ValueError("show_lr must be float")

        # Get the data to plot from the history dictionary. Also, handle skip_end=0
        # properly so the behaviour is the expected
        lrs = self.history["lr"]
        losses = self.history["loss"]
        if skip_end == 0:
            lrs = lrs[skip_start:]
            losses = losses[skip_start:]
        else:
            lrs = lrs[skip_start:-skip_end]
            losses = losses[skip_start:-skip_end]

        # Create the figure and axes object if axes was not already given
        fig = None
        if ax is None:
            fig, ax = plt.subplots()

        # Plot loss as a function of the learning rate
        ax.plot(lrs, losses)
        if log_lr:
            ax.set_xscale("log")
        ax.set_xlabel("Learning rate")
        ax.set_ylabel("Loss")

        if show_lr is not None:
            ax.axvline(x=show_lr, color="red")

        # Show only if the figure was created internally
        if fig is not None:
            plt.show()

        return ax


class LinearLR(_LRScheduler):
    """Linearly increases the learning rate between two boundaries over a number of
    iterations.
    Arguments:
        optimizer (torch.optim.Optimizer): wrapped optimizer.
        end_lr (float): the final learning rate.
        num_iter (int): the number of iterations over which the test occurs.
        last_epoch (int, optional): the index of last epoch. Default: -1.
    """

    def __init__(self, optimizer, end_lr, num_iter, last_epoch=-1):
        self.end_lr = end_lr

        if num_iter <= 1:
            raise ValueError("`num_iter` must be larger than 1")
        self.num_iter = num_iter

        super(LinearLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        # In earlier Pytorch versions last_epoch starts at -1, while in recent versions
        # it starts at 0. We need to adjust the math a bit to handle this. See
        # discussion at: https://github.com/davidtvs/pytorch-lr-finder/pull/42
        if PYTORCH_VERSION < version.parse("1.1.0"):
            curr_iter = self.last_epoch + 1
            r = curr_iter / (self.num_iter - 1)
        else:
            r = self.last_epoch / (self.num_iter - 1)

        return [base_lr + r * (self.end_lr - base_lr) for base_lr in self.base_lrs]


class ExponentialLR(_LRScheduler):
    """Exponentially increases the learning rate between two boundaries over a number of
    iterations.
    Arguments:
        optimizer (torch.optim.Optimizer): wrapped optimizer.
        end_lr (float): the final learning rate.
        num_iter (int): the number of iterations over which the test occurs.
        last_epoch (int, optional): the index of last epoch. Default: -1.
    """

    def __init__(self, optimizer, end_lr, num_iter, last_epoch=-1):
        self.end_lr = end_lr

        if num_iter <= 1:
            raise ValueError("`num_iter` must be larger than 1")
        self.num_iter = num_iter

        super(ExponentialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        # In earlier Pytorch versions last_epoch starts at -1, while in recent versions
        # it starts at 0. We need to adjust the math a bit to handle this. See
        # discussion at: https://github.com/davidtvs/pytorch-lr-finder/pull/42
        if PYTORCH_VERSION < version.parse("1.1.0"):
            curr_iter = self.last_epoch + 1
            r = curr_iter / (self.num_iter - 1)
        else:
            r = self.last_epoch / (self.num_iter - 1)

        return [base_lr * (self.end_lr / base_lr) ** r for base_lr in self.base_lrs]


class StateCacher(object):
    def __init__(self, in_memory, cache_dir=None):
        self.in_memory = in_memory
        self.cache_dir = cache_dir

        if self.cache_dir is None:
            import tempfile

            self.cache_dir = tempfile.gettempdir()
        else:
            if not os.path.isdir(self.cache_dir):
                raise ValueError("Given `cache_dir` is not a valid directory.")

        self.cached = {}

    def store(self, key, state_dict):
        if self.in_memory:
            self.cached.update({key: copy.deepcopy(state_dict)})
        else:
            fn = os.path.join(self.cache_dir, "state_{}_{}.pt".format(key, id(self)))
            self.cached.update({key: fn})
            torch.save(state_dict, fn)

    def retrieve(self, key):
        if key not in self.cached:
            raise KeyError("Target {} was not cached.".format(key))

        if self.in_memory:
            return self.cached.get(key)
        else:
            fn = self.cached.get(key)
            if not os.path.exists(fn):
                raise RuntimeError(
                    "Failed to load state in {}. File doesn't exist anymore.".format(fn)
                )
            state_dict = torch.load(fn, map_location=lambda storage, location: storage)
            return state_dict

    def __del__(self):
        """Check whether there are unused cached files existing in `cache_dir` before
        this instance being destroyed."""

        if self.in_memory:
            return

        for k in self.cached:
            if os.path.exists(self.cached[k]):
                os.remove(self.cached[k])


class DataLoaderIter(object):
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self._iterator = iter(data_loader)

    @property
    def dataset(self):
        return self.data_loader.dataset

    def inputs_labels_from_batch(self, batch_data):
        if not isinstance(batch_data, list) and not isinstance(batch_data, tuple):
            raise ValueError(
                "Your batch type not supported: {}. Please inherit from "
                "`TrainDataLoaderIter` (or `ValDataLoaderIter`) and redefine "
                "`_batch_make_inputs_labels` method.".format(type(batch_data))
            )

        inputs, labels, *_ = batch_data

        return inputs, labels

    def __iter__(self):
        return self

    def __next__(self):
        batch = next(self._iterator)
        return batch


class TrainDataLoaderIter(DataLoaderIter):
    def __init__(self, data_loader, auto_reset=True):
        super().__init__(data_loader)
        self.auto_reset = auto_reset

    def __next__(self):
        try:
            batch = next(self._iterator)
            # inputs, labels = self.inputs_labels_from_batch(batch)
        except StopIteration:
            if not self.auto_reset:
                raise
            self._iterator = iter(self.data_loader)
            batch = next(self._iterator)
            # inputs, labels = self.inputs_labels_from_batch(batch)

        return batch



class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, filename, size=-1):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, filename, size=-1):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, filename, size=-1):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()


class TextProcessor(DataProcessor):
    def __init__(self, data_dir, label_dir):
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.labels = None

    def get_train_examples(
        self, filename="train.csv", text_col="text", label_col="label", size=-1
    ):
        if size == -1:
            data_df = pd.read_csv(os.path.join(self.data_dir, filename))

            return self._create_examples(
                data_df, "train", text_col=text_col, label_col=label_col
            )
        else:
            data_df = pd.read_csv(os.path.join(self.data_dir, filename))
            #             data_df['comment_text'] = data_df['comment_text'].apply(cleanHtml)
            return self._create_examples(
                data_df.sample(size), "train", text_col=text_col, label_col=label_col
            )

    def get_dev_examples(
        self, filename="val.csv", text_col="text", label_col="label", size=-1
    ):

        if size == -1:
            data_df = pd.read_csv(os.path.join(self.data_dir, filename))
            return self._create_examples(
                data_df, "dev", text_col=text_col, label_col=label_col
            )
        else:
            data_df = pd.read_csv(os.path.join(self.data_dir, filename))
            return self._create_examples(
                data_df.sample(size), "dev", text_col=text_col, label_col=label_col
            )

    def get_test_examples(
        self, filename="val.csv", text_col="text", label_col="label", size=-1
    ):
        data_df = pd.read_csv(os.path.join(self.data_dir, filename))
        #         data_df['comment_text'] = data_df['comment_text'].apply(cleanHtml)
        if size == -1:
            return self._create_examples(
                data_df, "test", text_col=text_col, label_col=None
            )
        else:
            return self._create_examples(
                data_df.sample(size), "test", text_col=text_col, label_col=None
            )

    def get_labels(self, filename="labels.csv"):
        """See base class."""
        if self.labels is None:
            self.labels = list(
                pd.read_csv(os.path.join(self.label_dir, filename), header=None)[0]
                .astype("str")
                .values
            )
        return self.labels

    def _create_examples(self, df, set_type, text_col, label_col):
        """Creates examples for the training and dev sets."""
        if label_col is None:
            return list(
                df.apply(
                    lambda row: InputExample(
                        guid=row.index, text_a=str(row[text_col]), label=None
                    ),
                    axis=1,
                )
            )
        else:
            return list(
                df.apply(
                    lambda row: InputExample(
                        guid=row.index,
                        text_a=str(row[text_col]),
                        label=str(row[label_col]),
                    ),
                    axis=1,
                )
            )



class MultiLabelTextProcessor(TextProcessor):
    def _create_examples(self, df, set_type, text_col, label_col):
        def _get_labels(row, label_col):
            if isinstance(label_col, list):
                return list(row[label_col])
            else:
                # create one hot vector of labels
                label_list = self.get_labels()
                labels = [0] * len(label_list)
                # cast with string in case labels are integers
                labels[label_list.index(str(row[label_col]))] = 1
                return labels

        """Creates examples for the training and dev sets."""
        if label_col is None:
            return list(
                df.apply(
                    lambda row: InputExample(
                        guid=row.index, text_a=row[text_col], label=[]
                    ),
                    axis=1,
                )
            )
        else:
            return list(
                df.apply(
                    lambda row: InputExample(
                        guid=row.index,
                        text_a=row[text_col],
                        label=_get_labels(row, label_col),
                    ),
                    axis=1,
                )
            )


class BertDataBunch(object):
    def __init__(
        self,
        data_dir,
        label_dir,
        tokenizer,
        train_file="train.csv",
        val_file="val.csv",
        test_data=None,
        label_file="labels.csv",
        text_col="text",
        label_col="label",
        batch_size_per_gpu=16,
        max_seq_length=512,
        multi_gpu=True,
        multi_label=False,
        backend="nccl",
        model_type="bert",
        logger=None,
        clear_cache=False,
        no_cache=False,
        custom_sampler=None,
        pos_weight=None,
        weight=None
    ):

        # just in case someone passes string instead of Path
        if isinstance(data_dir, str):
            data_dir = Path(data_dir)

        if isinstance(label_dir, str):
            label_dir = Path(label_dir)

        if isinstance(tokenizer, str):
            # instantiate the new tokeniser object using the tokeniser name
            tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=True)

        self.tokenizer = tokenizer
        self.data_dir = data_dir
        self.train_file = train_file
        self.val_file = val_file
        self.test_data = test_data
        self.cache_dir = data_dir / "cache"
        self.max_seq_length = max_seq_length
        self.batch_size_per_gpu = batch_size_per_gpu
        self.train_dl = None
        self.val_dl = None
        self.test_dl = None
        self.multi_label = multi_label
        self.n_gpu = 1
        self.no_cache = no_cache
        self.model_type = model_type
        self.output_mode = "classification"
        self.custom_sampler = custom_sampler
        self.pos_weight = pos_weight
        self.weight = weight
        if logger is None:
            logger = logging.getLogger()
        self.logger = logger
        if multi_gpu:
            self.n_gpu = torch.cuda.device_count()

        if clear_cache:
            shutil.rmtree(self.cache_dir, ignore_errors=True)

        if multi_label:
            processor = MultiLabelTextProcessor(data_dir, label_dir)
        else:
            processor = TextProcessor(data_dir, label_dir)

        self.labels = processor.get_labels(label_file)

        if train_file:
            # Train DataLoader
            train_examples = None
            cached_features_file = os.path.join(
                self.cache_dir,
                "cached_{}_{}_{}_{}_{}".format(
                    self.model_type.replace("/", "-"),
                    "train",
                    "multi_label" if self.multi_label else "multi_class",
                    str(self.max_seq_length),
                    os.path.basename(train_file),
                ),
            )

            if os.path.exists(cached_features_file) is False or self.no_cache is True:
                train_examples = processor.get_train_examples(
                    train_file, text_col=text_col, label_col=label_col
                )

            train_dataset = self.get_dataset_from_examples(
                train_examples, "train", no_cache=self.no_cache
            )

            self.train_batch_size = self.batch_size_per_gpu * max(1, self.n_gpu)

            if self.custom_sampler is not None:
                train_sampler = self.custom_sampler
            else:
                train_sampler = RandomSampler(train_dataset)

            self.train_dl = DataLoader(
                train_dataset, sampler=train_sampler, batch_size=self.train_batch_size
            )

        if val_file:
            # Validation DataLoader
            val_examples = None
            cached_features_file = os.path.join(
                self.cache_dir,
                "cached_{}_{}_{}_{}_{}".format(
                    self.model_type.replace("/", "-"),
                    "dev",
                    "multi_label" if self.multi_label else "multi_class",
                    str(self.max_seq_length),
                    os.path.basename(val_file),
                ),
            )

            if os.path.exists(cached_features_file) is False:
                val_examples = processor.get_dev_examples(
                    val_file, text_col=text_col, label_col=label_col
                )

            val_dataset = self.get_dataset_from_examples(
                val_examples, "dev", no_cache=self.no_cache
            )

            # no grads necessary, hence double val batch size
            self.val_batch_size = self.batch_size_per_gpu * 2 * max(1, self.n_gpu)
            val_sampler = SequentialSampler(val_dataset)
            self.val_dl = DataLoader(
                val_dataset, sampler=val_sampler, batch_size=self.val_batch_size
            )

        if test_data:
            # Test set loader for predictions
            test_examples = []
            input_data = []

            for index, text in enumerate(test_data):
                test_examples.append(InputExample(index, text))
                input_data.append({"id": index, "text": text})

            test_dataset = self.get_dataset_from_examples(
                test_examples, "test", is_test=True, no_cache=self.no_cache
            )

            self.test_batch_size = self.batch_size_per_gpu * max(1, self.n_gpu)
            test_sampler = SequentialSampler(test_dataset)
            self.test_dl = DataLoader(
                test_dataset, sampler=test_sampler, batch_size=self.test_batch_size
            )

    def get_dl_from_texts(self, texts):

        test_examples = []
        input_data = []

        for index, text in enumerate(texts):
            test_examples.append(InputExample(index, text, label=None))
            input_data.append({"id": index, "text": text})

        test_dataset = self.get_dataset_from_examples(
            test_examples, "test", is_test=True, no_cache=True
        )

        test_sampler = SequentialSampler(test_dataset)
        return DataLoader(
            test_dataset, sampler=test_sampler, batch_size=self.batch_size_per_gpu
        )

    def save(self, filename="databunch.pkl"):
        tmp_path = self.data_dir / "tmp"
        tmp_path.mkdir(exist_ok=True)
        with open(str(tmp_path / filename), "wb") as f:
            pickle.dump(self, f)

    def get_dataset_from_examples(
        self, examples, set_type="train", is_test=False, no_cache=False
    ):

        if set_type == "train":
            file_name = self.train_file
        elif set_type == "dev":
            file_name = self.val_file
        elif set_type == "test":
            file_name = (
                "test"  # test is not supposed to be a file - just a list of texts
            )

        cached_features_file = os.path.join(
            self.cache_dir,
            "cached_{}_{}_{}_{}_{}".format(
                self.model_type.replace("/", "-"),
                set_type,
                "multi_label" if self.multi_label else "multi_class",
                str(self.max_seq_length),
                os.path.basename(file_name),
            ),
        )

        if os.path.exists(cached_features_file) and no_cache is False:
            self.logger.info(
                "Loading features from cached file %s", cached_features_file
            )
            features = torch.load(cached_features_file)
        else:
            # Create tokenized and numericalized features
            features = convert_examples_to_features(
                examples,
                label_list=self.labels,
                max_seq_length=self.max_seq_length,
                tokenizer=self.tokenizer,
                output_mode=self.output_mode,
                # xlnet has a cls token at the end
                cls_token_at_end=bool(self.model_type in ["xlnet"]),
                cls_token=self.tokenizer.cls_token,
                sep_token=self.tokenizer.sep_token,
                cls_token_segment_id=2 if self.model_type in ["xlnet"] else 0,
                # pad on the left for xlnet
                pad_on_left=bool(self.model_type in ["xlnet"]),
                pad_token_segment_id=4 if self.model_type in ["xlnet"] else 0,
                logger=self.logger,
            )

            # Create folder if it doesn't exist
            if no_cache is False:
                self.cache_dir.mkdir(exist_ok=True)
                self.logger.info(
                    "Saving features into cached file %s", cached_features_file
                )
                torch.save(features, cached_features_file)

        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor(
            [f.input_mask for f in features], dtype=torch.long
        )
        all_segment_ids = torch.tensor(
            [f.segment_ids for f in features], dtype=torch.long
        )

        if is_test is False:  # labels not available for test set
            if self.multi_label:
                all_label_ids = torch.tensor(
                    [f.label_id for f in features], dtype=torch.float
                )
            else:
                all_label_ids = torch.tensor(
                    [f.label_id for f in features], dtype=torch.long
                )

            dataset = TensorDataset(
                all_input_ids, all_input_mask, all_segment_ids, all_label_ids
            )
        else:
            dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)

        return dataset
