# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" GLUE processors and helpers """

import logging
import os

from processors.utils_align import DataProcessor, InputExample, InputFeatures
import tensorflow as tf
import numpy as np

logger = logging.getLogger(__name__)


def glue_convert_examples_to_features(examples, tokenizer,
                                      max_length=512,
                                      task=None,
                                      label_list=None,
                                      output_mode=None,
                                      pad_on_left=False,
                                      pad_token=0,
                                      pad_token_segment_id=0,
                                      mask_padding_with_zero=True):
    """
    Loads a data file into a list of ``InputFeatures``

    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length
        task: GLUE task
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``
        pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
        pad_token: Padding token
        pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
        mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
            and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
            actual values)

    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.

    """

    if task is not None:
        processor = glue_processors[task]()
        if label_list is None:
            label_list = processor.get_labels()
            logger.info("Using label list %s for task %s" % (label_list, task))
        if output_mode is None:
            output_mode = glue_output_modes[task]
            logger.info("Using output mode %s for task %s" % (output_mode, task))

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d" % (ex_index))
        inputs = tokenizer.encode_plus(
            example.text_a,
            example.text_b,
            add_special_tokens=True,
            max_length=max_length,
        )
        input_ids, token_type_ids= inputs["input_ids"], inputs["token_type_ids"]
        text_a_len=token_type_ids.count(0)
        text_b_len=len(token_type_ids)-text_a_len
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
            """
            生成对齐Attention
            p a b
            a
            b
            """
            align_mask =[[0 if mask_padding_with_zero else 1] *len(input_ids)]*padding_length\
                        +[[0 if mask_padding_with_zero else 1]*(padding_length+text_a_len)+[1 if mask_padding_with_zero else 0]*text_b_len]*text_a_len\
                        +[[0 if mask_padding_with_zero else 1]*padding_length+[1 if mask_padding_with_zero else 0]*text_a_len+[0 if mask_padding_with_zero else 1]*text_b_len]*text_b_len
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
            align_mask =[[0 if mask_padding_with_zero else 1]*text_a_len+[1 if mask_padding_with_zero else 0]*text_b_len+[0 if mask_padding_with_zero else 1]*padding_length]*text_a_len\
                        +[[1 if mask_padding_with_zero else 0]*text_a_len+[0 if mask_padding_with_zero else 1]*(text_b_len+padding_length)]*text_b_len \
                        +[[0 if mask_padding_with_zero else 1] * len(input_ids)] * padding_length

        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(len(attention_mask), max_length)
        assert len(align_mask[0]) == max_length, "Error with input length {} vs {}".format(len(align_mask[0]), max_length)
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(len(token_type_ids), max_length)

        if example.label is not None:
            if output_mode == "classification":
                label = label_map[example.label]
            elif output_mode == "regression":
                label = float(example.label)
            else:
                raise KeyError(output_mode)
        else:
            label=None

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            if label is not None:
                logger.info("label: %s (id = %d)" % (example.label, label))

        features.append(
                InputFeatures(input_ids=input_ids,
                              attention_mask=attention_mask,
                              align_mask=align_mask,
                              token_type_ids=token_type_ids,
                              label=label))

    def gen():
        for ex in features:
             yield  ({'input_ids': np.array(ex.input_ids),
                        'attention_mask': ex.attention_mask,
                        'align_mask':ex.align_mask,
                        'token_type_ids': ex.token_type_ids},
                        ex.label)

    return tf.data.Dataset.from_generator(gen,
        ({'input_ids': tf.int32,
            'attention_mask': tf.int32,
            'align_mask':tf.int32,
            'token_type_ids': tf.int32},
            tf.int64),
        ({'input_ids': tf.TensorShape([None]),
            'attention_mask': tf.TensorShape([None]),
            'align_mask': tf.TensorShape([None,None]),
            'token_type_ids': tf.TensorShape([None])},
            tf.TensorShape([])))

class MyProcessor(DataProcessor):
    """Processor for the QQP data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(tensor_dict['idx'].numpy(),
                            tensor_dict['question1'].numpy().decode('utf-8'),
                            tensor_dict['question2'].numpy().decode('utf-8'),
                            str(tensor_dict['label'].numpy()))

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train_merge.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_merge.tsv")), "dev")

    def get_predict_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test_merge.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, line[0])
            try:
                text_a = line[0] if set_type in ['train','dev'] else line[1]
                text_b = line[1] if set_type in ['train','dev'] else line[2]
                label = line[2] if set_type in ['train','dev'] else None
            except IndexError:
                continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

glue_tasks_num_labels = {
    "mytask":2,
}

glue_processors = {
    "mytask":MyProcessor,
}

glue_output_modes = {
    "mytask":"classification",
}

if __name__=="__main__":
    processor=MyProcessor()
    label_list=processor.get_labels()
    output_mode="classification"
    tokenizer=__import__("tokenization_roberta",fromlist=["RobertaTokenizer"])
    tokenizer=tokenizer.RobertaTokenizer.from_pretrained(r"D:\代码\服务器代码中转\transformer\pretrained\robertabase")
    train_example=processor.get_train_examples(r"D:\代码\服务器代码中转\transformer\data")
    dataset = glue_convert_examples_to_features(train_example,
                                            tokenizer,
                                            label_list=label_list,
                                            max_length=64,
                                            output_mode=output_mode,
                                            pad_on_left=False,  # pad on the left for xlnet
                                            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                            pad_token_segment_id=0,
                                            )

    dataset=dataset.repeat().shuffle(1000).batch(32)
    itertor=dataset.make_one_shot_iterator()
    inputs,label=itertor.get_next()
    with tf.Session() as sess:
        for i in range(10):
            print(sess.run([inputs,label]))

