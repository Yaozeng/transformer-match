import tensorflow as tf
from tokenization_roberta import RobertaTokenizer
from modeling_tf_roberta import TFRobertaForSequenceClassification
from modeling_roberta2 import RobertaForSequenceClassification
from configuration_roberta import RobertaConfig
from processors.glue import glue_convert_examples_to_features,MyProcessor

# Load dataset, tokenizer, model from pretrained model/vocabulary
processor=MyProcessor()
tokenizer = RobertaTokenizer.from_pretrained(r'D:\代码\服务器代码中转\bert\pretrained')
config=RobertaConfig.from_pretrained(r"D:\代码\服务器代码中转\bert\pretrained\config.json")
model = TFRobertaForSequenceClassification.from_pretrained(r'D:\代码\服务器代码中转\bert\pretrained\roberta-base-tf_model.h5',from_pt=False,config=config)
train_data=processor.get_train_examples(r"D:\代码\服务器代码中转\bert\data\train_merge.tsv")
eval_data=processor.get_dev_examples(r"D:\代码\服务器代码中转\bert\data\dev_merge.tsv")

# Prepare dataset for GLUE as a tf.data.Dataset instance
train_dataset = glue_convert_examples_to_features(train_data, tokenizer, 64, 'mytask')
valid_dataset = glue_convert_examples_to_features(eval_data, tokenizer, 64, 'mytask')
train_dataset = train_dataset.shuffle(100).batch(32).repeat(2)
valid_dataset = valid_dataset.batch(64)

# Prepare training: Compile tf.keras model with optimizer, loss and learning rate schedule 
optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

# Train and evaluate using tf.keras.Model.fit()
history = model.fit(train_dataset, epochs=2, steps_per_epoch=115,
                    validation_data=valid_dataset, validation_steps=7)

# Load the TensorFlow model in PyTorch for inspection
model.save_pretrained('./save/')
pytorch_model = RobertaForSequenceClassification.from_pretrained('./save/', from_tf=True)

# Quickly test a few predictions - MRPC is a paraphrasing task, let's see if our model learned the task
sentence_0 = "This research was consistent with his findings."
sentence_1 = "His findings were compatible with this research."
sentence_2 = "His findings were not compatible with this research."
inputs_1 = tokenizer.encode_plus(sentence_0, sentence_1, add_special_tokens=True, return_tensors='pt')
inputs_2 = tokenizer.encode_plus(sentence_0, sentence_2, add_special_tokens=True, return_tensors='pt')

pred_1 = pytorch_model(**inputs_1)[0].argmax().item()
pred_2 = pytorch_model(**inputs_2)[0].argmax().item()
print("sentence_1 is", "a paraphrase" if pred_1 else "not a paraphrase", "of sentence_0")
print("sentence_2 is", "a paraphrase" if pred_2 else "not a paraphrase", "of sentence_0")
