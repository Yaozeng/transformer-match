from optimization import AdamW, WarmupLinearSchedule
from configuration_bert import BertConfig
from modeling_bert6 import BertForSequenceClassification
from tokenization_bert import BertTokenizer

config_class, model_class, tokenizer_class = BertConfig,BertForSequenceClassification,BertTokenizer
config = config_class.from_pretrained("pretrained/cased_L-12_H-768_A-12/bert_config.json",num_labels=2,finetuning_task="qqp")
tokenizer = tokenizer_class.from_pretrained("pretrained/cased_L-12_H-768_A-12/vocab.txt")
model = model_class.from_pretrained("pretrained/bert-base.pt", from_tf=False, config=config)

for n, p in model.named_parameters():
    print(n)
optimizer_grouped_parameters = [
    {'params': [n for n, p in model.named_parameters() if 'classifier'in n or 'linear_transform' in n], 'lr': 1e-3}
]
print(optimizer_grouped_parameters)
#optimizer = AdamW(optimizer_grouped_parameters, lr=3e-5, eps=1e-8)
#scheduler = WarmupLinearSchedule(optimizer, warmup_steps=100, t_total=200)

#print(scheduler.get_lr()[0])