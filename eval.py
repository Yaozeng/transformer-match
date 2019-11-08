from file_utils import WEIGHTS_NAME
from configuration_roberta import RobertaConfig
from modeling_roberta import RobertaForSequenceClassification
from tokenization_roberta import RobertaTokenizer
import torch
import numpy as np
from processors.glue2 import glue_convert_examples_to_features as convert_examples_to_features
from processors.utils2 import DataProcessor, InputExample, InputFeatures
import xlrd

all_count=0
correct=0
results=[]
#paths=["D:\数据/data1.xlsx","D:\数据/data2.xlsx"]
paths=["./data/data1.xlsx","./data/data2.xlsx"]
config_class, model_class, tokenizer_class = RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer
config = config_class.from_pretrained(r"./output_mytask_sfu/checkpoint-500/config.json")
tokenizer = tokenizer_class.from_pretrained(r"./pretrained/robertalarge")
model = model_class.from_pretrained(r"./output_mytask_sfu/checkpoint-500/pytorch_model.bin", from_tf=False,
                                        config=config)
#config = config_class.from_pretrained(r"D:\代码\服务器代码中转\transformer\pretrained\checkpoint-500\config.json")
#tokenizer = tokenizer_class.from_pretrained(r"D:\代码\服务器代码中转\transformer\pretrained\robertalarge")
#model = model_class.from_pretrained(r"D:\代码\服务器代码中转\transformer\pretrained\checkpoint-500\pytorch_model.bin", from_tf=False,config=config)
model=model.cuda()
model.eval()

for path in paths:
    data = xlrd.open_workbook(path)
    table = data.sheets()[0]
    nrows = table.nrows
    for j in range(1, nrows):
        #output =None
        #labels=None
        all_count += 1
        samples = []
        sentences = table.row_values(j)[6].split("|")
        for sentence in sentences:
            sample = []
            sample.append(sentence)
            sample.append(table.row_values(j)[4])
            samples.append(sample)
        examples = []
        for i in range(len(samples)):
            guid = "%s" % (i)
            text_a = samples[i][0].lower()
            text_b = samples[i][1].lower()
            label = str(int(table.row_values(j)[3] >= 0.5))
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        features = convert_examples_to_features(examples,
                                                tokenizer,
                                                label_list=["0", "1"],
                                                max_length=64,
                                                output_mode="classification",
                                                pad_on_left=False,
                                                # pad on the left for xlnet
                                                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                pad_token_segment_id=0,
                                                )
        #for f in features:
            #input_ids = torch.tensor(f.input_ids, dtype=torch.long).unsqueeze(0)
            #attention_mask = torch.tensor(f.attention_mask, dtype=torch.long).unsqueeze(0)
            #align_mask = torch.tensor(f.align_mask, dtype=torch.long).unsqueeze(0)
            #label = torch.tensor(f.label, dtype=torch.long).unsqueeze(0)
            #input_ids = torch.tensor(f.input_ids, dtype=torch.long).unsqueeze(0).cuda()
            #attention_mask = torch.tensor(f.attention_mask, dtype=torch.long).unsqueeze(0).cuda()
            #align_mask = torch.tensor(f.align_mask, dtype=torch.long).unsqueeze(0).cuda()
            #label = torch.tensor(f.label, dtype=torch.long).unsqueeze(0).cuda()
            #print(label.shape)
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long).cuda()
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long).cuda()
        all_align_mask = torch.tensor([f.align_mask for f in features], dtype=torch.long).cuda()
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long).cuda()
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long).cuda()
        outputs = model(input_ids=all_input_ids, attention_mask=all_attention_mask, align_mask=all_align_mask,labels=all_labels)
        logits = outputs[1]
        #print(logits.shape)
        output = logits.detach().cpu().numpy()
        labels = all_labels.detach().cpu().numpy()
        #if output is None:
                #output = logits.detach().numpy()
                #labels = label.detach().numpy()
         #   output = logits.detach().cpu().numpy()
         #   labels = label.detach().cpu().numpy()
                #print(output.shape)
                #print(labels.shape)
        #else:
                #output = np.append(output, logits.detach().numpy(), axis=0)
                #labels = np.append(labels, label.detach().numpy(), axis=0)
        #    output = np.append(output, logits.detach().cpu().numpy(), axis=0)
         #   labels = np.append(labels, label.detach().cpu().numpy(), axis=0)
                #print(output.shape)
                #print(labels.shape)
        output=np.argmax(output,axis=1)
        results.append(np.any(output==labels))
        correct+=int(np.any(output==labels))
        print(correct)
        print(all_count)
        print(correct / all_count)
print(correct/all_count)
np.save("results.npy",results)

