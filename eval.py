from file_utils import WEIGHTS_NAME
from configuration_roberta import RobertaConfig
from modeling_roberta import RobertaForSequenceClassification
from tokenization_roberta import RobertaTokenizer
import torch
import numpy as np
from processors.glue2 import glue_convert_examples_to_features as convert_examples_to_features
from processors.utils2 import DataProcessor, InputExample, InputFeatures

config_class, model_class, tokenizer_class = RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer
config = config_class.from_pretrained("./output_mytask_sfu/checkpoint-500/config.json")
tokenizer = tokenizer_class.from_pretrained("./output_mytask_sfu")
model = model_class.from_pretrained("./output_mytask_sfu/checkpoint-500/pytorch_model.bin", from_tf=False,
                                        config=config)
model=model.cuda()
model.eval()
samples=[["HAVE HAVE YOU BEEN TO THE COULDN'T MAKE", 'No.'],["HAVE HAVE YOU BEEN TO THE COULDN'T MAKE", "No, I haven't been there."],["HAVE HAVE YOU BEEN TO THE COULDN'T MAKE", "No, I haven't been there yet."],["HAVE HAVE YOU BEEN TO THE COULDN'T MAKE", 'Not yet.'],['HAVE YOU BEEN TO GRANDMA', "Yes, I've been there once."],['I USUALLY START WORK AT HALF PAST', 'At nine thirty.'], ['I USUALLY START WORK AT HALF PAST', 'At nine thirty a.m.'],['I USUALLY START WORK AT HALF PAST', 'At nine thirty in the morning.'],['I USUALLY START WORK AT HALF PAST', 'At half past nine in the morning.'],['WHEN DO YOU USUALLY START WORK', 'At nine thirty.'],['WHEN DO YOU USUALLY START WORK', 'At nine thirty a.m.']\
         ,['WHEN DO YOU USUALLY START WORK', 'At nine thirty in the morning.'],['WHEN DO YOU USUALLY START WORK', 'At half past nine.'],['WHEN DO YOU USUALLY START WORK', 'At half past nine in the morning.'],['I USUALLY START WORK AT NINE HOW', 'At nine thirty a.m.']\
         ,['I USUALLY START WORK AT NINE HOW', 'At nine thirty in the morning.'],['I USUALLY START WORK AT NINE HOW', 'At half past nine.'],['I USUALLY START WORK AT NINE HOW', 'At half past nine a.m.'],['I USUALLY START WORK AT NINE HOW', 'At half past nine in the morning.']\
         ,['I USUALLY SAYS SISTER OH', 'At half past nine in the morning.']]
examples = []
for i in range(len(samples)):
    guid = "%s" % (i)
    text_a = samples[i][0].lower()
    text_b = samples[i][1].lower()
    label = "0"
    examples.append(
        InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
features = convert_examples_to_features(examples,
                                                tokenizer,
                                                label_list=["0","1"],
                                                max_length=64,
                                                output_mode="classification",
                                                pad_on_left=False,
                                                # pad on the left for xlnet
                                                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                pad_token_segment_id=0,
                                                )
output=[]
for f in features:
    input_ids = torch.tensor(f.input_ids, dtype=torch.long).unsqueeze(0).cuda()
    print(input_ids.shape)
    attention_mask = torch.tensor(f.attention_mask, dtype=torch.long).unsqueeze(0).cuda()
    align_mask = torch.tensor(f.align_mask, dtype=torch.long).unsqueeze(0).cuda()
    outputs = model(input_ids=input_ids,attention_mask=attention_mask,align_mask=align_mask)
    logits=outputs[0]
    logits=torch.nn.functional.softmax(logits,dim=-1)
    output.append(logits.detach().cpu().numpy().squeeze())
print(output)
np.save("output.npy",output)