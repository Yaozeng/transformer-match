#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import torch
os.system("python ./run_mytask_bert.py")
os.system("python ./run_mytask_bert_align.py")
os.system("python ./run_mytask_roberta.py")
os.system("python ./run_mytask_roberta_align.py")
os.system("python ./run_mytask_distil.py")
os.system("python ./run_mytask_distil_align.py")
os.system("python eval_bert.py")
os.system("python eval_bert_align.py")
os.system("python eval_roberta.py")
os.system("python eval_roberta_align.py")
os.system("python eval_distilbert.py")
os.system("python eval_distilbert_align.py")
