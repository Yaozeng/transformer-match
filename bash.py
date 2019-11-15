#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import torch
os.system("python -m torch.utils.bottleneck ./run_mytask_bert.py")
os.system("python -m torch.utils.bottleneck ./run_mytask_bert_align.py")
os.system("python -m torch.utils.bottleneck ./run_mytask_distil_align.py")