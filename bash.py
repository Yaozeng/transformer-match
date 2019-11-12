#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import torch
os.system("python -m torch.utils.bottleneck ./eval.py")
os.system("python -m torch.utils.bottleneck ./eval2.py")