#coding: utf-8

import numpy as np
import time
import sys
import os
import subprocess
import random

import rnn_load
import rnn_tools
import rnn_elman

# python の最大再帰数の設定（Cスタックがオーバーフローしてクラッシュすることを防ぐ）
sys.setrecursionlimit(1500)

if __name__ == '__main__':
    if not param:
        param = {
            
        }
