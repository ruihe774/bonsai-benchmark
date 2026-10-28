#!/bin/bash
cd /workspace
./ref/bin/llama-debug -m data/Ternary-Bonsai-2-4B-PTQ1_0.gguf -p "Hello" -n 1 -ngl 0 --verbose --tensor-filter "norm" > tools/debug1.txt 2>&1
