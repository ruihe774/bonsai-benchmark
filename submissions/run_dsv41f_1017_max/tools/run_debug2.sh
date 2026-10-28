#!/bin/bash
cd /workspace
./ref/bin/llama-debug -m data/Ternary-Bonsai-2-4B-PTQ1_0.gguf -p "Hello" -n 1 -ngl 0 --verbose --tensor-filter "-0$|^model|^inp|^embd|^result|^logits" > tools/debug2.txt 2>&1
ls -la tools/debug2.txt
