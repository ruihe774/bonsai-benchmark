#!/bin/bash
cd /workspace
./ref/bin/llama-debug --help > tools/llama_debug_help.txt 2>&1
./ref/bin/llama-cli --help > tools/llama_cli_help.txt 2>&1
