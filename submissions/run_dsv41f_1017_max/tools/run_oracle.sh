#!/bin/bash
cd /workspace
./ref/bin/bonsai-ref < "$1" > "$2" 2> "$3"
