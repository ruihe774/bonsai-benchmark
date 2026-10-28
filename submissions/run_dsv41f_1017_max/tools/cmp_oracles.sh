#!/bin/sh
cd /workspace
python tools/compare.py tools/oracle_logits_t1_cpu.bin tools/oracle_logits_t1_vk.bin > tools/cmp_oracle_cpuvk.log 2>&1
python tools/compare.py tools/oracle_logits_t3_vk.bin tools/oracle_logits_a.bin > tools/cmp_oracle_t3a.log 2>&1
