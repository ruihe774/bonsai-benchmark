#!/bin/sh
cd /workspace
python tools/compare.py tools/oracle_logits_t1_cpu.bin tools/vk_t1.bin > tools/cmp_vk_t1.log 2>&1
python tools/compare.py tools/oracle_logits_a.bin tools/vk_a.bin > tools/cmp_vk_a.log 2>&1
python tools/compare.py tools/oracle_logits_t1_vk.bin tools/vk_t1.bin > tools/cmp_vk_t1v.log 2>&1
