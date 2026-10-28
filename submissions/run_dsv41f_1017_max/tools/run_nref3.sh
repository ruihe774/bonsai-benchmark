#!/bin/sh
cd /workspace
python tools/nref.py '{"sign_first": true, "mt": "Tt", "emb_mode": "inv", "conv_rev": false, "kmap": "tile", "q_layout": "chunked", "gate_act": "sigmoid", "gdn_gate": "silu", "gdn_permute": true, "q_scale": true, "decay_first": true, "decay_exp": true}' 9419 tools/nref_t1b.bin > tools/nref_t1b.log 2>&1
python tools/compare.py tools/vk_t1.bin tools/nref_t1b.bin > tools/cmp_vk_nref.log 2>&1
python tools/compare.py tools/oracle_logits_t1_cpu.bin tools/nref_t1b.bin > tools/cmp_oracle_nref_t1.log 2>&1
