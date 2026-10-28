#!/bin/sh
cd /workspace
python tools/nref.py '{"sign_first": true, "mt": "Tt", "emb_mode": "inv", "conv_rev": false, "kmap": "tile", "q_layout": "chunked", "gate_act": "sigmoid", "gdn_gate": "silu", "gdn_permute": true, "q_scale": true, "decay_first": true, "decay_exp": true}' 248044,248046 tools/nref_a2.bin > tools/nref_a2.log 2>&1
python tools/compare.py tools/oracle_logits_a.bin tools/nref_a2.bin > tools/cmp_a2.log 2>&1
