#!/bin/zsh
# Copyright 2026 Peter Marelas
# SPDX-License-Identifier: Apache-2.0
# Waits for the forced run, then samples the two-sentence text with the oracle at two seeds.
cd "$(dirname "$0")"
while [ ! -f out/plain2_ref_f32.npz ]; do sleep 20; done
.venv/bin/python dump_reference.py --spec specs/plain2.json --steps 90 --seed 3 --dtype float32 --out out/plain2_oracle_s3.npz > out/plain2_oracle_s3.log 2>&1
.venv/bin/python dump_reference.py --spec specs/plain2.json --steps 90 --seed 1 --dtype float32 --out out/plain2_oracle_s1.npz > out/plain2_oracle_s1.log 2>&1
