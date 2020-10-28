# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/python3
# Lint as: python3
"""Filters an access trace.

Given a CSV file containing (pc, address) in hex, filters the file to only
include the desired cache set accesses and splits the resulting trace into
train (80%) / valid (10%) / test (10%).

Example usage:

  Suppose that the access trace exists at /path/to/file.csv
  Results in the following three files: train.csv, valid.csv, test.csv.

  python3 filter.py /path/to/file.csv
"""
import argparse
import csv
import os
import subprocess
import random

import numpy as np
import tqdm

if __name__ == "__main__":
  # The cache sets used in the paper:
  # An Imitation Learning Approach to Cache Replacement
  PAPER_CACHE_SETS = [4,3,2,14,8,10,5,1,11,0,7,13,6,12,9,15,16,17,
                    18,19,20,21,22,23,24,25,26,27,28,29,30,31]

  parser = argparse.ArgumentParser()
  parser.add_argument(
      "access_trace_filename", help="Local path to the access trace to filter.")
  parser.add_argument(
      "-s", "--cache_sets", default=PAPER_CACHE_SETS,
      help=("Specifies which cache sets to keep. Defaults to the 64 sets used"
            " in the paper."))
  parser.add_argument(
      "-a", "--associativity", default=16,
      help="Associativity of the cache.")
  parser.add_argument(
      "-c", "--capacity", default=1 * 1024 * 1024,
      help="Capacity of the cache.")
  parser.add_argument(
      "-l", "--cache_line_size", default=4096,
      help="Size of the cache lines in bytes.")
  parser.add_argument(
      "-b", "--batch_size", default=32,
      help=("Ensures that train.csv, valid.csv, and test.csv contain a number"
            " of accesses that is a multiple of this. Use 1 to avoid this."))
  args = parser.parse_args()

  PREFIX = "_filter_traces"
  output_filenames = ["train.csv", "valid.csv", "test.csv", "all.csv"]
  output_filenames += [PREFIX + str(i) for i in range(10)]

  for output_filename in output_filenames:
    if os.path.exists(output_filename):
      print(f"File {output_filename} already exists, overwriting.")

  num_cache_lines = args.capacity // args.cache_line_size
  num_sets = num_cache_lines // args.associativity # 1024
  cache_bits = int(np.log2(args.cache_line_size))
  set_bits = int(np.log2(num_sets)) # = 10
  assert set_bits == 4
  num_lines = 0
  accepted_cache_sets = set(args.cache_sets)

  # use a dictionary where each value is a list to keep the trace
  # e.g. trace[0] is a list of tuples for the first
  # max_accesses_per_tkey contiguous page accesses from original trace
  trace = {}
  tkey = 0
  trace[tkey] = []
  max_accesses_per_tkey = 1000

  print("reading trace...")
  with open(args.access_trace_filename, "r") as read:
    for pc, address in tqdm.tqdm(csv.reader(read)):
      pc = int(pc, 16)
      address = int(address, 16)
      #aligned_address = address >> cache_bits
      aligned_address = address
      set_id = aligned_address & ((1 << set_bits) - 1)
      if set_id in accepted_cache_sets:
        num_lines += 1
        if len(trace[tkey]) == max_accesses_per_tkey:
          tkey += 1
          trace[tkey] = []
        trace[tkey].append((pc,address))
        #write.write(f"0x{pc:x},0x{address:x}\n")

  print("writing trace...")
  with open("all.csv", "w") as write:
    while trace:
      # choose a random tkey and write to file
      rand_tkey = random.choice(list(trace.keys()))
      for pc, address in trace[rand_tkey]:
        write.write(f"0x{pc:x},0x{address:x}\n")
      # delete tkey from trace
      del trace[rand_tkey]

  split_length = num_lines // 10
  # Make split_length a multiple of batch_size
  split_length = split_length // args.batch_size * args.batch_size

  cmd = f"split -l {split_length} --numeric-suffixes all.csv {PREFIX}"
  print(cmd)
  subprocess.run(cmd, check=True, shell=True)

  # Removes the extra accesses that don't fit into batch_size multiples.
  cmd = f"wc -l {PREFIX}10"
  print(cmd)
  subprocess.run(cmd, check=True, shell=True)

  cmd = f"rm {PREFIX}10"
  print(cmd)
  subprocess.run(cmd, check=True, shell=True)

  # Last split is test
  # Second last split is valid
  # First 8 splits are train
  cmd = f"mv {PREFIX}09 test.csv"
  print(cmd)
  subprocess.run(cmd, check=True, shell=True)

  cmd = f"mv {PREFIX}08 valid.csv"
  print(cmd)
  subprocess.run(cmd, check=True, shell=True)

  cmd = f"cat {PREFIX}0[0-7] > train.csv"
  print(cmd)
  subprocess.run(cmd, check=True, shell=True)

  # Clean up
  cmd = f"rm {PREFIX}0[0-7]"
  print(cmd)
  subprocess.run(cmd, check=True, shell=True)

  #cmd = "rm all.csv"
  #print(cmd)
  #subprocess.run(cmd, check=True, shell=True)
