# cpp-bpe
Byte Pair Encoding tokenizer implemented in C++
---
- Train / encode mode
- Export merges to file
- Multithreading

Run using:
```
g++ -o tokenizer .\tokenizer.cpp
```

```
Usage: tokenizer <mode> <input_file> [options]
Modes:
  train <vocab_size>
  encode
Options:
  -m, --merges <merges_file>  Specify merges file (required for encode mode)
  -o, --output <output_file>  Specify output file (default: output.bin for encode, merges.txt for train)
  -t, --threads <n>  Specify number of threads to use (-1 for all available) (default: 1)
```
