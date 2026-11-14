# Tokenizer Engineering Lab — Benchmark Report

## 1. Project Overview

This project implements a custom Byte Pair Encoding (BPE) tokenizer
from scratch and evaluates its performance, memory consumption,
tokenization quality, and robustness.

### Objectives

- Understand BPE tokenization internally.
- Measure encoding latency.
- Measure token throughput.
- Analyze memory consumption.
- Evaluate tokenization quality.
- Test robustness against unusual inputs.

## 2. Experimental Environment

| Component | Configuration |
|---|---|
| Operating System | Windows |
| Python | 3.10.5 |
| Framework | Custom Python implementation |
| Tokenizer | BPE |
| Hardware | CPU |
| Benchmark corpus | Synthetic repeated vocabulary |

### Benchmark Vocabulary

The benchmark uses these 12 words:

- low
- lower
- lowest
- learning
- tokenizer
- benchmark
- transformer
- model
- data
- system
- python
- language

## 3. Latency Benchmark

Latency measures the time required to encode an input.

| Input Words | Token IDs | Median (ms) | Average (ms) |
|---:|---:|---:|---:|
| 100 | 406 | 4.6573 | 5.2888 |
| 1,000 | 4,156 | 48.1608 | 55.3441 |
| 10,000 | 41,656 | 561.7603 | 602.6134 |
| 100,000 | 416,656 | 4,511.5141 | 4,581.8132 |

### Tail Latency

| Input Words | P95 (ms) | P99 (ms) |
|---:|---:|---:|
| 100 | 8.6991 | 15.8705 |
| 1,000 | 88.9849 | 161.3152 |
| 10,000 | 855.4778 | 982.1643 |
| 100,000 | 4,892.8691 | 4,960.9029 |

### Observation

- Latency increases with input size.
- The growth is approximately linear in these measurements.
- Larger inputs require substantially more encoding time.
- These results measure tokenizer encoding, not complete model inference.

## 4. Throughput Benchmark

Throughput measures how many tokens are processed per second.

| Input Words | Tokens Per Run | Total Tokens | Time (seconds) | Tokens/sec |
|---:|---:|---:|---:|---:|
| 100 | 406 | 40,600 | 0.4982 | 81,499 |
| 1,000 | 4,156 | 415,600 | 4.5491 | 91,360 |
| 10,000 | 41,656 | 4,165,600 | 79.2756 | 52,546 |
| 100,000 | 416,656 | 41,665,600 | 492.4235 | 84,613 |

### Observation

- Throughput is not consistent across input sizes.
- The 10,000-word benchmark shows lower throughput.
- CPU scheduling, memory allocation, and execution variability
  may influence these results.
- These measurements represent sequential throughput,
  not concurrent request throughput.

## 5. Memory Benchmark

Memory usage was measured using Python's `tracemalloc`.

| Input Words | Token IDs | Current Memory (MB) | Peak Memory (MB) |
|---:|---:|---:|---:|
| 100 | 406 | 0.0043 | 0.0137 |
| 1,000 | 4,156 | 0.0361 | 0.1268 |
| 10,000 | 41,656 | 0.3351 | 1.2813 |

### Observation

- Memory consumption increases with input size.
- Peak memory is higher than current memory.
- Temporary allocations contribute to peak memory.
- `tracemalloc` measures Python-level allocations,
  not complete process memory usage.

## 6. Tokenization Quality Metrics

The following quality metrics were implemented or tested.

| Metric | Purpose | Status |
|---|---|---|
| Round-trip accuracy | Checks encode/decode reconstruction | PASS |
| Tokenization coverage | Checks vocabulary coverage | PASS |
| UNK rate | Measures unknown token frequency | Implemented |
| Compression ratio | Measures characters per content token | Implemented |
| Fertility | Measures tokens per word | PASS |
| Robustness | Tests unusual and edge-case inputs | PASS |

### Fertility Results

| Input | Words | Tokens | Fertility |
|---|---:|---:|---:|
| low lower | 2 | 3 | 1.5 |
| low lower low lower | 4 | 6 | 1.5 |
| low lower lowest low lower | 5 | 10 | 2.0 |

Formula:

Fertility = Total Content Tokens / Total Words

### Compression Ratio

Formula:

Compression Ratio = Original Characters / Content Tokens

Previously observed result:

- Original characters: 26
- Content tokens: 10
- Compression ratio: 2.6

Note: Lower token count and higher compression ratio do not
automatically guarantee better tokenization quality.

## 7. Robustness Testing

Five edge-case tests were executed.

| Test Case | Result |
|---|---|
| Empty string | PASS |
| Multiple spaces | PASS |
| Leading and trailing spaces | PASS |
| Punctuation | PASS |
| Unknown Unicode characters | PASS |

### Test Summary

Total tests: 5

Passed: 5

Failed: 0

## 8. Performance Analysis

### Latency

Encoding latency increases with input size.
The current implementation needs further optimization
for large inputs.

### Throughput

Throughput varies across input sizes.
Additional repeated experiments are required
to establish stable performance.

### Memory

Memory consumption increases with the number of
tokens processed.

### Tokenization Quality

The tokenizer supports basic BPE merging,
special tokens, encoding, decoding, padding,
truncation, and attention masks.

## 9. Limitations

1. Benchmark corpus is synthetic.
2. Tests primarily use a small vocabulary.
3. Measurements are sequential.
4. No concurrent request benchmarking.
5. No comparison against production tokenizers yet.
6. `tracemalloc` does not capture all native memory.
7. Quality metrics have not yet been evaluated on
   a large natural-language dataset.
8. Hardware and background system load can influence results.

## 10. Future Improvements

- Benchmark against established tokenizers.
- Test larger natural-language datasets.
- Improve BPE encoding performance.
- Evaluate multilingual tokenization.
- Add comprehensive Unicode testing.
- Perform concurrent throughput benchmarking.
- Measure process-level memory usage.
- Create reproducible benchmark experiments.
- Build automated benchmark reporting.

## 11. Conclusion

The custom BPE tokenizer has successfully completed
its initial implementation and core benchmarking phase.

Latency, throughput, memory consumption, tokenization
quality, and robustness have been evaluated.

The results provide a baseline for future optimization
and comparison against production-grade tokenizers.

This report represents the current experimental baseline,
not a final production performance claim.