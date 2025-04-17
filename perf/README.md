# LLM Performance Benchmark Suite (k6-based)

This suite enables comprehensive performance testing of OpenAI-compatible LLM APIs using k6, with support for load, soak, spike, and breakpoint tests. It reads prompt datasets in JSONL format and reports detailed latency and throughput metrics.

## Directory Structure

- `perf/scripts/llm_benchmark.js` — Main k6 test script
- `perf/scripts/utils.js` — JSONL parsing utility
- `perf/results/` — Output directory for test results

## Running a Test

### 1. Install k6

See: https://k6.io/docs/getting-started/installation/

### 2. Run a Benchmark

Example: **Load Test** (ramp to 10 VUs in 30s, hold for 120s)
```sh
k6 run perf/scripts/llm_benchmark.js \
  --env DATASET=datasets/llama4/prompts_full_50_100.jsonl \
  --env ENDPOINT=http://localhost:8000/v1/completions \
  --env PROMPT_TYPE=full \
  --env STAGES='[{"target":10,"duration":"30s"},{"target":10,"duration":"120s"}]' \
  --out csv=perf/results/load_test.csv
```

Example: **Soak Test** (ramp to 100 VUs in 4m, hold for 10m)
```sh
k6 run perf/scripts/llm_benchmark.js \
  --env DATASET=datasets/llama4/prompts_chat_50_100.jsonl \
  --env ENDPOINT=http://localhost:8000/v1/chat/completions \
  --env PROMPT_TYPE=chat \
  --env STAGES='[{"target":100,"duration":"4m"},{"target":100,"duration":"10m"}]' \
  --out csv=perf/results/soak_test.csv
```

- Adjust `STAGES` for spike, breakpoint, or custom scenarios.
- Set `API_KEY` if your endpoint requires authentication.

### 3. Output

- Results are saved as CSV in `perf/results/`.
- Custom metrics (TTFT, TPOT, error rate) are included in the CSV.

## Metrics

- **TTFT**: Time to First Token (http.waiting)
- **TPOT**: Time to Produce Output (http.duration)
- **Percentiles**: p50, p90, p95, p99, mean
- **Error %**: Fraction of non-200 responses
- **Throughput**: Requests/sec (from k6 summary)
- **MBU/MTU**: (To be computed in post-processing)

## Post-Processing & Reporting

You can use Python, Jupyter, or your preferred tool to analyze the CSV and compute:
- Percentiles, mean, error %
- TPOT estimation: For each prompt, compare generation time for output length M and M+10, average the difference/10
- MBU/MTU: Compute based on your LLM's output and prompt sizes

A sample Jupyter notebook and/or Python script for analysis can be added in `perf/reporting/`.

## Extending

- Add new datasets in `datasets/`
- Modify `perf/scripts/llm_benchmark.js` for custom logic or metrics