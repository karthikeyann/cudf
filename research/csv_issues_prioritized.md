# cuDF CSV Performance Optimization Sprint — Issue Tracker

Prioritized list of open GitHub issues relevant to CSV reader/writer performance,
organized for a performance optimization sprint.

**Scoring**: Weight 1-10 (10 = highest priority). Factors: performance impact,
user demand, Spark-RAPIDS blocker status, architectural significance.

---

## Tier 1: Architecture & Core Performance (Weight 9-10)

These are the issues that define the modernization roadmap and unlock the largest
performance gains.

| Weight | Issue | Title | Category | Why |
|--------|-------|-------|----------|-----|
| **10** | [#13916](https://github.com/rapidsai/cudf/issues/13916) | **Modernize CSV reader and expand reader options** | Architecture | Master tracking issue. Defines 4-step refactor plan (decompress, row offsets, type detect, decode). References ParPaRaw FST-based row offset detection. Every other perf issue feeds into this. |
| **9** | [#13797](https://github.com/rapidsai/cudf/issues/13797) | **Avoid host-side processing in CSV reader** | Performance | Eliminating host-side decompression/BOM/byte-range processing enables kvikIO + GPUDirect Storage path. Removes CPU-GPU round trips — potentially 2-5x for compressed inputs. |
| **9** | [#5080](https://github.com/rapidsai/cudf/issues/5080) | **CSV reader data type detection is slow** | Performance | Type detection = ~25% of total read time. Low-hanging fruit: better kernel design, sampling, or speculative typing could save significant wall time. |
| **9** | [#11728](https://github.com/rapidsai/cudf/issues/11728) | **`read_csv` context-passing interface for distributed/segmented parsing** | Architecture | DFA context-passing for byte-range parsing. Directly maps to ParPaRaw's parallel FSM approach. Enables safe distributed CSV parsing without sequential preprocessing. Critical for Spark + Dask scale-out. |
| **9** | [#12255](https://github.com/rapidsai/cudf/issues/12255) | **Support device-side de/compression of CSV files** | Performance | GPU decompression removes the biggest host-side bottleneck. With dask_cudf, 8 GPU workers are throttled by CPU decompression — this is a multiplier on multi-GPU throughput. |

## Tier 2: Kernel-Level Performance (Weight 7-8)

Direct performance improvements to existing GPU kernels and data paths.

| Weight | Issue | Title | Category | Why |
|--------|-------|-------|----------|-----|
| **8** | [#14066](https://github.com/rapidsai/cudf/issues/14066) | **Use grid stride in CSV reader kernels** | Performance | Current thread-per-row launch wastes occupancy on large files. Grid stride loop = better SM utilization, fewer kernel launches. Applies to parser + type inference kernels. Good first issue. |
| **8** | [#2678](https://github.com/rapidsai/cudf/issues/2678) | **Performance improvements for csv-writer** | Performance | Profiling showed column transpose host→device was >50% of write time. Device-side formatting already showed 20-30% improvement. More gains possible with fused kernels. |
| **8** | [#10426](https://github.com/rapidsai/cudf/issues/10426) | **`.to_csv()` OOM with large DataFrames** | Perf/Memory | 4GB DataFrame causes OOM on write. Needs chunked/streaming write path. Memory-bound users hit this wall. |
| **7** | [#4999](https://github.com/rapidsai/cudf/issues/4999) | **Add support for `low_memory` parameter in read_csv** | Perf/Memory | Chunked reading for memory-constrained systems. Currently reads entire file into GPU memory at once. 13 comments = high user demand. Pandas compatibility. |
| **7** | [#6572](https://github.com/rapidsai/cudf/issues/6572) | **Support '\n', '\r' and '\r\n' as line delimiters simultaneously** | Perf/Compat | Currently requires CPU preprocessing in Spark-RAPIDS to normalize line endings before GPU parsing. Eliminating this preprocessing = direct perf win for Spark users. |

## Tier 3: Correctness Issues That Block Performance Use Cases (Weight 6-7)

Bugs that force workarounds, fallback to CPU, or prevent using fast paths.

| Weight | Issue | Title | Category | Why |
|--------|-------|-------|----------|-----|
| **7** | [#11984](https://github.com/rapidsai/cudf/issues/11984) | **Add support for escape characters in CSV** | Spark blocker | Spark uses `\` escaping (not `""` doubling). Without this, Spark-RAPIDS must fall back to CPU for escaped CSVs. Touches row offset FSM + field parsing. |
| **7** | [#11948](https://github.com/rapidsai/cudf/issues/11948) | **CSV reader cannot handle unquoted quote character in a field** | Spark blocker | Misplaced quotes corrupt row offsets → truncated data. Spark blocker. Needs FSM state expansion (relevant to #13916 Step 2). |
| **7** | [#12145](https://github.com/rapidsai/cudf/issues/12145) | **Option to read empty strings as `""`, not `null`** | Spark blocker | Spark/Polars expect `""` → empty string, cuDF returns null. Blocks cudf-polars adoption. Linked from #20343. Decode step change. |
| **6** | [#14881](https://github.com/rapidsai/cudf/issues/14881) | **byte-range logic confused with empty rows** | Correctness | Lost rows when byte_range includes trailing empty rows. Blocks reliable distributed parsing (Dask/Spark byte-range splits). |
| **6** | [#12331](https://github.com/rapidsai/cudf/issues/12331) | **Support inferring column names with `byte_range=`** | Usability | Currently errors if header inference + byte_range are both used. Blocks easy distributed CSV reading. |
| **6** | [#12588](https://github.com/rapidsai/cudf/issues/12588) | **read_csv always reads quoted values as strings** | Correctness | Pandas infers types through quotes; cuDF doesn't. Forces users to manually specify dtypes → slower workflows. |
| **6** | [#2398](https://github.com/rapidsai/cudf/issues/2398) | **read_csv fails to correctly handle misplaced quotes** | Correctness | Misplaced quotes → wrong parsing. 5 comments. Related to #11948 FSM fix. |
| **6** | [#6313](https://github.com/rapidsai/cudf/issues/6313) | **read_csv should not cast to float if there are null entries** | Correctness | Integer columns with nulls get inferred as float. 7 comments. Type inference kernel issue. |
| **6** | [#7088](https://github.com/rapidsai/cudf/issues/7088) | **Overflow ints with nulls inferred as float** | Correctness | Related to #6313. Integer overflow + nulls → float. 7 comments. |

## Tier 4: Usability & Feature Gaps That Impact Performance Workflows (Weight 4-5)

Features that indirectly affect performance by enabling better usage patterns or
removing CPU fallbacks.

| Weight | Issue | Title | Category | Why |
|--------|-------|-------|----------|-----|
| **5** | [#17826](https://github.com/rapidsai/cudf/issues/17826) | **Support inferring `lineterminator` in read_csv** | Usability | Files with `\r\n` read as empty DataFrames. Forces manual intervention. Ties into #6572. |
| **5** | [#9987](https://github.com/rapidsai/cudf/issues/9987) | **Support multi-character separators in read_csv** | Usability | `sep="||"` etc. Pandas supports it, cuDF doesn't. 6 comments. Would need `seek_field_end` changes. |
| **5** | [#715](https://github.com/rapidsai/cudf/issues/715) | **Support escapechar in read_csv** | Compat | Related to #11984. Pandas `escapechar` parameter. 3 comments, open since 2019. |
| **5** | [#20343](https://github.com/rapidsai/cudf/issues/20343) | **`""` as null instead of empty string in cudf-polars** | Polars compat | Blocks cudf-polars CSV adoption. Root cause = #12145. |
| **5** | [#20518](https://github.com/rapidsai/cudf/issues/20518) | **`pl.scan_csv.slice` past length raises RuntimeError** | Polars compat | cudf-polars crash on out-of-bounds slice. Blocks Polars GPU engine CSV usage. |
| **5** | [#13856](https://github.com/rapidsai/cudf/issues/13856) | **read_csv(comment=#) still including commented lines** | Correctness | Comment handling bug in row offset detection. Ties to FSM refactor in #13916 Step 2. |
| **4** | [#5142](https://github.com/rapidsai/cudf/issues/5142) | **Snappy compressed CSV not implemented** | Feature | Blocks compressed CSV ingest. Referenced in #13916 Step 1. |
| **4** | [#4001](https://github.com/rapidsai/cudf/issues/4001) | **Support nanValue Spark CSV parse option** | Spark compat | Additional NaN representations for Spark. Decode step (#13916 Step 4). |
| **4** | [#10599](https://github.com/rapidsai/cudf/issues/10599) | **String to float parsing inconsistent** | Correctness | 21 comments (!). Float parsing differs from `to_numeric`. Probably wontfix per #13916, but high user pain. |
| **4** | [#15985](https://github.com/rapidsai/cudf/issues/15985) | **Deprecate windowslinetermination** | Cleanup | Dead code (getters/setters exist but unused). Good first issue for code cleanup. |
| **4** | [#6659](https://github.com/rapidsai/cudf/issues/6659) | **CSV writer returning full-subsecond data for duration types** | Writer bug | 7 comments. Duration/timestamp formatting. |
| **4** | [#6235](https://github.com/rapidsai/cudf/issues/6235) | **float min/max values truncated, read back as inf** | Writer bug | Round-trip fidelity issue. Precision in CSV writer. |
| **4** | [#7108](https://github.com/rapidsai/cudf/issues/7108) | **to_csv adding extra quotation marks** | Writer compat | Pandas compat for quoting behavior. |
| **4** | [#6187](https://github.com/rapidsai/cudf/issues/6187) | **Allow datasource/data_sink to decide host/device copies** | Architecture | Shared abstraction for all IO. Would benefit CSV + all other readers. |

## Tier 5: Broader Infrastructure (Weight 2-3)

Issues that touch CSV but are broader in scope.

| Weight | Issue | Title | Category | Why |
|--------|-------|-------|----------|-----|
| **3** | [#15907](https://github.com/rapidsai/cudf/issues/15907) | **Replace std::string with std::string_view internally** | Perf/Cleanup | Reduces unnecessary allocations across all IO. Minor but cumulative. |
| **3** | [#13159](https://github.com/rapidsai/cudf/issues/13159) | **64-bit size type option at build-time** | Architecture | Enables >2B rows. Needed for truly large CSV files. |
| **3** | [#12739](https://github.com/rapidsai/cudf/issues/12739) | **Update IO benchmarks for consistency** | Testing | Better benchmarks = better optimization feedback. In progress. |
| **3** | [#18547](https://github.com/rapidsai/cudf/issues/18547) | **Refactor compression type detection** | Cleanup | Simplify AUTO compression detection logic. |
| **3** | [#7602](https://github.com/rapidsai/cudf/issues/7602) | **dask_cudf generates files it cannot read back** | Compat | Round-trip failure. 9 comments. |
| **2** | [#5017](https://github.com/rapidsai/cudf/issues/5017) | **skiprows: support list-like and callable** | Compat | Pandas API gap. |
| **2** | [#18676](https://github.com/rapidsai/cudf/issues/18676) | **Support skip_rows_after_header** | Compat | Minor feature gap. |
| **2** | [#11135](https://github.com/rapidsai/cudf/issues/11135) | **read_csv support header param with list input** | Compat | Multi-header CSV. Niche. |
| **2** | [#12582](https://github.com/rapidsai/cudf/issues/12582) | **Int64Index for header=None** | Compat | Auto-generated column names. Referenced in #13916. |
| **2** | [#7704](https://github.com/rapidsai/cudf/issues/7704) | **Support list types in to_csv** | Writer feature | Niche. |
| **2** | [#10121](https://github.com/rapidsai/cudf/issues/10121) | **Support "mode" argument to to_csv** | Writer feature | Append mode. |
| **2** | [#12412](https://github.com/rapidsai/cudf/issues/12412) | **read_csv encoding parameter** | Compat | Currently only UTF-8. |
| **2** | [#20752](https://github.com/rapidsai/cudf/issues/20752) | **Error reading mixed numeric and boolean from CSV** | Bug | Type inference edge case. |
| **2** | [#6659](https://github.com/rapidsai/cudf/issues/6659) | **CSV writer duration type formatting** | Writer bug | Timestamp formatting. |
| **2** | [#14498](https://github.com/rapidsai/cudf/issues/14498) | **Enable cuDF spilling in cudf.pandas** | Memory | Broader than CSV but helps OOM on large CSV reads. |

---

## Sprint Roadmap Summary

### Phase 1: Foundation (High-impact, unblocks everything)
1. **#13916** — Adopt as the north star. All work feeds into this.
2. **#5080** — Type detection kernel optimization (25% of read time).
3. **#14066** — Grid stride kernels (quick win, good first issue).
4. **#13797** — Move preprocessing to device side.

### Phase 2: Architecture (FSM + streaming)
5. **#11728** — Context-passing interface (ParPaRaw-style distributed parsing).
6. **#6572** — Universal line terminator support (eliminates Spark preprocessing).
7. **#11984 + #11948** — Escape char + misplaced quote handling (Spark blockers, FSM expansion).
8. **#12255** — Device-side decompression.

### Phase 3: Memory & Scale
9. **#4999** — Chunked/low_memory reading.
10. **#10426** — Streaming CSV writer (OOM fix).
11. **#14881 + #12331** — byte_range fixes for distributed parsing.

### Phase 4: Polish & Compatibility
12. **#12145 + #20343** — Empty string vs null handling.
13. **#12588** — Type inference through quotes.
14. **#2678** — Writer performance (device-side formatting).

---

*Generated 2026-04-10 from rapidsai/cudf open issues matching "csv".*
*Total issues tracked: 46 | Performance-relevant: 28 | Spark blockers: 5 | Polars blockers: 2*
