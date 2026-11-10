# R2 数据层与容错训练工作流：Ultralytics YOLO + Cloudflare R2 + 租用云 GPU 容器（截至 2026-11-09）

Notes date: 2026-11-09. Version snapshot used: Ultralytics 8.4.208 (released 2026-11-08), rclone v1.76.0 (2026-11-06), mountpoint-s3 1.24.0 (2026-08-24), GeeseFS v0.43.9 (2026-09-03), JuiceFS v1.4.2 (2026-10-22), s3fs-fuse v1.97 (2025-12-08), s5cmd v2.3.0 (2024-12-16). Version/date lookups: [Ultralytics releases](https://github.com/ultralytics/ultralytics/releases/tag/v8.4.208), [PyPI](https://pypi.org/project/ultralytics/), [rclone releases](https://github.com/rclone/rclone/releases), [mountpoint-s3 releases](https://github.com/awslabs/mountpoint-s3/releases), [GeeseFS](https://github.com/yandex-cloud/geesefs/releases), [JuiceFS](https://github.com/juicedata/juicefs/releases), [s3fs-fuse](https://github.com/s3fs-fuse/s3fs-fuse/releases), [s5cmd](https://github.com/peak/s5cmd/releases).

Evidence tags used below: **[DOC]** vendor/project documentation; **[CODE]** read directly from Ultralytics `main` source on 2026-11-09 (version string 8.4.208; line numbers drift, function names are stable); **[VENDOR]** benchmark or claim from a party with a commercial interest; **[ANECDOTE]** forum/blog single report; **[CALC]** my own arithmetic (inputs always shown, in Inferences). Items without a tag in Cited Findings are plain documented facts.

Code-source base URL (all [CODE] links): `https://github.com/ultralytics/ultralytics/blob/main/ultralytics/` + file path.

---

## 1. Decision: FUSE-mount R2 during training, or copy to local NVMe at startup? Simplest reliable workflow

### Takeaway
Copy to local disk at startup (preferably as a few tar shards, extracted locally) and train from local files; do NOT train directly from a FUSE mount. The reasons are latency-bound small-file reads, FUSE privileges that rental platforms often do not grant, and Ultralytics' own scan/cache behaviour, not Class B cost, which is negligible. Checkpoints should be written locally and uploaded to R2 from an `on_model_save` callback; on (re)start the script downloads `last.pt` if present and calls `resume=True`. (中文：启动时把数据拷到本地 NVMe，训练只读本地；权重本地写、每个 epoch 由回调上传到 R2；重启时若 R2 有 last.pt 就下载并 resume。)

### Cited Findings
- R2 egress is free and Class B (GET/HEAD) is $0.36 per million, so repeated full-dataset downloads cost almost nothing in R2 fees — [R2 pricing](https://developers.cloudflare.com/r2/pricing/) (page "last updated Oct 30, 2026").
- Ultralytics reads images as ordinary files (`cv2.imread` on `im_files`) and, when training with augmentation and without `cache="ram"`, keeps only a small in-RAM buffer (`min(n_images, batch*8, 1000)`) of recently loaded images for mosaic — [CODE data/base.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/data/base.py) (`max_buffer_length`, `load_image`), [CODE data/augment.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/data/augment.py) (`Mosaic.get_indexes`).
- Rental-platform FUSE support is doubtful: one SkyPilot issue (opened 2026-01-15) reports a FUSE mount failing on RunPod's default runtime with "/dev/fuse: operation not permitted" — [SkyPilot issue #8592](https://www.goodfirstissue.org/skypilot-org/skypilot/issues/8592); an older post quotes RunPod staff saying FUSE is unsupported because it needs container privileges — [AnswerOverflow](https://www.answeroverflow.com/m/1297834108768096257) (date unclear). General guidance: a FUSE mount in Docker needs `--device /dev/fuse` plus `--cap-add SYS_ADMIN` (or privileged), which many platforms forbid — [rclone forum (Docker/FUSE threads)](https://forum.rclone.org/t/cannot-mount-in-docker-container-fuse-device-not-found/50187).
- No built-in Ultralytics S3/R2 weight uploader exists: the callbacks package contains only `base, clearml, comet, dvc, mlflow, platform, raytune, tensorboard, wb` — [GitHub contents API for ultralytics/utils/callbacks](https://api.github.com/repos/ultralytics/ultralytics/contents/ultralytics/utils/callbacks). Custom hooks are registered with `model.add_callback("on_model_save", fn)` — [Ultralytics callbacks docs](https://docs.ultralytics.com/usage/callbacks/).
- `on_model_save` runs right after `save_model()` succeeds (after validation), `on_fit_epoch_end` runs after that — [CODE engine/trainer.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/trainer.py) (training loop, `if (self.args.save or final_epoch) and self.save_model(): self.run_callbacks("on_model_save")`).
- `resume=True` restores model, optimizer, EMA, scaler, epoch counter and early-stop state from the checkpoint — [Ultralytics train docs](https://docs.ultralytics.com/modes/train/), [CODE trainer.py `resume_training` / `_load_checkpoint_state`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/trainer.py).
- R2 is strongly consistent (read-after-write, delete, list) and conditional `PutObject` (`If-None-Match`, `If-Match`) is evaluated atomically at commit — [R2 consistency](https://developers.cloudflare.com/r2/reference/consistency/), [R2 S3 API compatibility](https://developers.cloudflare.com/r2/api/s3/api/).

### Inferences
- [CALC-backed, see §4] A mount without a local cache is latency-bound (N small GETs through a user-space daemon per epoch), while a one-time copy is bandwidth-bound. For a dataset that fits on the container's disk (typical for 10k–200k JPEGs = ~1.5–30 GB), a lazily-filled FUSE cache (rclone `--vfs-cache-mode full`) ends up holding the same bytes on the same disk, so the mount adds FUSE risk without saving anything.
- Fewer moving parts is itself the reliability argument: copy-at-startup + local training has two failure surfaces (download, upload), both plain HTTPS, neither needing privileges. A mount adds a long-lived daemon whose hang/disconnect would stall every DataLoader worker.
- Recommended skeleton (own design, untested): (1) `.ready` marker check → download tar shards from R2 → extract to `/workspace/data` → touch `.ready`; (2) look for `runs/<exp>/manifest.json` in R2 → if present, download referenced checkpoint, verify hash + `torch.load`, set `resume=True`; (3) `YOLO(...).train(...)` with an `on_model_save` callback that uploads `trainer.last` (and `trainer.best` when it changed) synchronously; (4) on clean completion upload final `best.pt`/`last.pt` and a `DONE` marker; (5) the whole script is idempotent and safe to re-run in a loop by the container entrypoint. Details in §7.
- Only reasons to consider a mount: dataset larger than local disk, or many short-lived containers each touching a tiny subset. Neither matches a 10k–200k image detection set.

### Gaps
- No platform-specific documentation found for whether a given GPU rental vendor (RunPod, Vast.ai, Lambda, etc.) grants `/dev/fuse` + `SYS_ADMIN`; Vast.ai evidence was not found at all. Must be tested on the chosen provider.
- Spot/preemption signalling (SIGTERM grace period) is provider-specific and not researched.

---

## 2. Cloudflare R2 facts as of late 2026 (pricing, limits, S3 compatibility, endpoints, location, newer features)

### Takeaway
R2 is cheap and egress-free; a 10k–200k image dataset (≤ ~30 GB) costs well under $1/month to store and a few cents per full download. The S3 surface is good enough for rclone/s5cmd/boto3 but is a partial implementation (no tagging/versioning/ACL/object-lock/etc., composite-only checksums except CRC64NVME, unsorted-listing complaint from JuiceFS). The only documented request-rate limit relevant here is 1 write/second per object key.

### Cited Findings
**Pricing** — [R2 pricing](https://developers.cloudflare.com/r2/pricing/) (page last updated Oct 30, 2026):
- Storage: Standard $0.015/GB-month; Infrequent Access (IA) $0.01/GB-month.
- Class A (writes/lists: ListObjects, PutObject, CopyObject, CreateMultipartUpload, UploadPart, CompleteMultipartUpload, ListParts, …): Standard $4.50 per million; IA $9.00 per million.
- Class B (reads: GetObject, HeadObject, HeadBucket, GetBucketLocation, …): Standard $0.36 per million; IA $0.90 per million.
- Free operations: DeleteObject, DeleteBucket, AbortMultipartUpload.
- IA: $0.01/GB retrieval, 30-day minimum storage duration; Standard has no retrieval fee or minimum.
- Free tier (monthly, Standard only): 10 GB-month storage, 1 million Class A, 10 million Class B; egress free.
- Egress free via Workers API, S3 API and r2.dev; "other metered services connected to a bucket may charge."
- Requests rejected at authentication/authorization are not charged; requests that pass auth but fail otherwise are billed by class.
- First 100,000 buckets per account included.
- Super Slurper and Sippy are free but the Class A operations they perform on your bucket are billed.

**Limits** — [R2 limits](https://developers.cloudflare.com/r2/platform/limits/):
- Object size: 5 TiB max (table: 4.995 TiB); single-part upload max 5 GiB; max 10,000 multipart parts. (The page's footnote 4 says "4.995 GiB", contradicting the table's TiB figure — internal inconsistency in the doc.)
- Object key: 1,024 bytes; object metadata: 8,192 bytes.
- Concurrent writes to the same object key: 1 per second (higher returns HTTP 429).
- Bucket-management ops: 50/s per bucket; Cloudflare REST API: 1,200 requests per 5 minutes per account; r2.dev is rate-limited/not for production.
- Limits can be raised via a Limit Increase Request Form. No per-bucket read-request limit is listed.

**S3 API compatibility** — [R2 S3 API](https://developers.cloudflare.com/r2/api/s3/api/):
- Supported: ListObjectsV2 (recommended over ListObjects), GetObject, HeadObject, PutObject, DeleteObject(s), CopyObject, full multipart set, bucket CORS/lifecycle/encryption/location.
- Unsupported: object/bucket tagging, versioning, ACLs, policies, replication, notification, website, logging, object lock, analytics/inventory/metrics configs, public-access-block, ownership controls.
- Checksums: CRC64NVME supported only as `FULL_OBJECT`; CRC32, CRC32C, SHA1, SHA256 only as `COMPOSITE`. `Content-MD5` supported on PutObject, UploadPart, CreateMultipartUpload.
- Conditional requests: GET/HEAD/PUT/CopyObject support `If-Match`, `If-None-Match`, etc.; PutObject conditions are atomic (only one concurrent `If-None-Match: *` wins, others get 412). UploadPartCopy does not support conditional headers.
- Multipart quirk: re-uploading an existing part number replaces it; if that re-upload fails "the original part is lost and must be re-uploaded."
- Region: `auto` (empty or `us-east-1` treated as aliases). The page states "Feature implementation is currently in progress."
- Release note: CRC-64/NVME supported for single and multipart objects from 2025-07-03 — [R2 release notes](https://developers.cloudflare.com/r2/platform/release-notes/) (the fetched notes list only four 2025–2026 entries, the latest being a 2026-04-27 dashboard "empty bucket" feature).
- JuiceFS documentation says R2's ListObjects result "is not sorted" and that `juicefs gc/fsck/sync/destroy` therefore do not work on R2 — [JuiceFS object-storage docs](https://juicefs.com/docs/community/reference/how_to_set_up_object_storage/).
- Cloudflare decompresses objects uploaded with `Content-Encoding: gzip` by default (deviation from AWS); upload with `Cache-Control: no-transform` to avoid — [rclone S3 docs, Cloudflare R2](https://rclone.org/s3/#cloudflare-r2).

**Endpoint format, location, jurisdiction**:
- Default endpoint `https://<ACCOUNT_ID>.r2.cloudflarestorage.com`; boto3 uses `region_name="auto"` — [Cloudflare boto3 example](https://developers.cloudflare.com/r2/examples/aws/boto3/), [Cloudflare rclone example](https://developers.cloudflare.com/r2/examples/rclone/).
- Location hints (best effort, performance only, honoured at first creation of a bucket name): `wnam, enam, weur, eeur, apac, oc`. Jurisdictions (guaranteed residency, immutable after creation): `eu`, `fedramp` (Enterprise), and `us` (as returned by the docs fetch). Jurisdiction endpoint: `https://<ACCOUNT_ID>.<JURISDICTION>.r2.cloudflarestorage.com` — [R2 data location](https://developers.cloudflare.com/r2/buckets/data-location/).

**Lifecycle / multipart hygiene** — [R2 object lifecycles](https://developers.cloudflare.com/r2/buckets/object-lifecycles/): every bucket has a default rule that aborts incomplete multipart uploads 7 days after initiation; custom rules: expire after N days, expire on date, `AbortIncompleteMultipartUpload`, transition to STANDARD_IA (each transition is one Class A); max 1,000 rules per bucket.

**Recent performance-related features**:
- Local Uploads (open beta announced 2026-02-03): writes object data to a nearby location and replicates asynchronously to the bucket; Cloudflare's own tests claim up to 75% lower time-to-last-byte for uploads from a different region than the bucket; no extra charge (uploads still bill Class A); not supported for jurisdiction buckets; reads from near the bucket may see cross-region latency until replication finishes — [R2 Local Uploads](https://developers.cloudflare.com/r2/buckets/local-uploads/), [changelog 2026-02-03](https://developers.cloudflare.com/changelog/post/2026-02-03-r2-local-uploads/).
- Super Slurper: one-off bulk migration into R2 from S3, GCS, R2 and S3-compatible providers; skips objects >1 TB; migrated ETags may differ — [Super Slurper docs](https://developers.cloudflare.com/r2/data-migration/super-slurper/). Sippy: only the pricing-page note (free, Class A billed) was verified.
- R2 Data Catalog: managed Apache Iceberg catalog on R2, described as GA; the fetched page (via a summarizer) called it "Basin Catalog, formerly R2 Data Catalog" — a rename I could not independently confirm. Aimed at analytics tables, irrelevant for raw image storage — [R2 Data Catalog](https://developers.cloudflare.com/r2/data-catalog/).
- Cloudflare Containers can mount R2 as FUSE (tigrisfs, s3fs, gcsfuse examples), announced 2025-11-21 — this applies to Cloudflare's own container platform, not third-party GPU rentals — [changelog](https://developers.cloudflare.com/changelog/post/2025-11-21-fuse-support-in-containers/index.md).

### Inferences
- [CALC] Storage: 30 GB dataset × $0.015 = $0.45/month before the 10 GB free allowance (≈ $0.30/month after it). 200k objects of PUT = 200k Class A = $0.90 gross, but within the 1M/month free Class A.
- The documented 1 write/s/key limit means "overwrite last.pt every epoch" is safe as long as an epoch takes >1 s, but two containers writing the same key (e.g., a zombie old container plus a restarted one) will collide with last-writer-wins semantics; use a per-run unique prefix or conditional writes to detect a second writer.
- GeeseFS's README says R2 "throttles with 403 instead of 429 if you exceed 5 requests per second" (see §3). That is an undated, uncorroborated claim and is contradicted by the 87M-file anecdote in §5 (≈700 PUT/s with s5cmd, no 429/403 reported). Treat as unproven, but set retry/backoff conservatively.
- Location hint should match the GPU provider's region (e.g., `enam` for US-East datacenters, `weur` for Europe) — hints are best-effort and only affect first creation of the bucket name.

### Gaps
- Minimum/maximum multipart part size on R2 is not stated on the fetched limits page (S3 norm is 5 MiB min); the commonly reported "all parts must be equal size except the last" R2 rule was not verified here.
- R2 does not publish a per-bucket GET rate limit that I could find; Tigris/third-party latency numbers disagree (see §4).
- Whether R2's ETag for single-part PutObject is the body MD5 is not documented on the pages fetched (it is S3-standard behaviour); multipart ETags are treated as unreliable (see §7).

---

## 3. Mount tools against R2: rclone mount, s3fs-fuse, goofys, GeeseFS (TigrisFS), mountpoint-s3, JuiceFS

### Takeaway
rclone mount is the only mature, first-class-R2 option (explicit `provider = Cloudflare`), but every FUSE tool here needs `/dev/fuse` and `CAP_SYS_ADMIN` in Docker. mountpoint-s3 does not officially target R2 and has a fresh 1.24.0 compatibility regression for non-AWS stores; GeeseFS's README lists R2 as "broken"; goofys is effectively unmaintained; JuiceFS cannot expose existing R2 objects as plain files and has R2-listing caveats.

### Cited Findings
**rclone mount** (v1.76.0, 2026-11-06):
- Config: `type = s3`, `provider = Cloudflare`, `region = auto`, `endpoint = https://<ACCOUNT_ID>.r2.cloudflarestorage.com`, `acl = private`. For R2 tokens with "Object Read & Write" scope add `no_check_bucket = true` or object uploads may fail; use rclone ≥ 1.59 (older returns HTTP 401) — [rclone S3 docs §Cloudflare R2](https://rclone.org/s3/#cloudflare-r2), [Cloudflare rclone example](https://developers.cloudflare.com/r2/examples/rclone/).
- Provider quirks baked into rclone: `force_path_style: true`, `use_multipart_etag: false  # multipart ETags are random` — [rclone Cloudflare.yaml](https://github.com/rclone/rclone/blob/master/backend/s3/provider/Cloudflare.yaml).
- VFS cache modes: `off` (default; reads straight from remote, no on-disk cache), `minimal`, `writes`, `full` (all reads/writes buffered to disk as sparse files; only downloaded ranges are present). Cache defaults: `--vfs-cache-max-age 1h`, `--vfs-cache-poll-interval 1m`, `--vfs-write-back 5s`; the cache may exceed `--vfs-cache-max-size` because it is only checked every poll interval and open files can't be evicted; don't share one cache dir between two rclone instances — [rclone mount docs](https://rclone.org/commands/rclone_mount/).
- Read tuning flags: `--buffer-size` (RAM per open file), `--vfs-read-ahead` (disk), `--vfs-read-chunk-size` (default 128M), `--vfs-read-chunk-streams N` (parallel ranged reads; docs example `16` × `4M`; "performance scaled roughly as the setting" in AWS S3 tests), `--no-modtime` ("S3 and Swift benefit hugely … each read of the modification time takes a transaction"), `--read-only`, `--dir-cache-time` (default 5m), `--attr-timeout` — same page.
- `--read-only`: attempts to write fail silently — same page (line on read-only mounts).
- Docker: needs `--cap-add SYS_ADMIN --device /dev/fuse` (often also `--security-opt apparmor:unconfined`) — [rclone forum](https://forum.rclone.org/t/fusermount-permission-denied-in-docker-rclone/13914) (third-party/forum evidence, not an official rclone doc page).
- Old anecdote: `rclone mount` of R2 appearing to "hang" in 2022 was actually mount-as-root vs access-as-other-user, fixed with `--allow-other` — [rclone forum, 2022-10-25](https://forum.rclone.org/t/rclone-mount-s3-cloudflare-r2-hangs/33696) [ANECDOTE].

**s3fs-fuse** (v1.97, 2025-12-08): generic S3 mount (`-o url=https://… -o use_path_request_style`); README has no R2-specific section — [s3fs-fuse README](https://github.com/s3fs-fuse/s3fs-fuse). Cloudflare's own Containers changelog lists s3fs as a supported adapter for R2 — [changelog](https://developers.cloudflare.com/changelog/post/2025-11-21-fuse-support-in-containers/index.md). GeeseFS's comparison table characterises s3fs as supporting parallel readahead and disk cache but "without server-side update, still downloads whole file to update" [VENDOR comparison] — [GeeseFS README](https://github.com/yandex-cloud/geesefs).

**goofys**: last tagged release v0.24.0 (2020-04-04); repo not archived, last push 2024-07-18 — [goofys repo](https://github.com/kahing/goofys), [GitHub API](https://api.github.com/repos/kahing/goofys). GeeseFS is a fork of goofys (README).

**GeeseFS** (v0.43.9, 2026-09-03; actively maintained): "aggressive parallelism and asynchrony" for small files; no readahead on random reads; disk cache for reads. README explicitly lists **CloudFlare R2 under "Services known to be broken"**: "they return 403 Forbidden if you exceed 5 requests per seconds" instead of 429 — [GeeseFS README](https://github.com/yandex-cloud/geesefs) (undated claim). Requires `--list-type 2` for non-Yandex S3. **TigrisFS** (v1.2.4, 2026-09-29) is a GeeseFS-based fork by Tigris Data (a competitor of R2) — [TigrisFS README](https://github.com/tigrisdata/tigrisfs); Cloudflare's changelog uses TigrisFS in its R2-mount example (no throttling caveat given).

**mountpoint-s3** (1.24.0, 2026-08-24):
- Purpose-built for Amazon S3; its SEMANTICS and CONFIGURATION docs do not mention R2 or S3-compatible stores; `--endpoint-url` can override endpoint, `--force-path-style` and `--region` for signing — [SEMANTICS.md](https://github.com/awslabs/mountpoint-s3/blob/main/doc/SEMANTICS.md), [CONFIGURATION.md](https://github.com/awslabs/mountpoint-s3/blob/main/doc/CONFIGURATION.md).
- Semantics: random reads supported; sequential-only writes of new files; no rename/append (outside S3 Express); optional local data cache `--cache <dir>` (LRU, default keeps 5% free) with metadata TTL default 1 minute — same docs.
- Evidence it works with R2: Modal's `CloudBucketMount` builds on Mountpoint and documents R2 mounts (needs endpoint URL), "optimized for reading large files sequentially," small-file random access discouraged — [Modal cloud bucket mounts](https://modal.com/docs/guide/cloud-bucket-mounts) (a platform integration, not an AWS/Cloudflare compatibility statement).
- Regression risk: since 1.24.0 Mountpoint always adds `x-amz-optional-object-attributes: RestoreStatus` to ListObjectsV2; Ceph RGW resets the connection → mount fails; open issue with a proposed PR #2026 — [issue #2025](https://github.com/awslabs/mountpoint-s3/issues/2025) (2026-11-05). Whether R2 tolerates this header was not tested/found.
- GitHub issue search for "cloudflare"/"R2" in awslabs/mountpoint-s3 returned no R2-specific issue — [GitHub search API](https://api.github.com/search/issues?q=repo:awslabs/mountpoint-s3+cloudflare).

**JuiceFS over R2** (v1.4.2, 2026-10-22): `juicefs format --storage s3 --bucket https://<ACCOUNT_ID>.r2.cloudflarestorage.com/<bucket> …`; JuiceFS says R2 ListObjects is "not fully S3 compatible (result list is not sorted)" so `gc`, `fsck`, `sync`, `destroy` don't work; add `--backup-meta 0` to `juicefs mount`; SDK checksum env vars default to `when_required` for compatibility — [JuiceFS docs](https://juicefs.com/docs/community/reference/how_to_set_up_object_storage/).

**FUSE in containers (all tools)**: need `/dev/fuse` + `CAP_SYS_ADMIN` (sources under §1). Cloudflare's Containers platform grants FUSE; RunPod reportedly does not (§1).

### Inferences
- Maturity ranking for "R2 + read-only dataset mount": rclone (documented provider, frequent releases) > s3fs-fuse (mature, generic, slower on small files per GeeseFS's own — vendor — table) > TigrisFS/GeeseFS (fast design, but R2 declared broken by GeeseFS authors; TigrisFS example from Cloudflare suggests basic function) > mountpoint-s3 (excellent AWS-side performance, but unsupported for R2 and 1.24.0 sends a new header that non-AWS servers may reject) > JuiceFS (wrong model: it owns the bucket layout; background knowledge, not verified in this session: it stores data as chunked blocks plus a separate metadata engine, so existing plain image objects are not visible as files) > goofys (unmaintained).
- If a mount is nevertheless required, the least-risk configuration is rclone with `--read-only --vfs-cache-mode full --vfs-cache-max-size <disk budget> --vfs-cache-max-age 48h --no-modtime --vfs-read-chunk-streams 8 --dir-cache-time 1h`; `--vfs-cache-max-age` must exceed one epoch or files evicted at the default 1 h would be re-fetched mid-training (derived from the documented eviction rule).

### Gaps
- No R2-specific field report for mountpoint-s3, GeeseFS (post-2025) or s3fs-fuse with Ultralytics-style workloads was found.
- GeeseFS "5 requests/s → 403" claim could not be corroborated or dated.
- JuiceFS architecture statements in Inferences are background knowledge, not verified against its docs in this session.

---

## 4. Performance for many-small-file detection datasets: FUSE vs local NVMe, DataLoader effects, and Class B cost per epoch

### Takeaway
No public benchmark of rclone/s3fs/mountpoint vs local NVMe for YOLO training was found; the available evidence (analogous network-FS and S3 tests) consistently shows per-file latency, not bandwidth, dominating, and tar-sharding or local caching closing the gap. Class B cost of an uncached mount is small (roughly $0.01–$0.10 per epoch for 10k–200k images); latency and starvation of DataLoader workers, not money, is the problem.

### Cited Findings
- [VENDOR/ANECDOTE, secondary snippet] A UCloud case study of PyTorch image loading from a network FS (UFS) vs local SSD: CV2 72 vs 555 images/s, PIL 115 vs 3,508 images/s; packing small images into one file with an index brought UFS to or above SSD speed — [besthub.dev write-up](https://www.besthub.dev/articles/why-does-pytorch-struggle-with-ufs-storage-insights-and-optimizations-34886c1aef24). Not s3fs/rclone; analogy only.
- [VENDOR slides] Training directly from S3-FUSE spent >80% of time in the DataLoader (GPU util <20%, ~17%); with a local FUSE cache layer (Alluxio) DataLoader share fell 82%→1% and GPU util rose 17%→93% — [Alluxio/Ray meetup slides](https://www.slideshare.net/slideshow/data-infra-meetup-accelerate-distributed-pytorchray-workloads-in-then-cloud/265944354).
- [VENDOR] Tigris benchmark (g5.8xlarge, A10G, 100k JPEGs ≈115 KB ≈ 10 GB, S3 connector for PyTorch, not FUSE): per-object random access scaled from ~12 samples/s at 1 worker to ~134 at 16 workers then flattened; tar shards of 4–256 MB reached ~133–135 samples/s with only 8 workers; a warm local cache cut epoch time 5.7×; AWS S3 baseline ~138 samples/s — [Tigris blog](https://www.tigrisdata.com/blog/training-object-storage). (Bottleneck there was the ViT model, so these show the floor needed to saturate that GPU, not what a YOLO-n would need.)
- [Academic] "Profiling and Improving the PyTorch Dataloader for high-latency Storage": a concurrent-fetch DataLoader gives up to 12× faster batch loading on S3-like storage, making object-storage training time comparable to local drives (abstract only was read) — [arXiv 2211.04908](https://arxiv.org/abs/2211.04908).
- R2 small-object latency is disputed: a Tigris (competitor) benchmark reports R2 1 KB reads at P50 ≈ 606 ms / P90 ≈ 681 ms (client location, concurrency and date not stated; its R2 read and write rows are identical, suggesting a copy error) — [Tigris R2 benchmark](https://www.tigrisdata.com/docs/overview/benchmarks/cloudflare-r2/) [VENDOR]; an anonymous blog reports ~38 ms cold TTFB for a 10 KB asset from R2 in us-east-1 [ANECDOTE, methodology unstated] (via search results; same query surfaced no authoritative figure).
- Ultralytics defaults: `workers=8`; the actual dataloader worker count is `min(os.cpu_count() // n_gpus, workers, n_batches)` — [train docs](https://docs.ultralytics.com/modes/train/), [CODE data/build.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/data/build.py).
- Image access pattern (for counting GETs): each training sample loads one image from disk; Mosaic draws its additional images from the in-RAM `buffer` when `cache != "ram"`, otherwise from the whole dataset (in-RAM anyway) — [CODE augment.py `Mosaic.get_indexes`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/data/augment.py). Validation (no augmentation) reads every val image once per validation pass (base.py `load_image` has no buffer when `augment=False`).
- First-epoch label scan opens every image twice (`PIL.Image.open` + `verify()` in `check_image`, then a second `open(..., "rb")` to check the JPEG end marker) and reads every label file — [CODE data/utils.py `check_image`, `verify_image_label`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/data/utils.py). The cache hash `get_hash` does `os.stat` (size + mtime_ns) of every label and image file — same file, and [CODE dataset.py `get_cache_hash`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/data/dataset.py).
- Class B definition/price: HeadObject, GetObject (and range GETs) are Class B at $0.36/M; ListObjects is Class A at $4.50/M; 10M Class B/month free — [R2 pricing](https://developers.cloudflare.com/r2/pricing/).

### Inferences
- **[CALC] Class B ops and cost per epoch, mounted without a data cache.**
  Inputs: price $0.36 per 1M Class B; free tier 10M Class B/month (Standard, shared across all account usage); val set = 10% of train; ops per file access k = 1 (single GET; best case) to 3 (GET + HEAD for stat/open + a second open); reads per epoch = (N_train + N_val) × k, based on one primary image read per training sample (mosaic extras come from the RAM buffer) plus one read per val image per validation pass.

  | N_train + N_val | Ops/epoch k=1 | Ops/epoch k=3 | $/epoch (k=1 → k=3) | 100 epochs, gross (k=1 → k=3) | 100 epochs, after 10M free (k=1 → k=3) |
  |---|---|---|---|---|---|
  | 10k + 1k = 11k | 11k | 33k | $0.004 → $0.012 | $0.40 → $1.19 | $0 → $0 |
  | 50k + 5k = 55k | 55k | 165k | $0.020 → $0.059 | $1.98 → $5.94 | $0 → $2.34 |
  | 200k + 20k = 220k | 220k | 660k | $0.079 → $0.238 | $7.92 → $23.76 | $4.32 → $20.16 |

  Plus, if `labels.cache` can't be saved (read-only mount) or its hash mismatches, one rescan per start ≈ 2–3 opens × N images (≈ 0.4–0.6M ops for 200k images ≈ $0.14–$0.22 per start). Listing: ~1 ListObjectsV2 per 1,000 objects, e.g., 200k images + 200k labels ≈ 400 Class A ≈ $0.002.
  **Copy-at-startup comparison:** one GET per file once per container start: 200k files ≈ 200k Class B ≈ $0.07 (or 1 GET with a tar shard). Conclusion: R2 fees are noise for both strategies; do not choose on cost.
- **[CALC] Latency-bound throughput ceiling for an uncached mount.** Assumption (not measured): each image read through FUSE+R2 from a rented container costs L = 50–100 ms end-to-end including open/stat. Each DataLoader worker is sequential, so ceiling ≈ workers / L = 8 / 0.05–0.10 s = 80–160 images/s (and less if the container has fewer than 8 vCPUs, since `nw` is capped by `os.cpu_count()`). The vendor-implied YOLO26n training throughput on an RTX 4090 is ≥ ~200 images/s (see §8), so the GPU would be starved. Only a measurement on the chosen provider can pin L; the Tigris numbers above (≈ 83 ms/sample/worker implied by 12 samples/s at 1 worker, S3 not R2, includes ViT preprocessing) are consistent with L in this range.
- Kernel readahead and rclone read-ahead help sequential big-file reads, not 100–200 KB independent JPEGs; mosaic random access defeats sequential prefetch. Hence the mount would need `--vfs-cache-mode full` to be useful, which is functionally "lazy copy-at-startup" plus a daemon.
- With `cache="ram"`, the first epoch pays all reads once (parallel `ThreadPool` of `NUM_THREADS`, [CODE base.py `cache_images`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/data/base.py)); afterwards the filesystem is irrelevant — but RAM may not fit (see §6).
- **Tar/WebDataset note:** Tigris's data shows tar shards reach the same throughput with half the workers, because a shard is one sequential stream; this supports packing for the *download* step even though Ultralytics itself cannot read tar/WebDataset during training (it needs extracted image files).

### Gaps
- No measured FUSE-vs-NVMe images/s for Ultralytics YOLO with rclone/s3fs/mountpoint against R2 exists in what I found; the real L on the user's provider must be benchmarked (e.g., `rclone cat`/`curl` timing for 100 random objects, then a one-epoch dry run).
- The k (GETs per file through each FUSE tool) was not measured; k=1–3 is a bracket.
- The arXiv paper's detailed numbers were not read (abstract only).

---

## 5. Copy-at-startup tools: rclone copy/sync, s5cmd, aws s3 sync; reported throughput; tar/zip packing

### Takeaway
For many small objects, s5cmd (large worker pool) is reported fastest, rclone with high `--transfers` is acceptable, aws-cli is slowest — but all reported numbers are *uploads* of small files to R2, not downloads. Packing the dataset into a handful of ~0.5–2 GB tar shards turns a files/s-bound problem into a bandwidth-bound one and cuts operations from N to ~dozens. Plain tar (not gzip/zip) is appropriate for JPEG.

### Cited Findings
- **s5cmd** (v2.3.0, 2024-12-16 — no release in ~11 months): README claims "for uploads 32× faster than s3cmd and 12× faster than aws-cli"; downloads "can saturate a 40 Gbps link (~4.3 GB/s)" vs aws-cli 375 MB/s [VENDOR]. `--numworkers` default 256 (global worker pool), `cp --concurrency` default 5 parts per file; `s5cmd run` executes command files in parallel; `--endpoint-url` / `S3_ENDPOINT_URL` for S3-compatible stores (path-style fallback with a custom endpoint); retries 10× with backoff up to a minute (`--retry-count`); checks `Content-MD5` and `X-Amz-Content-Sha256` on upload; `sync` supports `--delete`; `cat` and `pipe` commands allow streaming — [s5cmd README](https://github.com/peak/s5cmd). Cloudflare's own s5cmd example page was not found (404 at the guessed URL).
- **87M small files to R2 [ANECDOTE, 2025-12-02]**: AWS CLI ≈ 50K files after 1 h (projected 1,740 h); rclone `--transfers 64 --fast-list` 50K–600K files/h (projected 145 h; spent time walking the tree and checking size/checksum); s5cmd 2.6M files in the first hour, "+4.5M/h" later (projected 33 h). No 429/throttling reported. These are uploads of small protobuf tiles — [ntbies write-up](https://ntbies-website.pages.dev/blog/article/2025-12-02-from-72-days-to-33-hours-uploading-87-millions-files-to-cloudflare-r2).
- **rclone to R2 large files [ANECDOTE]**: on a 10 Gbps server a 2 GB upload stalled near 20 MB/s, reached ~110 MB/s with `--ignore-checksum`, tuned `--transfers/--checkers` and `--fast-list`; a 10 GB file reached ~250 MB/s — [rclone forum 35350](https://forum.rclone.org/t/slow-upload-speed-for-10gbps-connection/35350) (from search excerpt).
- rclone R2 guidance: each multipart part is a separate Class A op; a multipart upload uses ≥ 3× the operations of a single PutObject; tune with `--s3-chunk-size` / `--s3-upload-cutoff` — [Cloudflare rclone example](https://developers.cloudflare.com/r2/examples/rclone/). rclone memory for multipart ≈ `--transfers × --s3-upload-concurrency × --s3-chunk-size`; docs suggest ~16M chunk and concurrency 8 as starting point — [rclone S3 docs](https://rclone.org/s3/).
- Ultralytics will accept a `.zip`/`.tar` as `data=` and unzip it into `DATASETS_DIR` (`check_det_dataset` → `safe_download(..., unzip=True)`) — [CODE data/utils.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/data/utils.py) — but this is a convenience, not a streaming reader; images are always read as files afterwards.
- JuiceFS sets the AWS SDK checksum environment variables to `when_required` by default "for better compatibility with S3-compatible services" (AWS users can opt into `when_supported`) — [JuiceFS docs](https://juicefs.com/docs/community/reference/how_to_set_up_object_storage/). This hints that default "always send checksum" SDK behaviour can be problematic on some S3-compatible stores; R2 itself documents composite/CRC64NVME checksum support (§2), so whether it is a problem on R2 is untested here.

### Inferences
- **[CALC] Download-time estimates** (inputs are assumptions; measure): dataset 200k JPEG × 150 KB ≈ 30 GB; container network 1 Gbit/s ≈ 110 MB/s usable → bandwidth floor ≈ 270 s (≈ 4.5 min) for the tar path if the link is saturated by parallel shard streams; at 10 Gbit/s ≈ 30–60 s. For per-file sync, files/s becomes binding: at 200 files/s (low end of rclone anecdote, 64 transfers) 200k files ≈ 1,000 s (≈ 17 min); at 700 files/s (s5cmd-like) ≈ 290 s (≈ 5 min). Using upload anecdotes as a proxy for download rates is an assumption.
- **Packing recommendation (own design):** pack `images/` and `labels/` for each split into ~1 GB plain `.tar` shards (JPEG is already compressed, so gzip/zip adds CPU for no size gain — background knowledge), upload to `datasets/<name>/<version>/shard-0000.tar …` plus `manifest.json` with sha256 per shard. Startup: download shards in parallel (`s5cmd cp`/`rclone copy --transfers 8`; for very large single shards `--multi-thread-streams` or s5cmd `--concurrency`), verify sha256, `tar -xf` in parallel to NVMe, `touch .ready`. Operations: ~30 GETs instead of 200k; extraction speed is local-disk bound.
- **What stays cheap:** a container restart repeats the whole copy; at ≤30 GB and free egress that costs minutes, not dollars.
- Do not ship `labels.cache` inside the shards (hash includes mtime_ns of every file, so it will mismatch unless tar preserves mtimes and is extracted exactly; rebuilding it costs one local scan of N images, well under a few minutes on NVMe). Let Ultralytics write it locally on first start.
- s5cmd vs rclone: pick **rclone** if you want one tool for download + checkpoint upload with R2 as a documented provider, plus `--checksum`; pick **s5cmd** for >100k small-object transfers. With tar shards the difference becomes immaterial. Because s5cmd has had no release since 2024-12 and R2 compatibility is only anecdotal, rclone is the lower-risk single-tool choice. (Judgement.)

### Gaps
- No file/s download measurement from R2 for rclone, s5cmd or aws-cli was found (only uploads).
- aws-cli v2 / boto3 ≥ 1.36 default-checksum behaviour against R2 was not verified; if using boto3, set the SDK to `when_required` (see §7) or test first.
- rclone defaults for `--checkers` and the exact `--transfers` default were not re-verified (the 87M article states default 4).
- Throughput of the user's rental container network is unknown and varies by provider.

---

## 6. Ultralytics YOLO specifics (version, dataset YAML, cache, labels.cache, resume, save_period, AMP, workers, callbacks, licensing)

### Takeaway
As of 2026-11-09 the current release is `ultralytics 8.4.208` (PyPI/GitHub, 2026-11-08), docs are titled for YOLO26 (released January 2026); YOLO11/YOLO12/YOLO26 all train through the same trainer. Key operational facts: `last.pt` is written non-atomically each epoch; `resume=True` rebuilds args from the checkpoint (only a small allow-list can be overridden); `labels.cache` is actually `<labels-dir>.cache`, written beside the labels and silently skipped (with a warning) if that directory isn't writable; `cache="disk"` writes uncompressed `.npy` files next to each image and also silently disables itself if not writable; no built-in S3 upload callback exists; weights are AGPL-3.0 by Ultralytics' position.

### Cited Findings
- **Version/cadence**: 8.4.208 released 2026-11-08; PyPI shows 8.4.203 (2026-11-05) through 8.4.208 (2026-11-08), i.e. six releases in four days — [PyPI JSON](https://pypi.org/pypi/ultralytics/json), [GitHub release](https://github.com/ultralytics/ultralytics/releases/tag/v8.4.208). License field AGPL-3.0.
- **YOLO26**: released January 2026; sizes n/s/m/l/x = 2.4/9.5/20.4/24.8/55.7M params, 5.5/20.9/68.4/86.8/194.4 GFLOPs, COCO mAP50-95 40.9/48.6/53.1/55.0/57.5, T4 TensorRT10 latency 1.7/2.5/4.7/6.2/11.8 ms; MuSGD optimizer, DFL removed, optional NMS-free head — [YOLO26 docs](https://docs.ultralytics.com/models/yolo26/).
- **Train args (docs, YOLO26 page)**: `resume` default False; `save_period` default -1 (disabled); `cache` False/`ram`(True)/`disk`; `workers` default 8; `amp` default True (`"fp16"`, `"bf16"`, `"fp32"` accepted); `batch=-1` auto-sizes to ~60% CUDA memory, fraction e.g. 0.70; `epochs` 100; `patience` 100; `close_mosaic` 10; `exist_ok` False; `time` (hours) overrides epochs; "checkpoints are saved at the end of each epoch by default, so at least one epoch must complete before a resume is possible"; the docs page does not mention network storage — [train docs](https://docs.ultralytics.com/modes/train/).
- **Dataset YAML path resolution**: a relative `path:` resolves against `DATASETS_DIR`; absolute paths used as-is — [CODE data/utils.py `check_det_dataset`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/data/utils.py).
- **labels.cache location and read-only behaviour**: `cache_path = Path(label_files[0]).parent.with_suffix(".cache")`, i.e. `<dataset>/labels/train.cache` for layout `labels/train/*.txt`, or `<dataset>/labels.cache` for flat `labels/*.txt`. `save_dataset_cache_file` checks `is_dir_writeable(path.parent)`; if not writable it logs "Cache directory … is not writable, cache not saved" and continues (rescan next run); a partial write is unlinked on exception. Cache validity = version match + `hash == get_hash(label_files + im_files + [scan_args])` where `get_hash` hashes each path with `st_size:st_mtime_ns`; a missing/stale/truncated cache triggers rescan — [CODE data/dataset.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/data/dataset.py) (`get_labels`, `_load_or_scan_cache`, `get_cache_hash`), [CODE data/utils.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/data/utils.py) (`save_dataset_cache_file`, `get_hash`).
- **cache modes** [CODE base.py]: `cache="ram"` loads/resizes all images into RAM at dataset init via `ThreadPool` (RAM check `check_cache_ram`); `cache="disk"` saves each image as `<image>.npy` next to the image (via `cache_images_to_disk`), after `check_cache_disk` samples 30 images, aborts with "directory not writable" if the image directory isn't writable, and requires free disk ≥ estimated size × 1.1; an existing `.npy` is used in any cache mode if not older than its image — [base.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/data/base.py).
- **Checkpoint writing** [CODE trainer.py `save_model`]: serialises to an in-memory buffer, then `self.last.write_bytes(buf)` (non-atomic, no temp-file + rename), `best.pt` if `best_fitness == fitness`, and `epoch{N}.pt` if `save_period > 0 and epoch % save_period == 0`. Checkpoint dict contains `epoch, best_fitness, stopper{best_fitness,best_epoch}, ema (fp16), updates, optimizer (fp16-converted), scaler, train_args, train_metrics, train_results, date, version, git, license` — [trainer.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/trainer.py).
- **resume semantics** [CODE trainer.py `check_resume`, `resume_training`]: `resume=True` (or a path to a `.pt`) → loads checkpoint, rebuilds args from `ckpt.train_args`; if `data=` is passed it overrides the checkpoint's data path; allowed overrides: `imgsz, batch, device, close_mosaic, augmentations, save_period, workers, cache, patience, time, freeze, val, plots, channels_last, distill_model, save_dir` (other differing args are ignored with a warning). `start_epoch = ckpt.epoch + 1` and `assert 0 < start_epoch < epochs` else "training … is finished, nothing to resume". If the file can't be found/loaded: `FileNotFoundError("Resume checkpoint not found…")`. After a run ends, `final_eval` calls `strip_optimizer(last.pt)` which sets `epoch = -1`, drops optimizer/EMA/scaler/updates ([CODE utils/torch_utils.py `strip_optimizer`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/torch_utils.py)) → a finished run's `last.pt` cannot be resumed. `save_dir` and `project/name` come from the checkpoint's args (`get_save_dir` returns `args.save_dir` if set).
- **AMP**: `check_amp` in current `main` documents that "no pretrained weights are downloaded" for its check — [CODE utils/checks.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/checks.py). (Older releases downloaded a model for this check; not re-verified by version.)
- **Callbacks**: full list `on_pretrain_routine_start/end, on_train_start, on_train_epoch_start/end, on_train_batch_start/end, on_model_save, on_fit_epoch_end, on_train_end, teardown, …`; `on_model_save` "runs when a checkpoint is saved, after validation"; `on_fit_epoch_end` after validation and any save; it also runs during final best-model evaluation (no save then) — [callbacks docs](https://docs.ultralytics.com/usage/callbacks/). In the loop, `on_model_save` is invoked only when `save_model()` returns True, on RANK -1/0 only — [CODE trainer.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/trainer.py). The built-in MLflow callback logs `trainer.best.parent` (weights dir) via `mlflow.log_artifact` only in `on_train_end` — [CODE callbacks/mlflow.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/mlflow.py) — useless for mid-run preemption.
- **Ultralytics Platform (cloud training)**: uploads best checkpoint periodically; "Retry Training" after failure is a rerun with the same config, not a resume — [Platform training docs](https://docs.ultralytics.com/platform/train).
- **Licensing (brief flag; not legal advice)**: Ultralytics states code/models are AGPL-3.0 or Enterprise; "any commercial product or service" and "internal business tools" are listed as needing an Enterprise License unless the whole project is open-sourced under AGPL; "All Ultralytics YOLO trained models fall under the AGPL-3.0 License by default," including trained-from-scratch; the page does not quote the AGPL network-use clause — [Ultralytics license page](https://ultralytics.com/license).
- **Network-filesystem issues**: a targeted GitHub-issue search did not surface any Ultralytics issue specifically about NFS/FUSE datasets or read-only `labels.cache` (YOLOv5 threads #11367, #7481 about slow scanning attributed by maintainers to hardware/dataset/augmentation) — [yolov5 #11367](https://github.com/ultralytics/yolov5/issues/11367), [#7481](https://github.com/ultralytics/yolov5/issues/7481).

### Inferences
- **Read-only mount consequences (CODE-derived):** (a) `labels.cache` not saved → every start re-scans all N images/labels over the mount (see §4 cost/latency); (b) `cache="disk"` silently becomes `None` ("directory not writable"); (c) `cache="ram"` still works (reads only). Writing the dataset to local disk avoids all three.
- **[CALC] RAM cache sizing:** at `imgsz=640` an average resized image of ~640×480×3 ≈ 0.92 MB uncompressed → 10k ≈ 9 GB, 50k ≈ 46 GB, 100k ≈ 92 GB, 200k ≈ 184 GB. Disk-cache `.npy` is the same uncompressed size (≈ 6× a 150 KB JPEG). So `cache="ram"` is viable only for ≲30–40k images on typical 64 GB rental hosts, and container memory limits may be lower than host RAM (unverified whether `check_cache_ram` sees cgroup limits). For ≥50k images prefer no cache + local NVMe + enough workers, or `cache="disk"` only if disk is plentiful and decoding (not I/O) is the bottleneck.
- **Pin the Ultralytics version** in the container image (e.g. `pip install ultralytics==8.4.208`) and pre-bundle the base weights: `main` moves several times per week, trainer internals (e.g., `save_model`, resume allow-list) change between releases, and a resumed run should use the same version as the interrupted one (checkpoint stores `version`).
- **Resume gotchas to script around**: (1) use an absolute `path:` in the dataset YAML and the same absolute `project`/`name` (e.g., `/workspace/runs/exp1`, `exist_ok=True`) on every launch so the restored `last.pt` lands at `…/weights/last.pt` and args/`save_dir` agree; (2) always pass `data=` on resume if the dataset location could change; (3) avoid `time=` with resume (epochs recomputed from mean epoch time); (4) after normal completion, resume is impossible → the script must detect completion (R2 `DONE` marker or `epoch == -1`) and exit instead of calling train; (5) RNG/dataloader order is not restored, so a resumed run is not bit-identical (inference; not verified).
- **Callback to upload**: use `on_model_save` (fires only after a successful save) and read `trainer.last`, `trainer.best`, `trainer.epoch`; run synchronously — for n/s models checkpoint files are tens of MB so the upload takes seconds against epoch times of minutes, and it guarantees R2 is up to date before the next epoch starts. Checkpoint size is not documented; rough [CALC] estimate: the checkpoint stores EMA fp16 (2 B/param; `model` is None) plus optimizer state converted to fp16 (SGD-type: ~1 buffer ≈ 2 B/param; AdamW-type: `exp_avg`/`exp_avg_sq` are kept FP32 per `convert_optimizer_state_dict_to_fp16`, ≈ 8 B/param) ≈ 4–10 B/param → n (2.4M params) ≈ 10–24 MB, s (9.5M) ≈ 38–95 MB (pretrained checkpoints may carry larger param counts than the fused figures). Measure the real `last.pt` size once.
- **AGPL flag for the user**: if the trained weights/service are part of a commercial product, Ultralytics' stated position is that an Enterprise License (or full AGPL source release) is required; consider this before choosing the Ultralytics stack for commercial deployment. Alternatives (permissively licensed detectors) were out of scope.

### Gaps
- No release-note-level confirmation found that `last.pt` writing will become atomic; it is non-atomic in `main` as of 2026-11-09.
- Whether `check_cache_ram` respects cgroup memory limits was not verified.
- Exact `labels/train.cache` placement for the user's directory layout depends on how labels are laid out; the formula above is exact, the example paths are illustrative.
- DDP/multi-GPU resume behaviour was not examined (single GPU assumed).

---

## 7. Interruption-safe workflow: local checkpoints, atomic-ish upload to R2, idempotent restart, partial uploads, integrity checks

### Takeaway
Because R2 PutObject is strongly consistent and incomplete multipart uploads are never visible (and auto-abort after 7 days), the main risks are not on the R2 side but (1) uploading a half-written local `last.pt`, (2) trusting multipart ETags, and (3) resuming the wrong/finished run. Use single-part PUTs (cutoff ≥ file size), upload only after `on_model_save`, record a sha256 + epoch in a small manifest written last, and validate (`hash` + `torch.load` + epoch check) before resuming; fall back to the previous generation if validation fails.

### Cited Findings
- R2 strong consistency: writes, metadata updates, deletes and lists are immediately visible globally; last writer wins on concurrent PUT of the same key — [R2 consistency](https://developers.cloudflare.com/r2/reference/consistency/). Conditional PUT (`If-None-Match: *`, `If-Match`) is atomic at commit; a 404 does not prove an earlier in-flight PutObject will not later commit — same page.
- Multipart: visible only once complete (read-after-write "once all parts are uploaded"); incomplete multipart uploads are auto-aborted after 7 days by a default bucket rule, shorter via `AbortIncompleteMultipartUpload` lifecycle rule; `AbortMultipartUpload` and `DeleteObject` are free; a multipart upload costs ≥ 3 Class A ops vs 1 for single PutObject — [R2 consistency](https://developers.cloudflare.com/r2/reference/consistency/), [R2 lifecycles](https://developers.cloudflare.com/r2/buckets/object-lifecycles/), [R2 pricing](https://developers.cloudflare.com/r2/pricing/), [Cloudflare rclone example](https://developers.cloudflare.com/r2/examples/rclone/).
- Checksum features: `Content-MD5` honoured on PutObject/UploadPart (server rejects on mismatch — standard S3 semantics; R2 lists the header as supported); CRC64NVME (`FULL_OBJECT`) supported since 2025-07-03; CRC32/CRC32C/SHA1/SHA256 only composite — [R2 S3 API](https://developers.cloudflare.com/r2/api/s3/api/), [release notes](https://developers.cloudflare.com/r2/platform/release-notes/).
- rclone treats Cloudflare multipart ETags as unusable (`use_multipart_etag: false  # multipart ETags are random`) — [Cloudflare.yaml](https://github.com/rclone/rclone/blob/master/backend/s3/provider/Cloudflare.yaml). rclone stores an MD5 in `X-Amz-Meta-Md5chksum` for multipart (and SSE) objects, and by default re-checks the multipart ETag after upload unless `--s3-no-head` is set (the Cloudflare quirk disables the ETag-based check); `--s3-upload-cutoff` up to 5 GiB forces single-part (cutoff must be ≥ chunk size) — [rclone S3 docs](https://rclone.org/s3/), [Cloudflare rclone example](https://developers.cloudflare.com/r2/examples/rclone/). A forum thread (non-R2 gateway) shows the debate: ETag is "the only reliable verification" vs trusting rclone's metadata MD5 — [rclone forum 48861](https://forum.rclone.org/t/multipart-upload-corrupted-etag-differ/48861) [ANECDOTE].
- s5cmd verifies `Content-MD5`/`X-Amz-Content-Sha256` on upload; fails the upload on mismatch — [s5cmd README](https://github.com/peak/s5cmd).
- JuiceFS notes SDK checksum envs default to `when_required` for S3-compatible compatibility — [JuiceFS docs](https://juicefs.com/docs/community/reference/how_to_set_up_object_storage/). Cloudflare's boto3 page only shows `endpoint_url` and `region_name="auto"` and says nothing about checksum config — [boto3 example](https://developers.cloudflare.com/r2/examples/aws/boto3/).
- Ultralytics side facts (non-atomic `write_bytes`, `on_model_save` timing, resume assert, stripped final checkpoints): see §6 sources.
- R2 Local Uploads can speed checkpoint/dataset uploads from a region far from the bucket (not for jurisdiction buckets) — [Local Uploads docs](https://developers.cloudflare.com/r2/buckets/local-uploads/).

### Inferences
- **Why partial uploads are mostly a non-issue (inference from S3-style semantics + R2 consistency doc):** a single-part PutObject either commits or does not; a killed client leaves the previous object intact. Residual risks: orphaned multipart upload parts (cost nothing visible, auto-aborted in 7 days — set a 1-day lifecycle rule if desired) and the "in-flight PUT may still commit" subtlety if a zombie old container is still uploading while a replacement starts.
- **Recommended R2 layout (own design):**
  ```
  datasets/<name>/<ver>/shard-0000.tar … , manifest.json            # immutable; versioned by prefix
  runs/<exp>/ckpt/last-e0042.pt                                      # unique key per epoch (never overwritten)
  runs/<exp>/ckpt/best.pt                                            # overwritten when best changes
  runs/<exp>/manifest.json                                           # {"epoch":42,"key":"…/last-e0042.pt","sha256":"…","size":N,"ultralytics":"8.4.208"} written LAST
  runs/<exp>/DONE                                                    # written after final best.pt/last.pt upload
  ```
  Keep the last K=3 epoch objects (delete older ones; DeleteObject is free) or use an expiry lifecycle rule. Overwriting a single `last.pt` key is simpler and also safe; the versioned-key + manifest scheme only adds protection against a corrupted-but-successfully-uploaded `last.pt` (e.g., NaN-poisoned or truncated by a crash during the local write) and against two writers.
- **Upload step (inside `on_model_save`):** (1) wait until `trainer.last` exists and its size is stable / validate by `torch.load(path, map_location="cpu")` and `ckpt["epoch"] == trainer.epoch` (cheap for n/s models); (2) compute sha256; (3) single-part PUT with `Content-MD5` (boto3 `put_object(..., ContentMD5=...)` or rclone `copyto --s3-upload-cutoff 5G`); (4) HEAD to confirm size; (5) PUT `manifest.json` (small; atomic). If the upload fails, log and retry next epoch rather than crash training (a failed upload loses at most the epoch interval).
- **Restart step (idempotent entrypoint):** `DONE` present → print/exit success. Else if `manifest.json` present → download referenced key → verify sha256 + size → `torch.load` → `ckpt["epoch"] >= 0` (not stripped) → place at `/workspace/runs/<exp>/weights/last.pt` → `YOLO(that).train(resume=True, data=…)`. On any validation failure fall back to previous manifest generation (keep `manifest.prev.json`) or, if none, start fresh. No manifest → fresh start. Wrap the entrypoint in a restart loop (`until python train.py; do sleep 30; done`) only for crashes; platform preemptions need the orchestrator to re-run the same command.
- **boto3 sketch (own design, untested against R2):**
  ```python
  import boto3, hashlib, json, os, torch
  from boto3.s3.transfer import TransferConfig
  from botocore.config import Config
  # if you see checksum/header errors on R2, also export AWS_REQUEST_CHECKSUM_CALCULATION=when_required
  # and AWS_RESPONSE_CHECKSUM_VALIDATION=when_required (AWS SDK settings; not from Cloudflare docs)
  s3 = boto3.client("s3", endpoint_url=f"https://{ACCOUNT_ID}.r2.cloudflarestorage.com",
                    region_name="auto", config=Config(retries={"max_attempts": 8, "mode": "standard"}))
  ONE_PART = TransferConfig(multipart_threshold=2*1024**3)           # force single-part PUT for ckpts
  def on_model_save(trainer):
      p = trainer.last; ck = torch.load(p, map_location="cpu", weights_only=False)
      assert ck["epoch"] == trainer.epoch
      sha = hashlib.sha256(p.read_bytes()).hexdigest(); key = f"runs/{EXP}/ckpt/last-e{trainer.epoch:04d}.pt"
      s3.upload_file(str(p), BUCKET, key, Config=ONE_PART, ExtraArgs={"Metadata": {"sha256": sha}})
      s3.put_object(Bucket=BUCKET, Key=f"runs/{EXP}/manifest.json",
                    Body=json.dumps({"epoch": trainer.epoch, "key": key, "sha256": sha}).encode())
  model = YOLO(resume_path or "yolo26n.pt"); model.add_callback("on_model_save", on_model_save)
  model.train(data=DATA_YAML, epochs=100, project="/workspace/runs", name=EXP, exist_ok=True,
              resume=bool(resume_path), workers=8, cache=False, amp=True)
  ```
  (`weights_only=False` is needed because Ultralytics checkpoints contain pickled model objects; only load checkpoints you wrote yourself.) Verify the exact `trainer.epoch` semantics (0- vs 1-based) in your installed version before asserting equality.
- **Integrity checks, ranked:** (a) your own sha256 in the manifest + metadata (independent of R2's ETag semantics), (b) `Content-MD5` on single-part PUT so R2 rejects corrupted bodies, (c) rclone's built-in compare for single-part objects, (d) avoid multipart-ETag comparisons on R2 entirely. For dataset shards use the same sha256-in-manifest scheme.
- **Data-integrity of dataset itself:** extraction is guarded by `.ready` written only after all shards verify + extract; if a container dies mid-extract the marker is absent and the next start re-extracts into a clean directory (delete first).
- Multi-writer protection: put the run ID or a random launch token in the manifest and write it with `If-Match` on the previous ETag (supported by R2 conditional PutObject) if concurrent containers are plausible; otherwise unnecessary.

### Gaps
- R2 behaviour for `ContentMD5` mismatch (exact error) and whether boto3 ≥1.36 default CRC32 trailer works on R2 were not tested; Cloudflare docs fetched do not discuss it.
- Provider SIGTERM grace periods for flushing a final checkpoint are unknown; the design assumes only a per-epoch save granularity.
- The exact meaning of `trainer.epoch` at `on_model_save` time (0-based index of the just-finished epoch, per `save_model` writing `"epoch": self.epoch`) should be confirmed in a test run.

---

## 8. Sizing: which GPU for n/s YOLO models, epoch time, and public benchmarks (RTX 3060 / 3090 / 4090)

### Takeaway
For n/s models a single 24 GB consumer card (RTX 3090/4090 class, even a 3060 for small sets) is enough; more expensive GPUs (A100/H100) won't help because the pipeline is CPU/dataloader-bound (mosaic augmentation, JPEG decode; workers capped by vCPU count). Public per-epoch benchmarks for YOLO training on 3060/3090/4090 are essentially absent; the best documented anchors are Ultralytics Platform's own time estimates and a 2023 ClearML study. Measure one epoch on the chosen instance.

### Cited Findings
- Ultralytics Platform estimates [VENDOR]: ~2 minutes for 500 images, YOLO26n, 50 epochs on an RTX 4090; ~6 minutes for 1,000 images, YOLO26n, 100 epochs on an RTX PRO 6000 (actual varies) — [Platform training docs](https://docs.ultralytics.com/platform/train). The same page lists its cloud GPUs (RTX 3090 24 GB, RTX 4090 24 GB, L4, A100, H100, B200, …) with RTX PRO 6000 as its "recommended default."
- ClearML (published 2023-03-29, early YOLOv8, VisDrone dataset with large images, `yolov8n`, 10 epochs, imgsz 640): RTX 3060 Ti "well below the other machines in raw speed"; standard 3080/3090 instances CPU-bottlenecked, roughly matching a T4; the "Mem & CPU Optimized" 3090 ≈ AWS g5.2xlarge (A10G), which was the fastest; $/epoch (2023 prices): 3060Ti 0.0102, 3080 0.0106, 3090 0.0128, optimized 3080 0.0096, optimized 3090 0.0098, g4dn.xlarge 0.0174, g5.2xlarge 0.0208, V100 p3.2xlarge 0.0881; 4090 and A100 not tested; exact epoch times only in an image — [ClearML blog](https://clear.ml/blog/consumer-gpus-vs-datacenter-gpus-for-cv-the-surprising-cost-effective-winner).
- Model complexity (inference-side numbers, for relative scale): YOLO26n 5.5 GFLOPs / 1.7 ms T4-TRT; YOLO26s 20.9 GFLOPs / 2.5 ms; YOLO26m 68.4 GFLOPs / 4.7 ms — [YOLO26 docs](https://docs.ultralytics.com/models/yolo26/).
- Ultralytics community's "unofficial benchmark" thread contains inference-only timings for RTX 2060/4090/T4 (single runs on different stacks, not comparable) — [community.ultralytics.com](https://community.ultralytics.com/t/unofficial-benchmark-results-how-fast-can-you-yolo/59). Lambda reports generic 4090 vs 3090 training speedups of 1.3×–1.9× (TF32) depending on model, not YOLO-specific — [Lambda blog (via search excerpt)](https://lambda.ai/blog/nvidia-rtx-4090-vs-rtx-3090-deep-learning-benchmark).
- A GitHub discussion asks about an A100 COCO epoch of "20 minutes 36 seconds" vs an RTX 4090; the replies could not be retrieved — [ultralytics discussion #8075](https://github.com/orgs/ultralytics/discussions/8075).
- `workers` is capped by `os.cpu_count() // n_gpus` — [CODE build.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/data/build.py).

### Inferences
- **[CALC] Epoch time anchors from the Platform estimates.** Inputs: 500 imgs × 50 ep = 25,000 image-passes in ~120 s → ≈ 208 img/s (RTX 4090, YOLO26n); 1,000 × 100 = 100,000 passes in ~360 s → ≈ 278 img/s (RTX PRO 6000, YOLO26n). Both include startup, label scan and validation overhead on tiny datasets, so steady-state throughput is probably higher; treat 200–280 img/s as a conservative figure for YOLO26n at 640 px with adequate CPU. Resulting training-only epoch times (N_train / throughput): 10k → 36–50 s; 50k → 3.0–4.2 min; 100k → 6–8 min; 200k → 12–17 min (plus validation, typically 10% of N at higher per-image speed). YOLO26s has ~3.8× the FLOPs of n (20.9 vs 5.5 GFLOPs); if GPU-bound it would be slower than n, but on dataloader-bound setups the n/s gap may be small (inference; measure).
- **GPU choice:** RTX 3090 / 4090 (24 GB) are adequate for n/s at batch 32–128, imgsz 640; RTX 3060 (12 GB) works at smaller batch for n/s but is slower per ClearML's 3060 Ti ordering (not tested on 3060 itself). A100/H100 mostly add cost: with the dataloader as limiter (ClearML's finding that CPU/memory-optimised instances beat higher-spec GPUs with weak CPUs), pay for vCPU count (≥ 8–16 vCPUs per GPU, fast local NVMe) before paying for a faster GPU. The 4090's 1.3–1.9× generic training advantage over the 3090 (Lambda) will not materialise if the CPU pipeline saturates first.
- Dataloader math: with `workers = min(vCPU, 8)` and ~4–10 ms/ image CPU cost for decode + mosaic/affine (assumption), ~100–250 img/s per 1–2 cores is typical; hence 8 vCPUs ≈ the scale of the vendor throughput anchors above. A container with 4 vCPUs would be capped near half. Check `os.cpu_count()` inside the container and set `workers` accordingly.
- Rental-cost framing: for a 50k-image set at ~3–4 min/epoch × 100 epochs ≈ 5–7 GPU-hours; at an assumed $0.3–0.8/h for 3090/4090-class rentals ≈ $1.5–5.6 — price inputs are assumptions, not sourced.

### Gaps
- No public per-epoch training benchmark for YOLO11/12/26 n/s on RTX 3060, 3090 or 4090 with stated dataset size, batch, workers and cache was found (searches for community and vendor sources returned only inference benchmarks or unrelated distillation numbers).
- GPU rental price levels (Vast.ai/RunPod/etc.) were not researched; the $/h figures above are placeholders.
- 2023 ClearML data predates YOLO11/YOLO26 and current Ultralytics dataloader changes; its epoch-time chart was an image and not extractable.
