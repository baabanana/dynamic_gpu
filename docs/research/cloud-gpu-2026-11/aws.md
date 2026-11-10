# AWS options for ephemeral GPU training of small YOLO models (EC2 Spot, SageMaker Managed Spot Training, AWS Batch on Spot) — state as of 2026-11-09

Evidence tags used below: **(DOC)** = official AWS/Cloudflare/GitHub-maintainer documentation; **(DATA)** = numbers I computed/extracted on 2026-11-09 from machine-readable official or third-party datasets (URL given); **(3P)** = third-party/vendor/blog; **(ANECDOTE)** = individual user report (forum/re:Post), not authoritative. Prices are USD, Linux, On-Demand unless stated. Spot prices move continuously; Vantage "spot avg/min/max" is a snapshot over an unspecified recent window, so treat as order-of-magnitude. I did not run any AWS command; every command template below is assembled by me from documented flags and is marked as untested.

---

## Q1. Which GPU instance families suit small YOLO training, and what do they cost (On-Demand vs Spot, by region, plus EBS and data-transfer costs)?

### Takeaway
g4dn.xlarge (1x T4 16 GB, 4 vCPU/16 GiB) is the cheapest sensible single-GPU box: ~$0.526/h On-Demand, ~$0.25-0.34/h Spot in most regions (us-east-1 snapshot $0.252). g5.xlarge (A10G 24 GB, ~$0.51-0.72 Spot) and g6.xlarge (L4 24 GB, ~$0.42-0.65 Spot in cheaper regions) are the next steps up; g6e (L40S 48 GB), g7e (RTX PRO 6000 Blackwell 96 GB) and p3 (V100) are overkill or poor value for small YOLO. Extra costs (root EBS, public IPv4, ingress) are a few cents per hour; ingress from R2 is free on both sides.

### Cited Findings
**Instance specs**
- g4dn.xlarge: 1x NVIDIA T4 (16 GiB), 4 vCPU, 16 GiB RAM, up to 25 Gbps network, 125 GB local NVMe instance store, EBS up to 3,500 Mbps. g5.xlarge: 1x A10G (24 GiB), 4 vCPU, 16 GiB, up to 10 Gbps, 250 GB NVMe. g6.xlarge: 1x L4 (24 GiB), 4 vCPU, 16 GiB, up to 10 Gbps, 250 GB NVMe. g6e.xlarge: 1x L40S (48 GiB), 4 vCPU, 32 GiB, 250 GB NVMe. — (3P/DATA) [Vantage g4dn.xlarge](https://instances.vantage.sh/aws/ec2/g4dn.xlarge), [g5.xlarge](https://instances.vantage.sh/aws/ec2/g5.xlarge), [g6.xlarge](https://instances.vantage.sh/aws/ec2/g6.xlarge), [g6e.xlarge](https://instances.vantage.sh/aws/ec2/g6e.xlarge); GPU memory/vCPU/network of these sizes also match the official AWS Batch GPU table — (DOC) [AWS Batch GPU jobs](https://docs.aws.amazon.com/batch/latest/userguide/gpu-jobs.html)
- p3.2xlarge = 1x V100 16 GiB, 8 vCPU, 61 GiB; g7e.2xlarge = 1x 96 GiB GPU, 8 vCPU, 64 GiB (smallest g7e size is 2xlarge); g7.2xlarge = 1 GPU 32 GiB, 8 vCPU (g7 smallest listed size is 2xlarge). — (DOC) [AWS Batch GPU jobs table](https://docs.aws.amazon.com/batch/latest/userguide/gpu-jobs.html). GPU model for g7e (NVIDIA RTX PRO 6000 Blackwell) is from (3P) [Vantage dataset](https://instances.vantage.sh/instances.json); I found no official source naming the g7 GPU model.
- Newer families present in Nov 2026: g6f (fractional L4: g6f.large = 1/8 GPU, ~3 GiB GPU memory, 2 vCPU, 8 GiB), g7, g7e, gr6, p5/p5en, p6-b200, p6-b300. Spot Instance Advisor dataset includes g6f, g7e, p6-b200, p6-b300; AWS Batch docs list g7/g7e/p6. — (DATA) [Spot Advisor JSON](https://spot-bid-advisor.s3.amazonaws.com/spot-advisor-data.json), [Vantage dataset](https://instances.vantage.sh/instances.json), (DOC) [Batch GPU jobs](https://docs.aws.amazon.com/batch/latest/userguide/gpu-jobs.html)

**Official On-Demand prices, us-east-1 (AWS Price List API, published 2026-11-07)**
- g4dn.xlarge $0.526/h; g5.xlarge $1.006/h; g6.xlarge $0.8048/h; g6e.xlarge $1.861/h; g7e.2xlarge $3.36312/h; p3.2xlarge $3.06/h. — (DATA) [AmazonEC2 us-east-1 price list](https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonEC2/current/us-east-1/index.json)

**On-Demand vs Spot by region (Vantage snapshot 2026-11-09; "int" = Spot Advisor interruption bucket)** — (3P/DATA) [Vantage instances.json](https://instances.vantage.sh/instances.json), interruption bucket from (DATA) [Spot Advisor JSON](https://spot-bid-advisor.s3.amazonaws.com/spot-advisor-data.json). Spot = Vantage spot_avg (min-max).

| Region | g4dn.xlarge OD / Spot (int) | g5.xlarge OD / Spot (int) | g6.xlarge OD / Spot (int) | g6e.xlarge OD / Spot (int) |
|---|---|---|---|---|
| us-east-1 | 0.526 / 0.252 (0.225-0.267) (10-15%) | 1.006 / 0.513 (0.428-0.598) (>20%) | 0.805 / 0.589 (0.426-0.744) (>20%) | 1.861 / 1.684 (>20%) |
| us-east-2 | 0.526 / 0.278 (5-10%) | 1.006 / 0.547 (>20%) | 0.805 / 0.421 (10-15%) | 1.861 / 1.670 (15-20%) |
| us-west-2 | 0.526 / 0.253 (>20%) | 1.006 / 0.587 (>20%) | 0.805 / 0.648 (>20%) | 1.861 / 1.853 (>20%) |
| eu-west-1 | 0.587 / 0.329 (5-10%) | 1.123 / 0.636 (10-15%) | not listed | not listed |
| eu-central-1 | 0.658 / 0.306 (15-20%) | 1.258 / 0.618 (15-20%) | 1.006 / 0.772 (<5%) | 2.327 / 2.288 (>20%) |
| ap-northeast-1 (Tokyo) | 0.710 / 0.340 (10-15%) | 1.459 / 0.720 (>20%) | 1.167 / 0.580 (>20%) | 2.699 / 1.443 (>20%) |
| ap-southeast-1 (Singapore) | 0.736 / 0.311 (15-20%) | not listed | not listed | not listed |
| ap-south-1 (Mumbai) | 0.579 / 0.263 (15-20%) | 1.208 / 0.545 (>20%) | 0.966 / 0.466 (>20%) | 2.235 / 1.661 (n/a) |
| ap-northeast-2 (Seoul) | 0.647 / 0.326 (10-15%) | 1.237 / 0.581 (5-10%) | 0.990 / 0.465 (>20%) | 2.288 / 1.304 (>20%) |

- Other single-GPU data points (us-east-1, Vantage): g7e.2xlarge OD 3.363 / Spot avg 2.059 (15-20%); g6f.large OD 0.202 / Spot 0.074 (<5%); g6f.xlarge OD 0.2375 / Spot 0.091 (5-10%); p3.2xlarge OD 3.06 / Spot "avg" 0.428 (>20%, no min/max reported, so this figure may be stale). — (3P/DATA) [Vantage instances.json](https://instances.vantage.sh/instances.json)
- The Vantage page headline Spot prices for g4dn/g5/g6/g6e.xlarge ($0.252/$0.513/$0.589/$1.684) equal its us-east-1 averages, i.e. the page defaults to us-east-1. — (3P) [Vantage g4dn.xlarge](https://instances.vantage.sh/aws/ec2/g4dn.xlarge), [g6e.xlarge](https://instances.vantage.sh/aws/ec2/g6e.xlarge)
- Spot is not always a big discount: e.g. g6e.xlarge us-west-2 Spot avg 1.853 vs OD 1.861; g6.xlarge us-east-1 0.589 vs 0.805. — (3P/DATA) same Vantage dataset.
- If `MaxPrice` is not specified you pay the current Spot price; AWS "do[es] not recommend using this parameter because it can lead to increased interruptions". — (DOC) [run-instances CLI reference](https://docs.aws.amazon.com/cli/latest/reference/ec2/run-instances.html)

**Ancillary costs**
- EBS gp3 = $0.08 per GB-month in us-east-1 (gp2 $0.10); gp3 includes 3,000 IOPS and 125 MB/s free baseline. — (DATA) [EC2 price list](https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonEC2/current/us-east-1/index.json), (DOC) [EBS pricing](https://aws.amazon.com/ebs/pricing/)
- Public IPv4: $0.005 per in-use address-hour, including addresses on EC2 instances in a VPC. — (DOC) [VPC pricing](https://aws.amazon.com/vpc/pricing/)
- Data transfer IN from the internet: $0.000/GB. Data transfer OUT to internet: $0.09/GB first 10 TB/month (then $0.085, $0.07, $0.05); 100 GB/month free out, aggregated across AWS services and regions (excluding China/GovCloud). — (DATA) [AWSDataTransfer price list](https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AWSDataTransfer/current/us-east-1/index.json) (published 2026-11-03), (DOC) [EC2 On-Demand pricing page](https://aws.amazon.com/ec2/pricing/on-demand/)
- Spot billing: if EC2 interrupts a Linux Spot instance in its first hour: no charge; after the first hour: charged for seconds used. If you stop/terminate it yourself: seconds used. — (DOC) [Billing for interrupted Spot Instances](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/billing-for-interrupted-spot-instances.html)
- Terminated instance: charges stop as soon as state is `shutting-down` or `terminated`. — (DOC) [Terminate EC2 instances](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/terminating-instances.html)
- R2 side: egress to Internet free; storage $0.015/GB-month; Class A $4.50/M ops, Class B $0.36/M ops; free tier 10 GB-month + 1M Class A + 10M Class B per month. — (DOC) [Cloudflare R2 pricing](https://developers.cloudflare.com/r2/pricing/)

### Inferences
- (my arithmetic) g4dn.xlarge Spot us-east-1 ≈ $0.252 + 100 GB gp3 (100×0.08/730 ≈ $0.011/h) + IPv4 $0.005 ≈ **$0.27/h**, so a 4-hour run ≈ **$1.1**; On-Demand equivalent ≈ $0.54/h ≈ $2.2 for 4 h. g5.xlarge Spot us-east-1 ≈ $0.53/h all-in ≈ $2.1 for 4 h.
- g4dn.xlarge is the cost floor for a full (non-fractional) GPU; the 4-vCPU/16 GiB shape may bottleneck YOLO dataloaders (data augmentation is CPU-heavy) but I found no sourced YOLO benchmark across T4/A10G/L4. g4dn.xlarge/g5.xlarge also have free local NVMe (125/250 GB) usable as fast dataset scratch (lost on termination).
- Lower vCPU shapes keep the quota ask small: g4/g5/g6/g6e.xlarge = 4 vCPU each; p3.2xlarge and g7e.2xlarge = 8 vCPU.
- g6f (fractional L4, ~3 GiB) is dramatically cheaper (Spot ≈ $0.07-0.09/h) and might fit yolo-nano/small with small batch size, but I did not verify that CUDA training works on g6f or the effective VRAM; treat as an untested option.
- p3 is poor value here (OD $3.06 vs $0.526 for g4dn) and its Spot data looks thin.
- Region choice is a trade-off between price, interruption bucket and where R2 data sits; the cheapest-looking Spot (e.g., us-east-2 g6.xlarge $0.42) varies day to day, so check live Spot price history for chosen regions before committing.

### Gaps
- Live Spot price history per AZ was not retrievable without AWS credentials; Vantage averages cover an unspecified window.
- No sourced training-throughput benchmark of YOLO (n/s/m) on T4 vs A10G vs L4 vs L40S, so price/performance is unknown.
- DLAMI recommended root-volume size was not found in the fetched docs (assumed ≥100 GB in the cost example above; unverified).
- NAT gateway costs were not researched (a public subnet with public IPv4 avoids them).
- Regional On-Demand prices outside us-east-1 are from Vantage only (us-east-1 verified against official price list).

---

## Q2. New-account friction: default GPU vCPU quotas, how long increases take, are new accounts commonly rejected?

### Takeaway
Officially every GPU quota starts at 0 vCPU, per region, with separate buckets for Spot and On-Demand (and a third, separate set in SageMaker), so a brand-new account cannot launch even one g4dn.xlarge until a quota request is granted. AWS says small increases are usually auto-approved, larger ones go to Support and may be denied or partially approved; no authoritative statistics on rejection rates for new accounts exist, only scattered anecdotes (denials tied to capacity or account age, approvals within ~1 day).

### Cited Findings
- Default EC2 Spot quotas (vCPU-based, per region): All G and VT Spot Instance Requests = **0** (quota code L-3819A6DF); All P Spot Instance Requests = **0** (L-7212CCBC); All DL/F/Inf/Trn/X Spot = 0; All Standard (A,C,D,H,I,M,R,T,Z) Spot = 5. — (DOC) [Spot Instance quotas](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/using-spot-limits.html)
- Default On-Demand quotas: Running On-Demand G and VT instances = **0** (L-DB2E81BA); Running On-Demand P instances = **0** (L-417A185B); Standard = 5. EC2 "automatically increases your On-Demand/Spot quotas based on your usage". — (DOC) [EC2 instance type quotas](https://docs.aws.amazon.com/ec2/latest/instancetypes/ec2-instance-quotas.html)
- Quotas are counted in vCPUs; a pending Spot request counts against the quota until fulfilled/closed; if you terminate instances but don't cancel requests, the requests keep counting until EC2 detects the termination. Error for exceeding: `Max spot instance count exceeded`. — (DOC) [Spot Instance quotas](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/using-spot-limits.html)
- Process: "Smaller increases are usually automatically approved while larger requests are submitted to Support… Support can approve, deny, or partially approve your requests"; requests get no priority; if urgent or denied, contact Support. Request via console or `aws service-quotas request-service-quota-increase --service-code ... --quota-code ... --desired-value ...` (needs `ServiceQuotasFullAccess` or `iam:CreateServiceLinkedRole`). — (DOC) [Requesting a quota increase](https://docs.aws.amazon.com/servicequotas/latest/userguide/request-quota-increase.html)
- Spot and On-Demand are separate buckets; the launch error `VcpuLimitExceeded ... vCPU limit of 0 ... for the instance bucket` is commonly caused by raising the wrong one (user had 8 vCPU applied for "All G and VT Spot Instance Requests" but On-Demand quota was 0; another had 48 vCPU approved under Spot while launching On-Demand g5.xlarge). — (ANECDOTE, answered by re:Post "EXPERT"-badged users, ~2023) [re:Post: failed with 8 applied quota](https://repost.aws/questions/QUdeFs2VrcQ0W5VDZuq51p6A/create-g4dn-xlarge-failed-with-8-applied-quota-value-vcpu-limit-of-0), [re:Post: Increase Service](https://repost.aws/questions/QUUfTri9u5THW9VklU5_I6hg/increase-service)
- Quotas are region-specific; one user got G quota raised to 4 for g4dn.xlarge and still saw the error; replying AWS expert: "once my request was denied due to unavailability of g4dn in that region and other time it took almost a day to get my request approved"; a commenter noted `VcpuLimitExceeded` can hide an `InsufficientInstanceCapacity` issue. — (ANECDOTE, ~2023) [re:Post: g4dn.xlarge Spot vCPU error](https://repost.aws/questions/QUk8X2T0xcSJWzKJRiTeJB9Q/g4dn-xlarge-spot-instance-vcpu-error)
- SageMaker has its own training quotas that also default to **0** per region: "ml.g4dn.xlarge for training job usage = 0", "ml.g4dn.xlarge for spot training job usage = 0", same 0 for ml.g5.xlarge, ml.g6.xlarge, ml.g6e.xlarge, ml.g4dn.2xlarge (spot and normal). Also "Number of instances across all spot training jobs" = 4; "Longest run time for a training job" = 432,000 s (5 days). — (DOC, parsed by me from the page HTML on 2026-11-09) [SageMaker endpoints and quotas](https://docs.aws.amazon.com/general/latest/gr/sagemaker.html)

### Inferences
- Minimum ask for a g4dn.xlarge/g5.xlarge/g6.xlarge Spot run = **4 vCPU** under "All G and VT Spot Instance Requests" in the specific region (and 4 vCPU under "Running On-Demand G and VT instances" if you want an On-Demand fallback). Requesting a tiny number (4-8 vCPU) is consistent with the "smaller increases usually auto-approved" statement, but I found no source stating the auto-approval threshold for GPU quotas.
- Ask for both Spot and On-Demand G/VT quota in the chosen region up front, since a user hitting the wrong bucket loses days.
- SageMaker route adds a second, independent quota request (training-job and spot-training-job quotas per instance type), so it is strictly more friction for a new account than EC2.
- Because outcomes depend on capacity and account history (anecdotal), plan for possible denial/delay; choose the region deliberately because quotas don't transfer.

### Gaps
- No authoritative data on how long GPU quota requests take for new accounts or on rejection rates; only 3-year-old re:Post anecdotes ("almost a day", one denial for capacity).
- Search-engine summaries mentioned (a) AWS weighing account age/billing history, (b) one request "automatically denied about two hours later", (c) a >1-week escalation, from re:Post and yomotherboard.com Q&A pages; I did not open those pages and cannot attribute or verify them, so they are not stated as findings.
- 2025-2026 first-hand reports for new accounts were not found.
- Whether the AWS Free Tier/credits change quota outcomes was not researched.

---

## Q3. Launching and teardown: minimal one-shot commands, DLAMI vs containers, guaranteeing billing stops

### Takeaway
A one-shot job is a single `aws ec2 run-instances` with a Deep Learning Base AMI (found via SSM parameter), a Spot market option, `--instance-initiated-shutdown-behavior terminate`, and user-data that ends with `shutdown -h now`; with that setting plus root-volume DeleteOnTermination (default true) the instance and its root disk disappear when the script exits. The default shutdown behavior is `stop`, which would leave billing for EBS, so the flag is essential.

### Cited Findings
- Default: when you shut down from inside an EBS-backed instance (`shutdown`, `poweroff`) the instance **stops** by default; set `InstanceInitiatedShutdownBehavior=terminate` to terminate instead. It only applies to OS-initiated shutdown (not StopInstances API/console); the `halt` command does not shut down (CPU just halts and the instance keeps running/billing). — (DOC) [Change instance initiated shutdown behavior](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/Using_ChangingInstanceInitiatedShutdownBehavior.html)
- `run-instances --instance-initiated-shutdown-behavior` accepts `stop` or `terminate` (default `stop`). — (DOC) [run-instances reference](https://docs.aws.amazon.com/cli/latest/reference/ec2/run-instances.html)
- `--instance-market-options 'MarketType=spot,SpotOptions={SpotInstanceType=one-time|persistent,InstanceInterruptionBehavior=hibernate|stop|terminate}'`; "persistent Spot Instance requests are only supported when InstanceInterruptionBehavior is set to hibernate or stop"; when hibernation is not configured the interruption behavior is automatically `terminate`. `ValidUntil` is not supported for one-time requests. User data "must be base64-encoded" (CLI tools may do this for you). `--metadata-options HttpTokens=required` enforces IMDSv2. — (DOC) [run-instances reference](https://docs.aws.amazon.com/cli/latest/reference/ec2/run-instances.html)
- Root volume attached at launch: DeleteOnTermination default = **delete**; data volume attached at launch via CLI = delete (via console = preserve); volumes attached after launch = preserve; preserved volumes keep incurring charges; AMI's own setting provides the default and can be overridden in `--block-device-mappings` (device name e.g. `/dev/sda1` or `/dev/xvda`). Instance-store data is lost on termination. — (DOC) [Preserve data when an instance is terminated](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/preserving-volumes-on-termination.html)
- A stopped (not terminated) Spot instance still bills for its EBS volumes. — (DOC) [Billing for interrupted Spot Instances](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/billing-for-interrupted-spot-instances.html)
- Manual terminate: `aws ec2 terminate-instances --instance-ids i-...`; fails if termination protection is on. — (DOC) [Terminate EC2 instances](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/terminating-instances.html)
- DLAMI: SSM parameter pattern `/aws/service/deeplearning/ami/$architecture/$ami_type/latest/ami-id` (architecture x86_64 or arm64; ami_type = DLAMI name without "deep", "learning", "ami"), e.g. `aws ssm get-parameter --name /aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-ubuntu-22.04/latest/ami-id --region us-east-1 --query "Parameter.Value" --output text`; needs `ssm:GetParameter`. Alternative `aws ec2 describe-images --owners amazon --filters 'Name=name,Values=Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04) ????????'`. — (DOC) [Finding the ID of a DLAMI](https://docs.aws.amazon.com/dlami/latest/devguide/find-dlami-id.html)
- Currently listed DLAMIs (Nov 2026): Base GPU AMI on AL2023 / Ubuntu 22.04 / 24.04 / 26.04 (+ Ubuntu Pro), Base with Single CUDA; single-framework GPU PyTorch 2.14 (AL2023, Ubuntu 26.04), 2.13 (AL2023, Ubuntu 26.04), 2.12/2.11/2.10 (AL2023, Ubuntu 24.04). AWS warns PyTorch 2.10-2.12 DLAMIs contain unpatchable CVEs and recommends 2.13 or later. — (DOC) [DLAMI release notes](https://docs.aws.amazon.com/dlami/latest/devguide/appendix-ami-release-notes.html)
- Ultralytics `resume=True` on `last.pt` and checkpoint-per-epoch defaults — see Q4.

**Untested command template (my construction from the flags above; verify device name, subnet/SG defaults and `file://` handling before relying on it):**
```bash
AMI=$(aws ssm get-parameter --region us-east-1 \
  --name /aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-ubuntu-22.04/latest/ami-id \
  --query Parameter.Value --output text)
aws ec2 run-instances --region us-east-1 --image-id "$AMI" \
  --instance-type g4dn.xlarge --count 1 \
  --instance-market-options 'MarketType=spot,SpotOptions={SpotInstanceType=one-time,InstanceInterruptionBehavior=terminate}' \
  --instance-initiated-shutdown-behavior terminate \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":100,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
  --metadata-options HttpTokens=required \
  --user-data file://train.sh
# train.sh (sketch): fetch dataset from R2 -> train -> push results to R2 -> `shutdown -h now` (NOT `halt`)
```

### Inferences
- Guarantee pattern: wrap the whole of user-data in a shell function with `trap '...; shutdown -h now' EXIT` plus a hard time limit (e.g., a background `sleep N && shutdown -h now`), so a hung or crashed training still ends in OS shutdown, which becomes terminate because of the launch flag. Use `shutdown -h now`/`poweroff`, never `halt`.
- For a belt-and-braces check, list instances from a local script after the run (`aws ec2 describe-instances --filters Name=instance-state-name,Values=running`) — not a documented AWS recommendation, just my suggestion. Budget alarms/Billing alerts were not researched.
- A one-time Spot request with `terminate` interruption behavior avoids the persistent-request pitfall (persistent requests can re-launch/continue to hold quota); one-time requests are the natural fit for "throwaway" jobs.
- DLAMI is the simplest route: NVIDIA driver/CUDA preinstalled; then `pip install ultralytics` into the preinstalled PyTorch env (the Ultralytics install path itself was not researched). A container route (ECS/NVIDIA container toolkit on a plain Ubuntu AMI) adds setup and I did not research it.
- Keep R2 credentials out of plaintext user-data if possible (design concern; Secrets Manager/SSM Parameter Store with an instance profile was not researched).

### Gaps
- `file://` handling for `--user-data` and the 16 KB user-data limit were not confirmed in the fetched CLI excerpt (the page text was truncated at 100k characters).
- The exact AMI root device name/default root size for the DLAMI and the SSM path names of the PyTorch 2.13/2.14 DLAMIs on Ubuntu 26.04/AL2023 were not captured (release-note pages per AMI give the exact `ssm get-parameter` query).
- Container-image launch path (ECS-optimized GPU AMI, Ultralytics Docker image) not researched.
- No sourced evidence on cloud-init semantics in the DLAMI (user-data running as root at first boot) — widely known but not verified here.

---

## Q4. Interruption: 2-minute notice, interruption rates by GPU type, checkpoint/resume on EC2 and SageMaker (checkpoint_s3_uri, max wait)

### Takeaway
EC2 Spot gives a best-effort 2-minute warning via IMDS (`/latest/meta-data/spot/instance-action`, poll every 5 s) and EventBridge; YOLO (Ultralytics) saves `last.pt` every epoch and `resume=True` restores optimizer/scheduler/epoch, so a checkpoint-to-R2-per-epoch loop plus a relaunch wrapper is enough. Spot Advisor shows interruption frequency varies widely (g4dn mostly 5-20%, g5/g6/g6e/p3 often >20%). SageMaker Managed Spot automates checkpoint sync to S3 but only to S3.

### Cited Findings
**EC2 notices**
- A Spot interruption notice is a warning issued two minutes before EC2 stops or terminates the instance (for hibernate there is no 2-minute warning); AWS recommends checking for it every **5 seconds**; delivered both as an EventBridge event and via instance metadata; "emitted on a best effort basis". — (DOC) [Spot Instance interruption notices](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/spot-instance-termination-notices.html)
- EventBridge event: `detail-type: "EC2 Spot Instance Interruption Warning"`, `source: aws.ec2`, `detail: {instance-id, instance-action: hibernate|stop|terminate}`. — same DOC.
- IMDSv2 check: `TOKEN=$(curl -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600"); curl -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/spot/instance-action` returns e.g. `{"action": "terminate", "time": "2017-09-18T08:22:00Z"}`; HTTP 404 when no interruption is pending. `termination-time` is kept only for backward compatibility. — same DOC.

**Interruption rates (Spot Instance Advisor dataset, downloaded 2026-11-09)** — buckets <5%, 5-10%, 10-15%, 15-20%, >20%; dataset's `global_rate` = "<10%". The AWS Advisor page itself says only: "average interruption frequency across all Regions and Instance types has historically been <5%, the actual interruption rate for your workloads will depend on point-in-time available capacity" and does not define the measurement window. — (DATA) [Spot Advisor JSON](https://spot-bid-advisor.s3.amazonaws.com/spot-advisor-data.json), (DOC) [Spot Instance Advisor page](https://aws.amazon.com/ec2/spot/instance-advisor/)

| Type | us-east-1 | us-east-2 | us-west-2 | eu-west-1 | eu-central-1 | ap-northeast-1 | ap-southeast-1 | ap-south-1 | ap-northeast-2 |
|---|---|---|---|---|---|---|---|---|---|
| g4dn.xlarge | 10-15% | 5-10% | >20% | 5-10% | 15-20% | 10-15% | 15-20% | 15-20% | 10-15% |
| g4dn.2xlarge | >20% | 15-20% | >20% | <5% | >20% | >20% | >20% | >20% | >20% |
| g5.xlarge | >20% | >20% | >20% | 10-15% | 15-20% | >20% | n/a | >20% | 5-10% |
| g6.xlarge | >20% | 10-15% | >20% | n/a | <5% | >20% | n/a | >20% | >20% |
| g6e.xlarge | >20% | 15-20% | >20% | n/a | >20% | >20% | n/a | n/a | >20% |
| p3.2xlarge | >20% | >20% | >20% | >20% | >20% | >20% | >20% | n/a | >20% |

(g6/g6e/g5 rows have no entry in some regions, consistent with limited instance availability there; g7e.2xlarge: 15-20% us-east-1/us-east-2, >20% us-west-2, 5-10% ap-northeast-1.)

**Billing on interruption**: no charge if EC2 interrupts in the first hour (Linux); otherwise charged by the second. — (DOC) [Billing for interrupted Spot Instances](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/billing-for-interrupted-spot-instances.html)

**YOLO (Ultralytics) checkpoint/resume**
- `save=True` (default) saves checkpoints and final weights; checkpoints are saved at the end of every epoch by default; `save_period` (default -1) sets an N-epoch interval; you must finish at least one epoch before you can resume. Resume: `model = YOLO("path/to/last.pt"); model.train(resume=True)`; it restores weights, optimizer state, LR scheduler, epoch number and dataset. Omitting `resume` starts a new session. — (DOC) [Ultralytics Train mode](https://docs.ultralytics.com/modes/train/)

**SageMaker Managed Spot**
- Enable with `EnableManagedSpotTraining=True` and `StoppingCondition.MaxWaitTimeInSeconds`, which "must be larger than `MaxRuntimeInSeconds`". Checkpoints are copied from a local path to S3 and, on restart, from S3 back to the local path; "recommend you use checkpointing" unless the job finishes quickly; built-in/marketplace algorithms that do not checkpoint are limited to `MaxWaitTimeInSeconds` of 3600 s. — (DOC) [Managed Spot Training](https://docs.aws.amazon.com/sagemaker/latest/dg/model-managed-spot-training.html)
- No on-demand fallback: with `EnableManagedSpotTraining=True` the job runs only on Spot; if capacity is unavailable the job waits until `MaxWaitTimeInSeconds` and then stops. Wall-clock duration includes interruptions and waits and is not what is billed; only `BillableTimeInSeconds` is billed; `TrainingTimeInSeconds` excludes "intervals that were lost to a Spot interruption within their first hour of runtime" and re-provisioning/waiting time; savings % = `(1 - BillableTimeInSeconds/TrainingTimeInSeconds) * 100`. — same DOC.
- Checkpoint mechanics: default local dir `/opt/ml/checkpoints`, auto-synced to the S3 `CheckpointConfig.S3Uri`; existing S3 checkpoints are written into the container at job start; checkpoints added to S3 after start are not copied down; deleting locally also deletes in S3; SageMaker Python SDK has no high-level checkpoint-frequency setting; custom containers/scripts must save to and load from the local path themselves; for multi-instance jobs file names must be set manually to avoid overwrites; S3 destination can be a general-purpose or directory (Express One Zone) bucket. — (DOC) [Checkpoints in SageMaker AI](https://docs.aws.amazon.com/sagemaker/latest/dg/model-checkpoints.html)
- `CheckpointConfig` ("output location for managed spot training checkpoint data") has `S3Uri` and `LocalPath`, optional in the API. When a job is stopped SageMaker sends `SIGTERM` and delays termination 120 s so the algorithm can save artifacts. — (DOC) [CreateTrainingJob API](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateTrainingJob.html)

### Inferences
- EC2 handling pattern: (1) per-epoch `rclone copy runs/.../weights/last.pt r2:bucket/ckpt/` (or `save_period` for periodic copies), (2) a background watcher polling IMDS every 5 s that runs a final sync (<2 min) on notice, (3) a relaunch step (manual `run-instances` or a small wrapper) that pulls `last.pt` and runs `yolo train resume`. Because Ultralytics resume works from `last.pt` written every epoch, loss is bounded to <1 epoch even without the watcher.
- Interruption risk by type: g4dn.xlarge is the lowest-risk GPU in the main regions (5-15% in us-east-2, eu-west-1, us-east-1, ap-northeast-1), while g5/g6/g6e are frequently >20% — so the cheaper-and-safer option coincides. A 2-4 hour job at >20% monthly-style interruption frequency (window undefined by AWS page) is still more likely than not to survive, but resume logic is needed regardless. (Probability reasoning is mine; the Advisor's window is unsourced.)
- Because a Spot interruption in the first hour is free (Linux), early-failure cost is nil; later interruptions cost only seconds used.

### Gaps
- Window and methodology of Advisor "interruption frequency" not stated on the fetched AWS page (widely described as a trailing-30-day rate, but I could not source it).
- EventBridge "EC2 Instance Rebalance Recommendation" signal not researched/fetched.
- SageMaker behavior when an interruption happens before the first checkpoint is only described by third-party tutorials in search results (job can fail); not verified against official docs.
- How Ultralytics' own `resume` interacts with a changed working directory/`runs` path on a fresh instance was not tested.

---

## Q5. SageMaker Managed Spot Training: S3 mandatory? Can the container read directly from R2? Per-hour premium over EC2?

### Takeaway
SageMaker requires an S3 location for model output (`OutputDataConfig.S3OutputPath`) and syncs checkpoints only to S3, but input channels are optional and containers have outbound internet by default, so a script can pull data from R2 itself; you would still need a (tiny) AWS S3 bucket. SageMaker instance prices are ~40% above EC2 On-Demand for xlarge GPU sizes (~25% for 2xlarge), and new accounts have 0 default quota for each ml.* training/spot-training instance type.

### Cited Findings
- `OutputDataConfig` is **Required: Yes** ("S3 location where you want to store model artifacts"); `InputDataConfig` is **Required: No**; input sources are S3, EFS or FSx and "Your input must be in the same AWS region as your training job"; `CheckpointConfig` is optional but for managed spot; `AlgorithmSpecification` can point to your own `TrainingImage`; `EnableNetworkIsolation` default behavior: when true "No inbound or outbound network calls can be made". — (DOC) [CreateTrainingJob API](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateTrainingJob.html)
- "SageMaker AI training and deployed inference containers are internet-enabled by default. This allows containers to access external services and resources on the public internet." Network isolation (`EnableNetworkIsolation=True`) blocks all outbound calls (including S3 from the container; SageMaker does the S3 transfers itself); using `VpcConfig` lets you restrict internet access from the VPC at the cost of extra network configuration. — (DOC) [Run Training and Inference Containers in Internet-Free Mode](https://docs.aws.amazon.com/sagemaker/latest/dg/mkt-algo-model-internet-free.html)
- Checkpoint sync destination is an S3 URI (general purpose or directory bucket). — (DOC) [Checkpoints in SageMaker AI](https://docs.aws.amazon.com/sagemaker/latest/dg/model-checkpoints.html)
- SageMaker managed spot price is "a discount off the Amazon SageMaker AI on-demand price" and is "not comparable to the Amazon EC2 Spot price history … [which] does not include the Amazon SageMaker AI managed-service rate"; "up to 90%" cost optimization claimed; (API page says "up to 80%"). — (DOC) [Managed Spot Training](https://docs.aws.amazon.com/sagemaker/latest/dg/model-managed-spot-training.html), [CreateTrainingJob API](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateTrainingJob.html)
- SageMaker Training On-Demand prices, us-east-1 (price list published 2026-11-07), vs EC2 On-Demand same region:

| Instance | SageMaker training $/h | EC2 OD $/h | Ratio |
|---|---|---|---|
| ml.g4dn.xlarge / g4dn.xlarge | 0.736 | 0.526 | 1.40 |
| ml.g5.xlarge / g5.xlarge | 1.408 | 1.006 | 1.40 |
| ml.g6.xlarge / g6.xlarge | 1.127 | 0.8048 | 1.40 |
| ml.g6e.xlarge / g6e.xlarge | 2.61 | 1.861 | 1.40 |
| ml.g4dn.2xlarge / g4dn.2xlarge | 0.94 | 0.752 | 1.25 |
| ml.g5.2xlarge / g5.2xlarge | 1.515 | 1.212 | 1.25 |
| ml.g6.2xlarge / g6.2xlarge | 1.222 | 0.9776 | 1.25 |
| ml.g7e.2xlarge / g7e.2xlarge | 4.2039 | 3.36312 | 1.25 |

 — (DATA) [SageMaker price list](https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonSageMaker/current/us-east-1/index.json), [EC2 price list](https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonEC2/current/us-east-1/index.json); EC2 2xlarge prices from the Vantage dataset: g4dn.2xlarge 0.752, g5.2xlarge 1.212, g6.2xlarge 0.9776 (3P) [instances.json](https://instances.vantage.sh/instances.json). Ratios are my arithmetic.
- The SageMaker price list has "Spot Training" SKUs (`USE1-TrSpt:...`) only for some types (e.g., ml.g6.*, ml.g6e.*, ml.p5.48xlarge, r5, t3) and their listed price equals the On-Demand training price; I found none for ml.g4dn/ml.g5 even though per-type "spot training job usage" quotas exist for them. — (DATA) [SageMaker price list](https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonSageMaker/current/us-east-1/index.json)
- Quota defaults 0 for ml.g4dn.xlarge / g5.xlarge / g6.xlarge / g6e.xlarge "for spot training job usage" and "for training job usage" in each region. — (DOC) [SageMaker quotas](https://docs.aws.amazon.com/general/latest/gr/sagemaker.html)
- Trained model artifacts always land in S3 first ("After training completes, SageMaker saves the resulting model artifacts to an Amazon S3 location that you specify"). — (DOC) [CreateTrainingJob API](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateTrainingJob.html)

### Inferences
- No conflict in principle with R2: skip input channels, set an env var with R2 creds, and have the training script (own container or script-mode with your code) download from R2 via the S3 API/rclone over the default internet path; push final weights to R2 at the end. But (a) a small S3 bucket is still mandatory for output + checkpoint sync, (b) `EnableNetworkIsolation` must stay false and any `VpcConfig` must have internet egress (NAT/IGW), (c) I found no source or report of anyone doing this with R2 specifically; it rests on "internet-enabled by default" and R2's S3 compatibility.
- Cost: EC2 Spot (g4dn.xlarge ≈ $0.25/h) vs SageMaker spot at an unknown discount on $0.736: even at a 60% discount SageMaker ≈ $0.29/h; premium vs EC2 Spot is roughly the 25-40% markup but the real discount isn't published, so check `BillableTimeInSeconds` after a trial run.
- Managed spot gives checkpoint sync/resume/wait handling "for free" but adds IAM role + S3 bucket + SDK/container work and a second quota request; for a single user comfortable with a shell script, EC2 is simpler and cheaper.

### Gaps
- Actual realized Spot discount for ml.g4dn.xlarge/ml.g5.xlarge training not published in sources I found (only "up to 90%/80%").
- ML storage volume pricing for SageMaker training not researched.
- No official statement or user report confirming a SageMaker training container reading directly from R2.
- Whether SageMaker Spot is available for the newest ml.g7/g7e types not verified.

---

## Q6. AWS Batch on Spot with GPU: worth the setup for this use case?

### Takeaway
Batch itself is free and supports GPU families, Spot, and automatic job retries (1-10 attempts) when an instance is terminated, but it does no checkpointing and still needs the same EC2 Spot quota plus compute environment, job queue, job definition and IAM roles. For a single user running occasional one-off YOLO jobs it is more setup than a one-shot `run-instances`; it pays off only for queues of many jobs or recurring pipelines.

### Cited Findings
- "There is no additional charge for AWS Batch"; you pay for the underlying resources you create; Spot discounts "applied at billing time". — (DOC) [AWS Batch pricing](https://aws.amazon.com/batch/pricing/)
- GPU jobs: `resourceRequirements` in the job definition pins a number of GPUs to the container (not available to other jobs on that instance); all instance types in a compute environment running GPU jobs "must be from the p3, p4, p5, p6, g3, g3s, g4, g5, g6, gr6, g7, or g7e instance families. If this isn't done a GPU job might get stuck in the RUNNABLE status"; only NVIDIA GPUs are supported for GPU scheduling; supported list includes g4dn, g5, g6, g6e, g7, g7e, p3. — (DOC) [Run GPU jobs](https://docs.aws.amazon.com/batch/latest/userguide/gpu-jobs.html)
- Retries: 1-10 attempts per job; failure scenarios include non-zero exit code, "Amazon EC2 instance failure or termination" and internal AWS errors; up to 6 `evaluateOnExit` entries; `AWS_BATCH_JOB_ATTEMPT` env var gives the attempt number; cancelled/terminated jobs are not retried. — (DOC) [Automated job retries](https://docs.aws.amazon.com/batch/latest/userguide/job_retries.html)
- Spot allocation strategies available are SPOT_CAPACITY_OPTIMIZED and SPOT_PRICE_CAPACITY_OPTIMIZED; AWS sources recommend the latter for most cases and recommend offering as many instance types/families as possible. — (DOC, from search-result snippets of) [Batch allocation strategies](https://docs.amazonaws.cn/en_us/batch/latest/userguide/allocation-strategies.html), [AWS Compute Blog](https://aws.amazon.com/blogs/compute/optimizing-for-cost-availability-and-throughput-by-selecting-your-aws-batch-allocation-strategy)
- A third-party guide shows a Spot GPU compute environment with `allocationStrategy: SPOT_CAPACITY_OPTIMIZED`, GPU families and `sharedMemorySize: 8192`. — (3P) [oneuptime: Configure AWS Batch for GPU workloads](https://oneuptime.com/blog/post/2026-02-12-configure-aws-batch-for-gpu-workloads/view)

### Inferences
- Batch launches EC2 Spot instances in your account, so the default-0 "All G and VT Spot Instance Requests" quota (Q2) still applies; Batch removes none of the new-account friction.
- Setup objects: compute environment, job queue, job definition (container image with CUDA + Ultralytics), service role + instance profile/ECS instance role, ECR or public image; plus you must supply checkpoint/resume logic yourself (Batch retries just rerun the container; pairing retries with R2 checkpoints and `resume=True` works). Teardown semantics (scale-to-zero when `minvCpus=0`) were not verified in my sources.
- **Three-way comparison for a solo user/small team (my synthesis of the findings in Q1-Q5):**

| | EC2 Spot (run-instances + user-data) | SageMaker Managed Spot | AWS Batch on Spot |
|---|---|---|---|
| Setup effort | Lowest: 1 CLI command + script; default VPC; quota request (4 vCPU Spot G/VT) | Medium: IAM role, S3 bucket, SDK/container, per-instance-type quota request | Highest: CE + queue + job def + roles |
| Price (g4dn.xlarge, us-east-1) | ~$0.25 Spot + ~$0.016 EBS/IPv4 | ~$0.736 OD list; discounted, rate unpublished; ~40% markup at xlarge | = EC2 Spot (no Batch fee) |
| Quotas | EC2 Spot G/VT (default 0) | SageMaker ml.* spot-training quota (default 0) | EC2 Spot G/VT (default 0) |
| Interruption handling | DIY: IMDS/EventBridge + R2 checkpoints + relaunch | Automatic checkpoint sync to S3 + auto-resume within MaxWait | Retry strategy (1-10), checkpointing DIY |
| Data in R2 | Direct (rclone/boto3/mount); free ingress | Container can fetch via internet (default), but S3 still required for output/checkpoints | Direct, as EC2 |
| Teardown | OS shutdown + terminate flag | Managed | Managed (CE scale-down not verified) |

### Gaps
- Batch scale-to-zero/teardown behavior and Spot interruption event details specific to Batch (status reasons like `Host EC2 (instance i-...) terminated`) not verified from official docs in this research.
- No cost data for Batch on top of EC2 beyond "no additional charge".

---

## Q7. FUSE mounting on EC2: rclone mount / s3fs against R2, and mountpoint-s3 with non-AWS endpoints

### Takeaway
On a full EC2 VM, FUSE works; rclone has a first-class "Cloudflare" S3 provider and Cloudflare documents rclone and a FUSE-adapter example (tigrisfs, read-only) for R2. AWS's mountpoint-s3 officially does not support non-AWS S3-like services (best-effort `--endpoint-url`; maintainers closed an S3-compatible-providers PR in Mar 2026 and open issues show breakage with some providers) and I found no R2-specific mountpoint-s3 report. For YOLO training, copying the dataset to local disk is simpler and more robust than mounting.

### Cited Findings
**rclone**
- rclone S3 backend lists "Cloudflare R2 Storage" as provider value `Cloudflare`, with region `auto`. — (DOC) [rclone s3 docs](https://rclone.org/s3/) (config examples not read because page text was truncated)
- Cloudflare's recommended rclone config: `type=s3`, `provider=Cloudflare`, access key/secret, `endpoint=https://<accountid>.r2.cloudflarestorage.com`, `acl=private`, `no_check_bucket=true` (only for object-level-permission tokens); use rclone ≥ v1.59 (earlier may return HTTP 401); multipart uploads cost more Class A operations (≥3× PutObject). — (DOC) [Cloudflare R2 + rclone](https://developers.cloudflare.com/r2/examples/rclone/)
- `rclone mount` VFS cache modes: `off` (default; reads/writes direct, some ops impossible), `minimal`, `writes`, `full` (all reads/writes buffered to disk, sparse cache files). `--vfs-cache-max-size` is checked only every poll interval so the cache can exceed it; `--vfs-read-ahead` buffers on disk in `full` mode; `--buffer-size` is per open file in memory; for S3-type remotes `--no-modtime` saves a transaction per file; suggested starting point for high-performance object stores `--vfs-read-chunk-streams 16 --vfs-read-chunk-size 4M`; needs `fusermount`/`fusermount3` on PATH; on newer Ubuntu AppArmor may block (`aa-disable /usr/bin/fusermount3`). The page has no ML-training guidance. — (DOC) [rclone mount](https://rclone.org/commands/rclone_mount/)
- A user mounting R2 as root and reading as another user got permission errors and "hang" in foreground; fix: `--allow-other` (mount runs in foreground unless `--daemon`). rclone v1.60.0, Oct 2022. — (ANECDOTE) [rclone forum: mount R2 hangs](https://forum.rclone.org/t/rclone-mount-s3-cloudflare-r2-hangs/33696)
- Search snippets show users running rclone mounts on R2 for Plex/Nextcloud and discussing mount flags to reduce R2 API (Class A/B) costs; no benchmark. — (ANECDOTE/3P) [rclone forum: Nextcloud R2 mount flags](https://forum.rclone.org/t/how-to-optimize-mount-flags-to-reduce-r2-s3-api-costs-when-using-for-nextcloud-data-directory/34670), [rclone forum: R2 Plex scrubbing](https://forum.rclone.org/t/cloudflare-r2-plex-video-scrubbing-performance/32335)

**Cloudflare's own FUSE example**
- Cloudflare Containers example mounts R2 read-only with `tigrisfs --endpoint "https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com" --region auto -o ro -f "$R2_BUCKET_NAME" /mnt/r2`, and states FUSE "does not provide local-disk performance or full POSIX filesystem semantics" but suits reading datasets/models. — (DOC) [Mount R2 buckets with FUSE](https://developers.cloudflare.com/containers/examples/r2-fuse-mount/)
- Cloudflare Sandbox docs describe s3fs-based R2 mounts (needs FUSE 3, key pair, endpoint `https://<ACCOUNT_ID>.r2.cloudflarestorage.com`, region `auto`) — (DOC, seen only via search-result summary) [Cloudflare Sandbox: mount an R2 bucket](https://developers.cloudflare.com/sandbox/files/mount-an-r2-bucket/). I did not find a standalone s3fs-fuse + R2 how-to or issue thread.
- R2 S3 API: supports GetObject (Range), HeadObject, ListObjects/ListObjectsV2, multipart, CopyObject; does not support object tagging or ACLs; region is `auto` (empty or `us-east-1` alias to `auto`); checksums CRC64NVME (full object) and CRC32/CRC32C/SHA1/SHA256 (composite) supported; the page does not mention path-style vs virtual-hosted or `x-amz-optional-object-attributes`. — (DOC) [R2 S3 API compatibility](https://developers.cloudflare.com/r2/api/s3/api/)

**mountpoint-s3 (awslabs)**
- README: "Mountpoint for Amazon S3 is designed for high-performance access to the Amazon S3 service. While it may be functional against other storage services that use S3-like APIs, we aren't able to provide support for those use cases, and they may inadvertently break when we make changes…"; contributions of minor compat fixes accepted if testable against Amazon S3. — (DOC) [mountpoint-s3 README, "Compatibility with other storage services"](https://github.com/awslabs/mountpoint-s3#compatibility-with-other-storage-services)
- Maintainer (2023): Mountpoint "probably works today against other S3-like services — you can point the `--endpoint-url` config anywhere you want", but is not officially tested/supported; newer S3 features such as additional checksums may break or need workarounds; a user reported it working with NetApp StorageGRID. Request closed May 2023 as not prioritized. — (DOC/ANECDOTE) [Issue #144](https://github.com/awslabs/mountpoint-s3/issues/144)
- Aug 2025: maintainer reply to a RustFS/MinIO request: not supported, try `--endpoint-url`. — (DOC) [Issue #1553](https://github.com/awslabs/mountpoint-s3/issues/1553)
- Oct 2025: TLS negotiation failure against Supabase S3 fixed by adding `--force-path-style` (diagnosed via `--debug-crt`); maintainer: "While we don't support third-party S3 services…". — (DOC/ANECDOTE) [Issue #1661](https://github.com/awslabs/mountpoint-s3/issues/1661)
- PR #1749 "Add support for S3-compatible providers" (Jan 2026) was closed unmerged on 2026-03-04; maintainers: "we won't be able to accept an enumeration of other third-party object storage", asked for the concrete failing invocation. — (DOC) [PR #1749](https://github.com/awslabs/mountpoint-s3/pull/1749)
- Provider-specific breakage reports: OVH needed `--upload-checksums=off` and crashed with a checksum fatal error (issue #1154, Nov 2024, v1.10.0); a third-party provider's CompleteMultipartUpload response without Content-Length broke writes (issue #1741, open); **Mountpoint 1.24.0 always sends `x-amz-optional-object-attributes: RestoreStatus` on ListObjectsV2 and Ceph RGW resets the connection, mount fails (issue #2025, open, filed 2026-11-05)**. — (DOC/ANECDOTE) [#1154](https://github.com/awslabs/mountpoint-s3/issues/1154), [#1741](https://github.com/awslabs/mountpoint-s3/issues/1741), [#2025](https://github.com/awslabs/mountpoint-s3/issues/2025)
- My GitHub issue searches in awslabs/mountpoint-s3 for "R2", "Cloudflare" and "r2.cloudflarestorage.com" returned no R2-specific issue or PR (only unrelated CRT-update PRs mentioning Cloudflare). — (DATA) GitHub search API run on 2026-11-09.
- Modal's docs say its CloudBucketMount mounts S3 buckets using AWS Mountpoint and supports R2 by passing the `https://<account-id>.r2.cloudflarestorage.com` endpoint (detected by the hostname) — (3P/vendor, via search-result snippet) [Modal CloudBucketMount](https://modal.com/docs/sdk/py/latest/CloudBucketMount.md). This suggests mount-s3 works against R2 in at least one production setting, but it is a vendor statement, not AWS's.
- Mountpoint limitations seen in issues: no rename ("rename failed: operation not supported by Mountpoint"; emulation request #1174 open). — (ANECDOTE/DOC) [#1154](https://github.com/awslabs/mountpoint-s3/issues/1154)

### Inferences
- Robust, documented-by-Cloudflare path for the use case: `rclone copy r2:bucket/dataset /mnt/nvme/dataset` (or `rclone mount --vfs-cache-mode full` read-only if the dataset is too large for disk) rather than mountpoint-s3/s3fs; training re-reads images each epoch, so local disk (instance-store NVMe 125-250 GB on g4dn/g5, or gp3) avoids per-read R2 requests and latency.
- If mounting is wanted, mount-s3 with `--endpoint-url https://<acct>.r2.cloudflarestorage.com --region auto --force-path-style` (plus possibly `--upload-checksums=off` for writes) is plausible but unsupported by AWS and unverified for R2; the open ListObjectsV2 header issue (#2025) is a concrete risk because R2's docs don't state tolerance for that header. Test with `mount-s3 --debug --debug-crt` before relying on it.
- R2 mounts generate many small GetObject/List operations (Class B/A); the free tier (10M Class B/month) usually covers a small dataset, but a mount with repeated listing can add up (see Nextcloud cost thread).

### Gaps
- I did not find a verified, reproducible `s3fs-fuse` command against R2 or any measured throughput for rclone mount/mountpoint-s3 against R2 from EC2.
- The R2 section of rclone's S3 page (config details/caveats) was not read due to page truncation.
- Whether R2 accepts `x-amz-optional-object-attributes: RestoreStatus` is unknown.
- Cloudflare docs fetched don't say which R2 operations are billed as Class A vs Class B (commonly GET/HEAD = B; PUT/LIST = A, but not verified here).

---

## Q8. Network path: R2 to EC2 throughput and cost

### Takeaway
Cost is effectively zero for the transfer itself: R2 egress is free and EC2 ingress from the internet is $0.00/GB; you pay only R2 operation counts beyond the free tier. I found no reliable measured R2→EC2 download throughput, so benchmark it from the chosen region.

### Cited Findings
- R2 egress to Internet: free (S3 API, Workers API, r2.dev); free tier 10M Class B and 1M Class A ops/month on Standard storage; Class B $0.36/M, Class A $4.50/M; "If you connect other metered services to a bucket, those services may charge you". — (DOC) [R2 pricing](https://developers.cloudflare.com/r2/pricing/)
- AWS: data transfer in from the internet $0.000 per GB; out to the internet $0.09/GB (first 10 TB), 100 GB/month free. — (DATA) [AWSDataTransfer price list](https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AWSDataTransfer/current/us-east-1/index.json), (DOC) [EC2 pricing](https://aws.amazon.com/ec2/pricing/on-demand/)
- Instance NIC: g4dn.xlarge "up to 25 Gbps", g5/g6.xlarge "up to 10 Gbps", g6e.xlarge "up to 20 Gbps" (burstable "up to" figures). — (DOC) [AWS Batch GPU table](https://docs.aws.amazon.com/batch/latest/userguide/gpu-jobs.html)
- R2 endpoint is account-scoped `https://<ACCOUNT_ID>.r2.cloudflarestorage.com`; rclone docs describe `auto` region as "R2 buckets are automatically distributed across Cloudflare's data centers for low latency". — (DOC) [R2 S3 API](https://developers.cloudflare.com/r2/api/s3/api/), [rclone s3](https://rclone.org/s3/)
- Weak throughput evidence: a RcloneView tutorial reports ~114 MB/s moving S3→R2 from rclone on EC2 with defaults, ~4× with tuned chunk size/concurrency (upload direction, vendor, unverified); a DEV.to post reports 10.5-12.8 Gbps for 10 GB-object R2 *uploads* (unverified methodology); Backblaze Q1 2026 report (US-East multithreaded downloads): R2 led at 100 MiB objects; in EU-Central R2 trailed other providers (client not EC2). — (3P) [RcloneView tutorial](https://rcloneview.com/support/tutorials/tags/aws-s-3), [DEV.to benchmark](https://dev.to/johalputt/benchmark-cloudflare-r2-vs-aws-s3-2026-for-large-object-upload-throughput-12a2), [StorageReview on Backblaze Q1 2026](https://www.storagereview.com/news/backblaze-publishes-q1-2026-cloud-storage-performance-results)

### Inferences
- A 20 GB image dataset costs $0 to move (ingress free, R2 egress free) and a few thousand-to-million Class B ops (mostly inside the free tier); the only real transfer cost is uploading results back to R2 if they exceed the 100 GB/month AWS free-out allowance ($0.09/GB beyond; YOLO weights are small).
- Throughput will be governed by R2 distance to the chosen AWS region and parallelism (`rclone --transfers`, multi-thread streams), not by instance NIC for xlarge GPU sizes; prefer a region geographically near R2's serving location, and measure with a 1-minute `rclone copy` before launching long jobs.
- Prefer a public subnet (default VPC) so traffic doesn't traverse a NAT gateway (NAT pricing not researched).

### Gaps
- No credible measured R2→EC2 download throughput (rclone/boto3) by region; no data on R2 per-bucket request-rate limits; no source on how R2 chooses serving locations relative to AWS regions.

---

## Q9. Access from mainland China: AWS global regions vs AWS China (Beijing/Ningxia, separate accounts, ICP) — which is practical

### Takeaway
AWS China (Beijing operated by Sinnet, Ningxia by NWCD) is a fully separate account system whose sign-up, per official FAQ, is for mainland Chinese legal entities with business-license verification, so an individual/small foreign-style setup is not practical there; ICP filing matters only if you host a website. For a mainland-based user the realistic route is a global-region account, but I found no authoritative information on sign-up from mainland China or on network quality from the mainland to global regions.

### Cited Findings
- China-region accounts are separate from global accounts ("a set of credentials that are distinct and separate from other Amazon Web Services global Accounts"); Beijing region is operated by Beijing Sinnet Technology (Sinnet), Ningxia by Ningxia Western Cloud Data Technology (NWCD); ICP Recordal is required for non-commercial internet information services (websites) hosted in those regions; each operator supports ICP filing for its customers. — (DOC) [AWS China overview](https://www.amazonaws.cn/en/about-aws/china/)
- Sign-up FAQ: account "is only opened for registration by a Chinese (mainland) legal entity at this stage"; business registration certificate (Business License / unified social credit code) required, plus a named information-security administrator (PRC ID card, passport, or HK/Macao/Taiwan permits) and phone verification; the page doesn't mention individuals or payment methods and doesn't require ICP at registration. — (DOC) [AWS China sign-up FAQ](https://www.amazonaws.cn/en/about-aws/china/faqs/signup/)
- GPU availability in China regions (older announcements): G4 (T4) instances available in Beijing (Sinnet) and Ningxia (NWCD) as On-Demand, Reserved or Spot (2020); G5 (A10G) instances available in both China regions (announcement dated 2024-04-26); P3 launched in Beijing (2017). — (DOC) [G4 in China (2020)](https://www.amazonaws.cn/en/new/2020/introducing-amazon-ec2-g4-is-now-available-in-aws-china-beijing-region-operated-by-sinnet/), [G5 in China (2024)](https://amazonaws.cn/en/new/2024/amazon-ec2-g5-instances-are-available-in-amazon-web-services-china-regions); a third-party listing dated 2026-03-30 shows g4dn sizes priced in Ningxia — (3P) [instances.cloudchipr.com g4dn.4xlarge](https://instances.cloudchipr.com/aws/ec2/g4dn.4xlarge)
- A community deployment guide notes the default g5 type isn't supported in China regions and must be changed manually (g4dn). — (3P, single community project, may be dated) [stable-diffusion-on-eks AWS CN guide](https://aws-samples.github.io/stable-diffusion-on-eks/en/implementation-guide/deployment/aws-cn)
- An AWS China blog on multi-region GPU scheduling says single regions such as Singapore may lack some GPU types (e.g., G6/G6e, G5) or hit capacity failures, recommending multi-region fallback. — (DOC, AWS China blog, via search snippet) [AWS China blog](https://aws.amazon.com/cn/blogs/china/agentic-scheduler-with-strands-agentcore-for-multi-region-gpu-inference). Consistent with the Spot Advisor dataset lacking g5/g6/g6e entries in ap-southeast-1 (DATA).
- AWS China blog snippets state AWS Client VPN can't be used for cross-border access and cross-border connectivity for enterprises goes via qualified carriers/AWS Cross-Border Connection over Direct Connect. — (DOC, via search snippet of a blog index page; I could not identify the exact article) [AWS China networking blog index](https://aws.amazon.com/cn/blogs/china/category/networking-content-delivery/page/15)
- Global-region candidates for a mainland user with datasets in R2: Tokyo (ap-northeast-1) g4dn.xlarge Spot ≈ $0.34 (10-15%), Seoul (ap-northeast-2) ≈ $0.33, Mumbai ≈ $0.26 (15-20%), Singapore ≈ $0.31 (15-20%), us-east-1 ≈ $0.25. — (3P/DATA) Q1 table.

### Inferences
- Within AWS China: ICP is irrelevant for a pure batch-training workload (no website), but the mainland-legal-entity + business-license sign-up barrier makes it impractical for an individual/small foreign-style team; also R2 sits outside the China regions, so cross-border data pulling from R2 would be needed (no source on how it performs).
- Practical route for a mainland-based person: global AWS account in a nearby non-mainland region (Tokyo/Seoul/Singapore/Hong Kong or US); pick by g4dn availability (all of Tokyo, Seoul, Singapore, Mumbai list g4dn.xlarge Spot) and the R2→EC2 test, and keep the job fully on the AWS side so that only the control connection (SSH/CLI, minimal data) crosses the border; the heavy R2→EC2 transfer does not touch the mainland.
- Global-account sign-up requirements for mainland residents (card/phone/address) and mainland latency/stability to AWS regions are unknown from my sources.

### Gaps
- No authoritative information on registering a global AWS account from mainland China, or on connectivity/latency from the mainland to ap-northeast-1/ap-southeast-1/etc.
- No current (late-2026) official confirmation of g4dn/g5 Spot availability or prices in AWS China regions; no information on whether Spot quotas/GPU capacity exist for small customers there.
- No source on R2 reachability or speed from AWS China regions.
- Hong Kong (ap-east-1) appears in the Vantage dataset with unusually low g4dn.xlarge Spot ($0.162, no min/max) but I did not verify the data or opt-in requirements, so I did not recommend it.
