# Security Policy

dgpu handles cloud platform API keys and storage credentials, and it creates machines that bill by the hour. In this project, a bug that leaves an instance running or exposes a key counts as a security issue.

## Reporting a vulnerability

**Don't open a public issue.** Report it privately through GitHub:

**[Report a vulnerability](https://github.com/baabanana/dynamic_gpu/security/advisories/new)** (Security tab → "Report a vulnerability")

Please include:

- the dgpu version or commit (`dgpu --version`, `git rev-parse HEAD`)
- the backend (Vast.ai or AutoDL) and whether it happened on a real platform or in tests
- steps to reproduce, or the code path, and what an attacker or a bad outcome gains (a key, a running instance, data)

**Never include real credentials, account IDs or bucket endpoints** in a report. Replace them with placeholders.

This is a volunteer-maintained project. We aim to acknowledge reports within 7 days and to discuss a fix and disclosure timeline with you. We'll credit you in the advisory unless you prefer otherwise.

## Supported versions

dgpu is pre-1.0. Only the latest code on `dev` and the latest release (once there is one) receive security fixes.

## What we consider a vulnerability

- **Credential exposure:** a Vast API key or AutoDL token reaches an instance, storage, a log, an exception message, a startup command or an environment variable. Also: storage credentials become visible in a platform console or API listing.
- **Billing leak:** a path where an instance that dgpu created stays running after the job ended, failed, timed out, was cancelled or crashed, and neither the runner, the watchdog, `submit` nor `dgpu gc` destroys it.
- **Touching other people's instances:** `gc` or any other command destroys or changes an instance that doesn't carry a dgpu label, or misreads a foreign label as its own.
- **Privilege-drop bypass:** training code running as the unprivileged user can read the storage credentials or the Vast container key, or signal or kill the runner or watchdog.
- **Integrity:** the bundled rclone binary is used without matching its pinned SHA-256, or a malicious object in the bucket or a malicious `job.yaml` can write outside the intended directories (path traversal in `pull`, data sync or packaging).
- **Silent data loss** that the design says must be recorded but isn't.

## Known limitations (not vulnerabilities)

These are documented trade-offs. See [design §9](docs/dgpu-design.md#9-后续风险).

- **Hosts can read the instance's disk.** On community machines, the host can see your training data and the storage token. Scope the token to a single bucket, set `datacenter_only = true` for Vast, or don't put sensitive data on rented machines.
- **`run_as_root: true`** deliberately gives training code root, including access to the storage token.
- **dgpu can't drop privileges** when the container doesn't run as root. `dgpu status` and the log say so.
- **An instance whose container never starts** (`python3` missing, image can't be pulled) can only be destroyed by `submit` or `gc`. If neither runs, it bills until your balance runs out. Keep balances small and auto-recharge off.

## If you leaked a credential

Rotate it immediately, then check for unexpected instances:

- **Vast.ai:** delete the key under Account → API Keys and create a new one.
- **AutoDL:** reset the developer token in the console settings.
- **Cloudflare R2:** revoke the API token under R2 → Manage API Tokens and create a new one, scoped to one bucket.
- Run `dgpu gc --dry-run` and check each platform's console for instances you don't recognise.
