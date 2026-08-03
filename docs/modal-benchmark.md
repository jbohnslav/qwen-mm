# Minimal Modal benchmark runner

`scripts/modal_benchmark.py` proves that qwen-mm can build and run its paired
benchmark orchestration inside one ephemeral native Linux x86 Modal container.
It is deliberately not a service: `modal run` creates the app for this command
and Modal stops it automatically when the command exits.

The image starts from the amd64 digest
`python@sha256:28255a3ace7eb4c48bc1b57b90af29e1bc82b4fd6c60614a8e3dce61b87ff941`
(`python:3.11.15-slim-bookworm`), installs Rust 1.97.1 and uv 0.11.29, copies
the source into an image layer, and syncs the locked workspace. The function
requests four physical CPU cores and 8 GiB of memory. It then:

1. rejects non-Linux, non-x86, and explicitly emulated CPU environments;
2. verifies the remote source fingerprint against the local packaged tree;
3. builds a wheel, installs it into a clean venv, runs installed-wheel smoke,
   and verifies the native module is an x86-64 ELF;
4. runs `make benchmark-v2-self-test` and the benchmark result validator; and
5. returns a hash-verified ZIP with the wheel, raw result, report, logs,
   and host/build/source provenance.

No Modal token or other secret is added to the image or artifact.

## Run

The Modal CLI must already be configured locally. From the repository root:

```console
make modal-benchmark-test
modal run scripts/modal_benchmark.py --output /tmp/qwen-mm-modal-d0.zip
```

The local entrypoint validates the complete returned archive before atomically
writing the requested output. Any image build, source mismatch, architecture,
wheel, smoke, benchmark, schema, digest, or download error makes the command
fail without replacing an existing artifact.

## Evidence boundary and D1 handoff

This runner produces native Linux x86 **diagnostic** evidence. Modal guarantees
the requested CPU resources, but this artifact does not claim exclusive host
ownership and cannot by itself satisfy D4's dedicated-host certification.

D1 should reuse the same `benchmark_image`, `run_d0` resource/provenance
pattern, and integrity-checked return path after `qwen_mm.benchmark:create_adapter`
exists. Replace the self-test command with the candidate-only profile capture
and/or paired smoke command, retain the native-x86 checks and source binding,
and give the follow-on artifact a distinct versioned schema rather than
relabeling this self-test result as performance evidence.
