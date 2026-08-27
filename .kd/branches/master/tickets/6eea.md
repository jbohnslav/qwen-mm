---
id: "6eea"
status: closed
deps: []
links: []
created: 2026-08-27T12:50:02Z
type: task
priority: 2
closed_at: 2026-08-27T13:15:31Z
resolution: completed
closed_context: codex:ad56e22ce3deade5
parent: 7544
---
# Load direct image paths, URLs, and data URIs

Accept the image source forms used directly by official Qwen examples.

## Acceptance Criteria

- [x] Qwen-style image content accepts `pathlib` paths, plain paths, file URLs,
      HTTP(S) URLs, and image data URIs without a separate images list.
- [x] Qwen3.5 OpenAI-style `image_url` content accepts both the string and
      nested URL forms used by the official examples.
- [x] Supplying an HTTP(S) URL directly means qwen-mm fetches it with no
      permission flag, private-network filtering, security-policy ceremony, or
      warning beyond ordinary actionable network failures.
- [x] Existing bytes, RGB arrays, separate images lists, numeric references, repeated references, and per-occurrence options remain compatible.
- [x] Reads fail actionably and respect existing encoded-item and encoded-batch resource ceilings.
- [x] Installed-wheel tests cover local paths, file URLs, HTTP redirects and errors, data URIs, multi-image ordering, and OpenAI image_url normalization.

## Worklog

- [2026-08-27 08:57] [codex:84dfad21] — Started direct media-source normalization implementation in a native subagent; public behavior will fetch caller-supplied HTTP(S) URLs directly without security-policy flags, private-network filtering, or warnings.
- [2026-08-27 09:18] [codex:84dfad21] — Added the public Python `Processor` facade and `_media.py` normalization boundary. Qwen `image` values now accept path-like objects, plain/file/HTTP(S) paths, image data URIs, bytes, encoded dictionaries, and RGB arrays directly; OpenAI `image_url` string and nested `url` forms normalize into the same ordered native request schema. Explicit caller-supplied HTTP(S) URLs are fetched directly with ordinary redirects and a 30-second reliability timeout—there is no permission flag, address filtering, policy mode, or warning ceremony.
- [2026-08-27 09:18] [codex:84dfad21] — Direct reads stream against both `encoded_bytes_per_item` and the cumulative `encoded_bytes_per_batch` budget across requests and existing/direct image lists. File size and HTTP `Content-Length` are checked before streaming when available; ordinary base64 data URIs preflight decoded size before materialization. Path, HTTP status/network, malformed-data, and resource-limit failures include an actionable source and typed context.
- [2026-08-27 09:18] [codex:84dfad21] — Added `crates/qwen-mm-python/tests/media_sources.py` to the installed-wheel gate. It covers `Path`, plain path, `file://`, data URI, direct localhost HTTP redirect (demonstrating no private-address filter), HTTP 404, streamed limits, item/batch limits, multi-image order, OpenAI forms, direct bytes/RGB, separate lists, numeric and repeated references, and per-occurrence options. `make python-binding-test` passed all installed-wheel suites: pretrained construction, documentation, media sources, and usability/binding.
- [2026-08-27 09:21] [codex:84dfad21] — Full `make check` passed after the shared facade, construction, padding/output, and media-source changes: Ruff lint/format, Rust formatting and clippy with warnings denied, 97 core/native tests plus focused example tests (expected asset-dependent ignores), doc tests, offline core build, fixture verification, and reference unit tests all completed successfully.
- [2026-08-27 09:15] [codex:ad56e22c] — Root review accepted direct-media normalization. Explicit path, file URL, HTTP(S), data URI, and OpenAI image_url sources normalize before the deterministic native boundary; caller-supplied URLs fetch directly with no opt-in/filter/policy/warning, while timeout and cumulative item/batch limits remain reliability controls. Root fresh-wheel make python-binding-test and repository-wide make check both passed.

## Lifecycle

- 2026-08-27T13:15:31Z [codex:ad56e22ce3deade5] — closed (completed)
