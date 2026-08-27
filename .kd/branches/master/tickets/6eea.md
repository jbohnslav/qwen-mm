---
id: "6eea"
status: open
deps: []
links: []
created: 2026-08-27T12:50:02Z
type: task
priority: 2
parent: 7544
---
# Load direct image paths, URLs, and data URIs

Accept the image source forms used directly by official Qwen examples.

## Acceptance Criteria

- [ ] Qwen-style image content accepts `pathlib` paths, plain paths, file URLs,
      HTTP(S) URLs, and image data URIs without a separate images list.
- [ ] Qwen3.5 OpenAI-style `image_url` content accepts both the string and
      nested URL forms used by the official examples.
- [ ] Supplying an HTTP(S) URL directly means qwen-mm fetches it with no
      permission flag, private-network filtering, security-policy ceremony, or
      warning beyond ordinary actionable network failures.
- [ ] Existing bytes, RGB arrays, separate images lists, numeric references, repeated references, and per-occurrence options remain compatible.
- [ ] Reads fail actionably and respect existing encoded-item and encoded-batch resource ceilings.
- [ ] Installed-wheel tests cover local paths, file URLs, HTTP redirects and errors, data URIs, multi-image ordering, and OpenAI image_url normalization.
