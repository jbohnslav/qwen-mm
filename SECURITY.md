# Security

Please report vulnerabilities privately through
https://github.com/jbohnslav/qwen-mm/security/advisories/new.
Include the package version, supported model profile, and a minimal reproducer.
Do not put access tokens, private images, or private model inputs in public issues.

The current 0.1 release line is supported. Processor assets are hash-pinned;
model weights are not bundled. Explicit image URLs are fetched as requested by
the caller, so applications should apply their own URL policy at their boundary.
