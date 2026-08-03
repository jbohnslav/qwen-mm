# D3 measured no-LUT Phase C verification

- Result: PASS, 290/290 cases, zero skips.
- Base revision: `f92a3ab8860445af1f6abeb31f996024aef1b64f`.
- Temporary clean candidate revision: `44c5cffd8b8263c7d6488784a32aac9817e23be0`.
- Exact measured patch: `no-lut.patch`, SHA-256
  `cb3f6496bee26a026cdf6478c48f6b34052639ea25b8d8170023f377afbc94fc`.
- Candidate source fingerprint:
  `5fcc18554382cc284cef5981664d650d86b2d0f0323b5f1117b00799020bc84b`.
- Git gate inputs: clean (`gate_input_status=[]`).
- Report SHA-256:
  `a67b42c70bae4c0a2237cdfb3b1495597ef6fd05ab30a2da45e03b25aa31cbae`.
- Summary SHA-256:
  `83b227056b6b6edcb436345db4c5e77e09d100bac84c107dcde1ec106e74b951`.
- Full run log SHA-256:
  `e6b4ea10de09de33f096e238905ae5c496b16110fb30497644b0a770eee95cb4`.
- Initial sandbox-DNS attempt log SHA-256:
  `feeece75fc2e7f84cce78b409da6577caf940b03c68e13aa0aa462ed38e53ddf`.
- Built wheel SHA-256:
  `ce02af3c594bbd1fc86d660799dd9148e7e3a0ce9cff4b496adb3affc25a2e2a`.
- Installed native extension SHA-256:
  `acae2fff3c903d3edecb67b3c6951bef75fd4cb1dff58b9c25c006c6f3065c71`.
- Installed package artifact SHA-256:
  `f947311a8f7b462cf26d26395f1d09445955fb5fc5d2897648754b966bfc4a16`.
- Separate local release module SHA-256:
  `6cc7562a3bca5fa06a59bfefe31f5f8adab530ac8c00441133f4030254e1f601`.
- Release build log SHA-256:
  `d549ac74e37d8608076248b907ddfbd44064c67ebe9a23fd39540b67ffb82658`.

The first fresh-cache invocation failed only because the sandbox could not
resolve PyPI. The identical clean commit reran with network access, completed
the authoritative installed-wheel matrix, and passed the report validator.
The build's dead-code warning is expected: the measured no-LUT patch preserved
an unreachable lookup helper and its test but never called the helper from the
production writer.
