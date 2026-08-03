# D3 clean selected-source Phase C verification

- Result: PASS, 290/290 cases, zero skips.
- Base revision: `f92a3ab8860445af1f6abeb31f996024aef1b64f`.
- Temporary clean candidate revision: `932d6b11b66a2739e4dc484df14a6e7731966f74`.
- Exact selected patch: `selected.patch`, SHA-256
  `14c2a6888efad7fdc1a8fd98c6a34bd7fc5597ebb56914832eed58bf50533bc8`.
- Cleanup from measured no-LUT: `selected-cleanup.patch`, SHA-256
  `067e74b6b790bd98f98202e73b29a4c23409abf0e29f339a79f571f4e787c36b`.
- Candidate source fingerprint:
  `6140dfc67761875a734560608d7b55554591287af32647a9a7d330f39a992098`.
- Git gate inputs: clean (`gate_input_status=[]`).
- Report SHA-256:
  `0b4b1e00815e7b8b7bc7548bc3c6b896c8a8bbeb1a3a38627e77e3dabc04f7fe`.
- Summary SHA-256:
  `83b227056b6b6edcb436345db4c5e77e09d100bac84c107dcde1ec106e74b951`.
- Full run log SHA-256:
  `a751e3ff712ab072cbeecfc96283114594370ea6c4a0a7bdd9e58e88294f0f5b`.
- Built wheel SHA-256:
  `854b5f9bf5639cf954f2653d510833f81c5be1db8eb66089737ad2ac0d7fef69`.
- Installed native extension SHA-256:
  `94422cf4e8792773777ae615089ae6593f5a2113dc97ab2622297770d9e0016a`.
- Installed package artifact SHA-256:
  `f947311a8f7b462cf26d26395f1d09445955fb5fc5d2897648754b966bfc4a16`.
- Separate local release module SHA-256:
  `378d4b5eb1dcfdc0f9da6e2732977c750f929923e071db6573880ecd757e9a9a`.
- Release build log SHA-256:
  `7e3c2e2214e8a059b83509080e9e43f5993460875867c698d8cb669856fc9b95`.

The only source delta from the measured no-LUT patch deletes the unreachable
lookup helper, its test import, and its unit test. The complete
`execute_image_patchify_plan_into` source is byte-identical in measured no-LUT
and selected source; both extracts have SHA-256
`74794d63d401d9e6deaeef0c3cea7a6ca80d6fbedcdda4ec86e9efd6abb92a84`.
The selected build contains no dormant-lookup warning.
The local release module does not byte-match the measured no-LUT local release
module after deleting unreachable source. No binary-equivalence claim is made;
source-level production-path equality plus the selected source's own build and
full conformance evidence are the retained proof.
