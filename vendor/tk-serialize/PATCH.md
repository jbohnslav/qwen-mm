# Qwen3.5 regex fallback for tokenizers 1.0.0-rc.2

Source: crates.io `tk-serialize` 0.1.0-rc.2, Apache-2.0.

`src/from_json/pre_tokenizers.rs::read_split` always used `Split::native`,
which leaves `Search::Unavailable` for regexes outside bitcannon's recognized
grammars. Qwen3.5 uses a pattern including Unicode marks (`\p{M}`), which
is not recognized in this release. Encoding therefore fails even with the
regex feature enabled.

The local patch retains recognized native grammars and constructs unrecognized
splits with `Split::new` when `fancy-regex` is enabled. The root workspace
explicitly enables this feature on the reader. Model assets, patterns,
vocabulary IDs, and merge ranks are unchanged. Remove this patch when an
upstream release initializes the fallback correctly and passes our Qwen tests.

Regression coverage: the core text tests with `--include-ignored`, including
Qwen3.5 committed token IDs, image/video spans, and decomposed Unicode. The
paired benchmark harness also compares each result against the official adapter.
