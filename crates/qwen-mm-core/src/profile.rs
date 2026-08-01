//! Immutable, offline compatibility profiles.

use std::{collections::BTreeMap, str::FromStr};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

use crate::error::{ErrorCategory, QwenError, Result};

const CONTRACT_ID: &str = "qwen-mm-compat-v1";
const BUNDLED_MANIFEST: &str = include_str!("../../../reference/compatibility/v1.json");
const QWEN3_VL_ALIAS: &str = "qwen3-vl-8b";
const QWEN35_ALIAS: &str = "qwen3.5-9b";
const QWEN3_VL_FINGERPRINT: &str =
    "9e2e515f166fdad60e68528aadd7ef2a410724b74855f1b90dfbe1eadcaa7ae1";
const QWEN35_FINGERPRINT: &str = "4f870d0c41812a7f5fe318d9ab7db64a805bac569c9fa4570e957752fd69edb5";

/// The two aliases accepted by compatibility contract v1.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ProfileAlias {
    /// `Qwen/Qwen3-VL-8B-Instruct` at the frozen revision.
    Qwen3Vl8b,
    /// `Qwen/Qwen3.5-9B` at the frozen revision.
    Qwen35_9b,
}

impl ProfileAlias {
    /// Returns the exact public alias.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Qwen3Vl8b => QWEN3_VL_ALIAS,
            Self::Qwen35_9b => QWEN35_ALIAS,
        }
    }

    const fn expected_fingerprint(self) -> &'static str {
        match self {
            Self::Qwen3Vl8b => QWEN3_VL_FINGERPRINT,
            Self::Qwen35_9b => QWEN35_FINGERPRINT,
        }
    }
}

impl FromStr for ProfileAlias {
    type Err = QwenError;

    fn from_str(alias: &str) -> Result<Self> {
        match alias {
            QWEN3_VL_ALIAS => Ok(Self::Qwen3Vl8b),
            QWEN35_ALIAS => Ok(Self::Qwen35_9b),
            _ => Err(profile_error("unknown profile alias").with_context("alias", alias)),
        }
    }
}

/// Resolved upstream class identities frozen into a profile.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProcessorClasses {
    /// Processor class.
    pub processor: String,
    /// Tokenizer class.
    pub tokenizer: String,
    /// Image processor class.
    pub image_processor: String,
    /// Image implementation backend.
    pub image_backend: String,
    /// Video processor class.
    pub video_processor: String,
    /// Video implementation backend.
    pub video_backend: String,
}

/// Frozen image/video preprocessing constants.
#[derive(Clone, Debug, PartialEq)]
pub struct VisualProfile {
    /// Spatial patch size.
    pub patch_size: u64,
    /// Temporal patch size.
    pub temporal_patch_size: u64,
    /// Spatial merge size.
    pub merge_size: u64,
    /// Flattened normalized patch width.
    pub patch_width: u64,
    /// RGB normalization means.
    pub image_mean: [f64; 3],
    /// RGB normalization standard deviations.
    pub image_std: [f64; 3],
    /// Image processor configuration minimum pixels (not the composed default).
    pub image_config_min_pixels: u64,
    /// Image processor configuration maximum pixels (not the composed default).
    pub image_config_max_pixels: u64,
    /// Effective composed-path image minimum pixels.
    pub composed_image_min_pixels: u64,
    /// Effective composed-path image maximum pixels.
    pub composed_image_max_pixels: u64,
    /// Video processor minimum pixels.
    pub video_config_min_pixels: u64,
    /// Video processor maximum pixels.
    pub video_config_max_pixels: u64,
}

/// Frozen tokenizer behavior and special-token identities.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TokenizerProfile {
    /// Padding side; v1 requires `right`.
    pub padding_side: String,
    /// Truncation side recorded upstream; public v1 never truncates.
    pub truncation_side: String,
    /// Repository tokenizer model maximum.
    pub model_max_length: u64,
    /// Optional beginning-of-sequence token.
    pub bos_token_id: Option<i64>,
    /// End-of-sequence token.
    pub eos_token_id: i64,
    /// Right-padding token.
    pub pad_token_id: i64,
    /// Image placeholder token.
    pub image_token_id: i64,
    /// Video placeholder token.
    pub video_token_id: i64,
    /// Vision-start token.
    pub vision_start_token_id: i64,
    /// Vision-end token.
    pub vision_end_token_id: i64,
}

/// One fully validated immutable model profile.
#[derive(Clone, Debug, PartialEq)]
pub struct Profile {
    /// Accepted local alias.
    pub alias: ProfileAlias,
    /// Canonical contract fingerprint.
    pub fingerprint: String,
    /// Hugging Face repository identifier.
    pub model_id: String,
    /// Immutable repository revision.
    pub revision: String,
    /// Resolved class and backend identities.
    pub classes: ProcessorClasses,
    /// Preprocessing constants.
    pub visual: VisualProfile,
    /// Tokenizer identity and constants.
    pub tokenizer: TokenizerProfile,
    /// Required repository artifact SHA-256 values by relative name.
    pub artifacts: BTreeMap<String, String>,
}

/// A validated local manifest containing exactly the two v1 profiles.
#[derive(Clone, Debug)]
pub struct ProfileRegistry {
    contract_id: String,
    python: String,
    lock: LockRecord,
    packages: BTreeMap<String, String>,
    source_files: BTreeMap<String, String>,
    environment: BTreeMap<String, Value>,
    oracle_kwargs: Value,
    profiles: BTreeMap<ProfileAlias, Profile>,
}

impl ProfileRegistry {
    /// Loads and validates the manifest compiled into `qwen-mm-core`.
    ///
    /// This performs no filesystem or network access.
    ///
    /// # Errors
    ///
    /// Returns `profile_mismatch` if the compiled asset is malformed or has
    /// drifted from either accepted immutable profile.
    pub fn bundled() -> Result<Self> {
        Self::from_manifest_str(BUNDLED_MANIFEST)
    }

    /// Parses a candidate manifest under the immutable v1 policy.
    ///
    /// This entry point exists for packagers and drift tests. Any parse,
    /// schema, canonical-fingerprint, or accepted-fingerprint failure is
    /// returned as [`ErrorCategory::ProfileMismatch`].
    ///
    /// # Errors
    ///
    /// Returns `profile_mismatch` for every rejected candidate manifest.
    pub fn from_manifest_str(json: &str) -> Result<Self> {
        let raw: Value = serde_json::from_str(json).map_err(|error| {
            profile_error("malformed compatibility manifest")
                .with_context("detail", error.to_string())
        })?;
        let manifest: Manifest = serde_json::from_value(raw.clone()).map_err(|error| {
            profile_error("malformed compatibility manifest")
                .with_context("detail", error.to_string())
        })?;

        if manifest.schema_version != 1 {
            return Err(profile_error("unsupported manifest schema")
                .with_context("schema_version", u64::from(manifest.schema_version)));
        }
        if manifest.contract_id != CONTRACT_ID {
            return Err(profile_error("unexpected compatibility contract")
                .with_context("contract_id", manifest.contract_id));
        }
        if manifest.profiles.len() != 2 {
            return Err(profile_error("manifest must contain exactly two profiles")
                .with_context("actual", manifest.profiles.len()));
        }

        let raw_object = raw.as_object().ok_or_else(|| {
            profile_error("malformed compatibility manifest")
                .with_context("detail", "root must be an object")
        })?;
        let raw_profiles = required_object(raw_object, "profiles")?;
        let mut profiles = BTreeMap::new();

        for (alias_text, manifest_profile) in manifest.profiles {
            let alias = alias_text.parse::<ProfileAlias>()?;
            let raw_profile = raw_profiles.get(&alias_text).ok_or_else(|| {
                profile_error("profile disappeared during manifest parsing")
                    .with_context("alias", alias_text.clone())
            })?;
            validate_fingerprint(
                raw_object,
                raw_profile,
                alias,
                &manifest_profile.fingerprint,
            )?;
            let profile = Profile::from_manifest(alias, manifest_profile);
            validate_profile_semantics(&profile)?;
            if profiles.insert(alias, profile).is_some() {
                return Err(
                    profile_error("duplicate profile alias").with_context("alias", alias_text)
                );
            }
        }

        for expected in [ProfileAlias::Qwen3Vl8b, ProfileAlias::Qwen35_9b] {
            if !profiles.contains_key(&expected) {
                return Err(profile_error("required profile is absent")
                    .with_context("alias", expected.as_str()));
            }
        }

        Ok(Self {
            contract_id: manifest.contract_id,
            python: manifest.python,
            lock: manifest.lock,
            packages: manifest.packages,
            source_files: manifest.source_files,
            environment: manifest.environment,
            oracle_kwargs: manifest.oracle_kwargs,
            profiles,
        })
    }

    /// Returns the frozen compatibility contract identifier.
    #[must_use]
    pub fn contract_id(&self) -> &str {
        &self.contract_id
    }

    /// Returns the frozen Python version.
    #[must_use]
    pub fn python_version(&self) -> &str {
        &self.python
    }

    /// Returns the frozen dependency-lock path and SHA-256.
    #[must_use]
    pub fn lock(&self) -> (&str, &str) {
        (&self.lock.path, &self.lock.sha256)
    }

    /// Returns an exact package version, if recorded.
    #[must_use]
    pub fn package_version(&self, package: &str) -> Option<&str> {
        self.packages.get(package).map(String::as_str)
    }

    /// Returns an installed-source SHA-256, if recorded.
    #[must_use]
    pub fn source_hash(&self, path: &str) -> Option<&str> {
        self.source_files.get(path).map(String::as_str)
    }

    /// Returns the frozen environment entry, including explicit JSON nulls.
    #[must_use]
    pub fn environment_value(&self, name: &str) -> Option<&Value> {
        self.environment.get(name)
    }

    /// Returns the exact frozen oracle keyword object.
    #[must_use]
    pub const fn oracle_kwargs(&self) -> &Value {
        &self.oracle_kwargs
    }

    /// Resolves a validated profile by typed alias.
    #[must_use]
    pub fn get(&self, alias: ProfileAlias) -> &Profile {
        // Construction proves both keys are present.
        &self.profiles[&alias]
    }

    /// Resolves a validated profile by exact string alias.
    ///
    /// # Errors
    ///
    /// Returns `profile_mismatch` when `alias` is not one of the two frozen
    /// aliases.
    pub fn resolve(&self, alias: &str) -> Result<&Profile> {
        Ok(self.get(alias.parse()?))
    }
}

impl Profile {
    fn from_manifest(alias: ProfileAlias, profile: ManifestProfile) -> Self {
        Self {
            alias,
            fingerprint: profile.fingerprint,
            model_id: profile.model_id,
            revision: profile.revision,
            classes: ProcessorClasses {
                processor: profile.classes.processor,
                tokenizer: profile.classes.tokenizer,
                image_processor: profile.classes.image_processor,
                image_backend: profile.classes.image_backend,
                video_processor: profile.classes.video_processor,
                video_backend: profile.classes.video_backend,
            },
            visual: VisualProfile {
                patch_size: profile.visual.patch_size,
                temporal_patch_size: profile.visual.temporal_patch_size,
                merge_size: profile.visual.merge_size,
                patch_width: profile.visual.patch_width,
                image_mean: profile.visual.image_mean,
                image_std: profile.visual.image_std,
                image_config_min_pixels: profile.visual.image_config_min_pixels,
                image_config_max_pixels: profile.visual.image_config_max_pixels,
                composed_image_min_pixels: profile.visual.composed_image_min_pixels,
                composed_image_max_pixels: profile.visual.composed_image_max_pixels,
                video_config_min_pixels: profile.visual.video_config_min_pixels,
                video_config_max_pixels: profile.visual.video_config_max_pixels,
            },
            tokenizer: TokenizerProfile {
                padding_side: profile.tokenizer.padding_side,
                truncation_side: profile.tokenizer.truncation_side,
                model_max_length: profile.tokenizer.model_max_length,
                bos_token_id: profile.tokenizer.bos_token_id,
                eos_token_id: profile.tokenizer.eos_token_id,
                pad_token_id: profile.tokenizer.pad_token_id,
                image_token_id: profile.tokenizer.image_token_id,
                video_token_id: profile.tokenizer.video_token_id,
                vision_start_token_id: profile.tokenizer.vision_start_token_id,
                vision_end_token_id: profile.tokenizer.vision_end_token_id,
            },
            artifacts: profile.artifacts,
        }
    }

    /// Returns an artifact SHA-256 by repository-relative name.
    #[must_use]
    pub fn artifact_hash(&self, name: &str) -> Option<&str> {
        self.artifacts.get(name).map(String::as_str)
    }

    /// Reports whether this profile supports Qwen3.5 thinking semantics.
    #[must_use]
    pub const fn supports_thinking(&self) -> bool {
        matches!(self.alias, ProfileAlias::Qwen35_9b)
    }
}

fn validate_fingerprint(
    root: &Map<String, Value>,
    raw_profile: &Value,
    alias: ProfileAlias,
    claimed: &str,
) -> Result<()> {
    let computed = compute_profile_fingerprint(root, raw_profile)?;
    if computed != claimed {
        return Err(profile_error("stale profile fingerprint")
            .with_context("alias", alias.as_str())
            .with_context("claimed", claimed)
            .with_context("computed", computed));
    }
    if claimed != alias.expected_fingerprint() {
        return Err(
            profile_error("profile is not an accepted immutable revision")
                .with_context("alias", alias.as_str())
                .with_context("fingerprint", claimed),
        );
    }
    Ok(())
}

fn compute_profile_fingerprint(root: &Map<String, Value>, raw_profile: &Value) -> Result<String> {
    let mut profile = raw_profile.clone();
    profile
        .as_object_mut()
        .ok_or_else(|| profile_error("profile must be an object"))?
        .remove("fingerprint")
        .ok_or_else(|| profile_error("profile fingerprint is absent"))?;

    let mut canonical = Map::new();
    for key in [
        "contract_id",
        "python",
        "lock",
        "packages",
        "source_files",
        "environment",
        "oracle_kwargs",
    ] {
        canonical.insert(
            key.to_owned(),
            root.get(key).cloned().ok_or_else(|| {
                profile_error("manifest fingerprint input is absent").with_context("field", key)
            })?,
        );
    }
    canonical.insert("profile".to_owned(), profile);
    let bytes = serde_json::to_vec(&Value::Object(canonical)).map_err(|error| {
        profile_error("could not canonicalize profile").with_context("detail", error.to_string())
    })?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

fn validate_profile_semantics(profile: &Profile) -> Result<()> {
    let visual = &profile.visual;
    let common_visuals_match = visual.patch_size == 16
        && visual.temporal_patch_size == 2
        && visual.merge_size == 2
        && visual.patch_width == 1536
        && visual
            .image_mean
            .iter()
            .all(|value| value.to_bits() == 0.5_f64.to_bits())
        && visual
            .image_std
            .iter()
            .all(|value| value.to_bits() == 0.5_f64.to_bits())
        && visual.composed_image_min_pixels == 4096
        && visual.composed_image_max_pixels == 16_777_216;
    let classes_match = profile.classes.processor == "Qwen3VLProcessor"
        && profile.classes.tokenizer == "Qwen2Tokenizer"
        && profile.classes.image_processor == "Qwen2VLImageProcessor"
        && profile.classes.image_backend == "torchvision"
        && profile.classes.video_processor == "Qwen3VLVideoProcessor"
        && profile.classes.video_backend == "torchvision";
    if !common_visuals_match || !classes_match || profile.tokenizer.padding_side != "right" {
        return Err(
            profile_error("behavior-changing profile constants do not match v1")
                .with_context("alias", profile.alias.as_str()),
        );
    }
    Ok(())
}

fn required_object<'a>(root: &'a Map<String, Value>, key: &str) -> Result<&'a Map<String, Value>> {
    root.get(key)
        .and_then(Value::as_object)
        .ok_or_else(|| profile_error("manifest field must be an object").with_context("field", key))
}

fn profile_error(message: impl Into<String>) -> QwenError {
    QwenError::new(ErrorCategory::ProfileMismatch, message)
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    schema_version: u32,
    contract_id: String,
    #[allow(dead_code)]
    created_at: String,
    python: String,
    lock: LockRecord,
    packages: BTreeMap<String, String>,
    source_files: BTreeMap<String, String>,
    environment: BTreeMap<String, Value>,
    oracle_kwargs: Value,
    profiles: BTreeMap<String, ManifestProfile>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct LockRecord {
    path: String,
    sha256: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ManifestProfile {
    fingerprint: String,
    model_id: String,
    revision: String,
    classes: ManifestClasses,
    visual: ManifestVisual,
    tokenizer: ManifestTokenizer,
    artifacts: BTreeMap<String, String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ManifestClasses {
    processor: String,
    tokenizer: String,
    image_processor: String,
    image_backend: String,
    video_processor: String,
    video_backend: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ManifestVisual {
    patch_size: u64,
    temporal_patch_size: u64,
    merge_size: u64,
    patch_width: u64,
    image_mean: [f64; 3],
    image_std: [f64; 3],
    image_config_min_pixels: u64,
    image_config_max_pixels: u64,
    composed_image_min_pixels: u64,
    composed_image_max_pixels: u64,
    video_config_min_pixels: u64,
    video_config_max_pixels: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ManifestTokenizer {
    padding_side: String,
    truncation_side: String,
    model_max_length: u64,
    bos_token_id: Option<i64>,
    eos_token_id: i64,
    pad_token_id: i64,
    image_token_id: i64,
    video_token_id: i64,
    vision_start_token_id: i64,
    vision_end_token_id: i64,
}

#[cfg(test)]
mod tests {
    use serde_json::Value;

    use super::{
        BUNDLED_MANIFEST, ProfileAlias, ProfileRegistry, QWEN3_VL_FINGERPRINT, QWEN35_FINGERPRINT,
    };
    use crate::error::ErrorCategory;

    #[test]
    fn bundled_profiles_resolve_all_frozen_identity() {
        let registry = ProfileRegistry::bundled().expect("bundled manifest must validate");
        assert_eq!(registry.contract_id(), "qwen-mm-compat-v1");
        assert_eq!(registry.python_version(), "3.11.15");
        assert_eq!(
            registry.lock(),
            (
                "reference/uv.lock",
                "30d82d0dfa563005a9ef997f90372d4029e5d1d26c0e20a3f7a03bc0d026acab"
            )
        );
        assert_eq!(registry.package_version("transformers"), Some("5.14.1"));
        assert_eq!(
            registry.source_hash("qwen_vl_utils/vision_process.py"),
            Some("f3710ced8ffe735da57d08cf5236cf9d326ff646ed1bdad2e62593a7801adc33")
        );

        let vl = registry.get(ProfileAlias::Qwen3Vl8b);
        assert_eq!(vl.fingerprint, QWEN3_VL_FINGERPRINT);
        assert_eq!(vl.model_id, "Qwen/Qwen3-VL-8B-Instruct");
        assert_eq!(vl.revision, "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b");
        assert_eq!(vl.visual.patch_width, 1536);
        assert_eq!(vl.tokenizer.pad_token_id, 151_643);
        assert_eq!(
            vl.artifact_hash("chat_template.json"),
            Some("5c72a170d2a4a1a3bc5adad2e689ae28138a9700e5b8c96c0266331e86c0acce")
        );

        let qwen35 = registry.resolve("qwen3.5-9b").expect("known alias");
        assert_eq!(qwen35.fingerprint, QWEN35_FINGERPRINT);
        assert_eq!(qwen35.revision, "c202236235762e1c871ad0ccb60c8ee5ba337b9a");
        assert_eq!(qwen35.tokenizer.eos_token_id, 248_046);
        assert!(qwen35.supports_thinking());
        assert_eq!(
            qwen35.artifact_hash("chat_template.jinja"),
            Some("a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715")
        );
    }

    #[test]
    fn every_manifest_failure_is_profile_mismatch() {
        let malformed = "{";
        let unknown = BUNDLED_MANIFEST.replace("qwen3-vl-8b", "qwen3-vl-7b");
        let stale = BUNDLED_MANIFEST.replace("\"patch_size\": 16", "\"patch_size\": 14");
        let changed_revision = BUNDLED_MANIFEST.replace(
            "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
            "1c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
        );
        let changed_asset = BUNDLED_MANIFEST.replacen(
            "5c72a170d2a4a1a3bc5adad2e689ae28138a9700e5b8c96c0266331e86c0acce",
            "6c72a170d2a4a1a3bc5adad2e689ae28138a9700e5b8c96c0266331e86c0acce",
            1,
        );
        for candidate in [
            malformed,
            &unknown,
            &stale,
            &changed_revision,
            &changed_asset,
        ] {
            let error =
                ProfileRegistry::from_manifest_str(candidate).expect_err("must reject drift");
            assert_eq!(error.category(), ErrorCategory::ProfileMismatch);
        }
    }

    #[test]
    fn recomputed_but_unaccepted_drift_is_rejected() {
        let mut manifest: Value = serde_json::from_str(BUNDLED_MANIFEST).expect("valid JSON");
        manifest["profiles"]["qwen3-vl-8b"]["revision"] = Value::String("changed".to_owned());
        let recomputed = super::compute_profile_fingerprint(
            manifest.as_object().expect("root object"),
            &manifest["profiles"]["qwen3-vl-8b"],
        )
        .expect("canonical fingerprint");
        manifest["profiles"]["qwen3-vl-8b"]["fingerprint"] = Value::String(recomputed);
        let error = ProfileRegistry::from_manifest_str(&manifest.to_string()).expect_err("drift");
        assert_eq!(error.category(), ErrorCategory::ProfileMismatch);
    }

    #[test]
    fn unknown_resolution_is_profile_mismatch() {
        let registry = ProfileRegistry::bundled().expect("bundled manifest");
        let error = registry.resolve("repo-head").expect_err("unknown alias");
        assert_eq!(error.category(), ErrorCategory::ProfileMismatch);
        assert_eq!(error.context()["alias"].to_string(), "repo-head");
    }
}
