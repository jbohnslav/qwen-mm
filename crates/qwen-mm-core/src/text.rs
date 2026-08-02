//! Offline chat rendering, multimodal placeholder expansion, and tokenization.

use std::{fmt::Write as _, fs, path::Path};

use serde::{
    Deserialize, Deserializer,
    de::{MapAccess, SeqAccess, Visitor},
};
use sha2::{Digest, Sha256};
use tokenizers::Tokenizer;

use crate::{
    error::{ErrorCategory, QwenError, Result},
    limits::{
        ProfiledRequest, ResourceLimits, checked_capacity_bytes, checked_mul, preflight_batch,
    },
    output::{CoordinateRange, Matrix},
    profile::{Profile, ProfileAlias, ProfileRegistry},
    request::{ContentItem, Message, MessageContent, Request, Role},
};

const IMAGE_TOKEN: &str = "<|image_pad|>";
const VIDEO_TOKEN: &str = "<|video_pad|>";
const VISION_START_TOKEN: &str = "<|vision_start|>";
const VISION_END_TOKEN: &str = "<|vision_end|>";

/// The modality represented by one rendered placeholder.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VisualModality {
    /// Still image.
    Image,
    /// Raw-frame video.
    Video,
}

/// Deterministic post-visual-processing information for one media occurrence.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum VisualExpansion<'a> {
    /// One image grid in `[temporal, height, width]` patch units.
    Image {
        /// Referenced index in [`Request::images`].
        input_index: usize,
        /// Exact `image_grid_thw` row.
        grid_thw: [u64; 3],
    },
    /// One video grid plus one timestamp per temporal grid position.
    Video {
        /// Referenced index in [`Request::videos`].
        input_index: usize,
        /// Exact `video_grid_thw` row.
        grid_thw: [u64; 3],
        /// Exact frame-group timestamps, in seconds.
        timestamps: &'a [f64],
    },
}

impl VisualExpansion<'_> {
    const fn modality(self) -> VisualModality {
        match self {
            Self::Image { .. } => VisualModality::Image,
            Self::Video { .. } => VisualModality::Video,
        }
    }

    const fn input_index(self) -> usize {
        match self {
            Self::Image { input_index, .. } | Self::Video { input_index, .. } => input_index,
        }
    }
}

/// A request paired with its ordered, already-computed visual expansion plan.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlannedTextRequest<'a> {
    /// Structured request to render.
    pub request: Request<'a>,
    /// One entry per media occurrence in message/content traversal order.
    pub visuals: &'a [VisualExpansion<'a>],
}

/// Exact coordinates for one visual placeholder replacement.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TextReplacement {
    /// Image or video placeholder.
    pub modality: VisualModality,
    /// Placeholder coordinates in the rendered prompt, in Unicode code points.
    pub rendered_code_points: CoordinateRange,
    /// Replacement coordinates in the expanded prompt, in Unicode code points.
    pub expanded_code_points: CoordinateRange,
    /// Replacement coordinates in the expanded token sequence.
    pub expanded_tokens: CoordinateRange,
}

/// Exact rendered and visual-expanded text for one request.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedTextRequest {
    /// Output of the pinned chat template before visual expansion.
    pub rendered_prompt: String,
    /// Prompt after image/video placeholders have been expanded.
    pub expanded_prompt: String,
    /// Replacements in prompt encounter order.
    pub replacements: Vec<TextReplacement>,
}

/// Owned text-stage output with official C-contiguous arrays.
#[derive(Clone, Debug, PartialEq)]
pub struct PreparedTextBatch {
    /// Per-request prompt bytes and replacement coordinates.
    pub requests: Vec<PreparedTextRequest>,
    /// `int64 [batch, sequence]`, right padded.
    pub input_ids: Matrix<i64>,
    /// `int64 [batch, sequence]`, zero on padding.
    pub attention_mask: Matrix<i64>,
    /// `int64 [batch, sequence]`: text/padding 0, image 1, video 2.
    pub mm_token_type_ids: Matrix<i64>,
}

/// A hash-validated, profile-bound tokenizer and chat renderer.
pub struct TextProcessor {
    profile: Profile,
    tokenizer: Tokenizer,
}

pub(crate) struct SingleTextPlan {
    pending: PendingTextRequest,
    token_row: EncodedTokenRow,
    additional_output_bytes: u64,
}

/// Reusable token and metadata plan for a homogeneous-profile batch.
#[derive(Clone, Debug)]
pub(crate) struct BatchTextPlan {
    requests: Vec<PreparedTextRequest>,
    token_rows: Vec<Vec<u32>>,
    rows: usize,
    columns: usize,
}

impl BatchTextPlan {
    pub(crate) const fn rows(&self) -> usize {
        self.rows
    }

    pub(crate) const fn columns(&self) -> usize {
        self.columns
    }

    pub(crate) fn requests(&self) -> &[PreparedTextRequest] {
        &self.requests
    }

    pub(crate) fn token_count(&self, request_index: usize) -> usize {
        self.token_rows[request_index].len()
    }

    pub(crate) fn count_token(&self, token_id: i64) -> usize {
        self.token_rows
            .iter()
            .flatten()
            .filter(|&&value| i64::from(value) == token_id)
            .count()
    }
}

impl TextProcessor {
    /// Loads only local model assets and binds them to an immutable profile.
    ///
    /// `assets_directory` must directly contain the pinned repository snapshot
    /// files. The tokenizer, tokenizer configuration, and profile-specific chat
    /// template are SHA-256 checked before any of them are parsed. This function
    /// performs no network access and has no fallback asset source.
    ///
    /// # Errors
    ///
    /// Returns `profile_mismatch` for a changed profile, absent or changed asset,
    /// malformed tokenizer/configuration, or special-token identity mismatch.
    pub fn from_local_assets(
        profile: &Profile,
        assets_directory: impl AsRef<Path>,
    ) -> Result<Self> {
        let registry = ProfileRegistry::bundled()?;
        let expected_profile = registry.get(profile.alias);
        if profile != expected_profile {
            return Err(
                profile_error("tokenizer profile is not the immutable bundled profile")
                    .with_context("profile", profile.alias.as_str()),
            );
        }

        let directory = assets_directory.as_ref();
        let template_name = match profile.alias {
            ProfileAlias::Qwen3Vl8b => "chat_template.json",
            ProfileAlias::Qwen35_9b => "chat_template.jinja",
        };
        for asset in ["tokenizer.json", "tokenizer_config.json", template_name] {
            validate_asset(profile, directory, asset)?;
        }
        validate_template_asset(profile.alias, &directory.join(template_name))?;
        validate_tokenizer_config(profile, &directory.join("tokenizer_config.json"))?;

        let tokenizer_path = directory.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|_| {
            profile_error("pinned tokenizer asset could not be parsed")
                .with_context("asset", "tokenizer.json")
                .with_context("profile", profile.alias.as_str())
        })?;
        validate_tokenizer_semantics(profile, &tokenizer)?;

        Ok(Self {
            profile: profile.clone(),
            tokenizer,
        })
    }

    /// Returns the immutable profile used by this processor.
    #[must_use]
    pub const fn profile(&self) -> &Profile {
        &self.profile
    }

    /// Renders once and proves that the resulting visual-marker sequence is
    /// exactly the request's media-occurrence sequence.
    pub(crate) fn render_validated_request(
        &self,
        request: &Request<'_>,
        request_index: usize,
    ) -> Result<String> {
        let rendered = render_chat(&self.profile, request, request_index)?;
        validate_rendered_visual_sequence(request, &rendered, request_index)?;
        Ok(rendered)
    }

    /// Renders, expands, tokenizes, and right-pads a homogeneous-profile batch.
    ///
    /// Visual work is deliberately not performed here. Each supplied visual
    /// entry must match one request content occurrence by modality and input
    /// index, and its grid/timestamps determine the exact replacement text.
    ///
    /// # Errors
    ///
    /// Returns the stable v1 category for request/option/preflight failures,
    /// malformed visual plans, token limits, arithmetic failures, or tokenizer
    /// invariants. No output matrix is built until all token limits pass.
    pub fn prepare_batch(
        &self,
        batch: &[PlannedTextRequest<'_>],
        limits: ResourceLimits,
    ) -> Result<PreparedTextBatch> {
        self.prepare_batch_with_additional_output_bytes(batch, limits, 0)
    }

    /// Prepares text while reserving part of the materialized-output budget
    /// for arrays produced by another composed stage.
    pub(crate) fn prepare_batch_with_additional_output_bytes(
        &self,
        batch: &[PlannedTextRequest<'_>],
        limits: ResourceLimits,
        additional_output_bytes: u64,
    ) -> Result<PreparedTextBatch> {
        let registry = ProfileRegistry::bundled()?;
        let profiled = batch
            .iter()
            .map(|item| ProfiledRequest {
                profile_alias: self.profile.alias.as_str(),
                request: item.request,
            })
            .collect::<Vec<_>>();
        preflight_batch(&registry, &profiled, limits)?;
        let unpadded = self.prepare_unpadded(batch)?;
        limits.check_rendered_tokens(&unpadded.token_counts)?;
        self.materialize_text_batch(
            unpadded.pending,
            unpadded.token_rows,
            limits,
            additional_output_bytes,
        )
    }

    /// Expands/tokenizes one already-rendered request and checks its complete
    /// composed output capacity without materializing official matrices.
    pub(crate) fn plan_single_from_rendered(
        &self,
        request: Request<'_>,
        visuals: &[VisualExpansion<'_>],
        rendered_prompt: String,
        limits: ResourceLimits,
        additional_output_bytes: u64,
    ) -> Result<SingleTextPlan> {
        validate_visual_plan(&request, visuals, 0)?;
        validate_rendered_visual_sequence(&request, &rendered_prompt, 0)?;
        let (expanded_prompt, replacements) =
            expand_visuals(&self.profile, &rendered_prompt, visuals, 0)?;
        let token_row = self.encode_prompt_once(&expanded_prompt, 0)?;
        let token_count = u64::try_from(token_row.ids.len())
            .map_err(|_| arithmetic("rendered token count does not fit parity arithmetic"))?;
        limits.check_rendered_tokens(&[token_count])?;
        check_text_output_capacity(1, token_row.ids.len(), limits, additional_output_bytes)?;
        Ok(SingleTextPlan {
            pending: PendingTextRequest {
                rendered_prompt,
                expanded_prompt,
                replacements,
            },
            token_row,
            additional_output_bytes,
        })
    }

    /// Materializes a previously checked single-request text plan without
    /// rerendering, expanding, or tokenizing it.
    pub(crate) fn execute_single_text_plan(
        &self,
        plan: SingleTextPlan,
        limits: ResourceLimits,
    ) -> Result<PreparedTextBatch> {
        self.materialize_text_batch(
            vec![plan.pending],
            vec![plan.token_row],
            limits,
            plan.additional_output_bytes,
        )
    }

    /// Expands and tokenizes already-rendered image requests without creating
    /// any official output matrix. The returned plan can be written into more
    /// than one caller-owned destination set.
    pub(crate) fn plan_batch_from_rendered<'a>(
        &self,
        requests: &[Request<'a>],
        visuals: &[Vec<VisualExpansion<'a>>],
        rendered_prompts: Vec<String>,
        limits: ResourceLimits,
        additional_output_bytes: u64,
    ) -> Result<BatchTextPlan> {
        if requests.len() != visuals.len() || requests.len() != rendered_prompts.len() {
            return Err(invariant(
                "batch text planning inputs have inconsistent request counts",
            ));
        }

        let mut prepared_requests = Vec::with_capacity(requests.len());
        let mut token_rows = Vec::with_capacity(requests.len());
        let mut token_counts = Vec::with_capacity(requests.len());
        for (request_index, ((request, visual_plan), rendered_prompt)) in requests
            .iter()
            .zip(visuals)
            .zip(rendered_prompts)
            .enumerate()
        {
            validate_visual_plan(request, visual_plan, request_index)?;
            validate_rendered_visual_sequence(request, &rendered_prompt, request_index)?;
            let (expanded_prompt, replacements) =
                expand_visuals(&self.profile, &rendered_prompt, visual_plan, request_index)?;
            let token_row = self.encode_prompt_once(&expanded_prompt, request_index)?;
            let count = u64::try_from(token_row.ids.len()).map_err(|_| {
                arithmetic("rendered token count does not fit parity arithmetic")
                    .with_context("request_index", request_index)
            })?;
            let replacements =
                Self::locate_replacement_tokens(&token_row.offsets, replacements, request_index)?;
            prepared_requests.push(PreparedTextRequest {
                rendered_prompt,
                expanded_prompt,
                replacements,
            });
            token_counts.push(count);
            token_rows.push(token_row.ids);
        }
        limits.check_rendered_tokens(&token_counts)?;
        let columns = token_rows
            .iter()
            .map(Vec::len)
            .max()
            .ok_or_else(|| invariant("validated batch disappeared"))?;
        check_text_output_capacity(requests.len(), columns, limits, additional_output_bytes)?;
        Ok(BatchTextPlan {
            requests: prepared_requests,
            token_rows,
            rows: requests.len(),
            columns,
        })
    }

    /// Writes one prevalidated text plan into exact or oversized row-major
    /// slices. The composed processor validates all destinations before this
    /// method is called, so this operation itself is infallible.
    pub(crate) fn write_batch_plan(
        &self,
        plan: &BatchTextPlan,
        input_ids: &mut [i64],
        attention_mask: &mut [i64],
        mm_token_type_ids: &mut [i64],
    ) {
        let elements = plan.rows * plan.columns;
        debug_assert!(input_ids.len() >= elements);
        debug_assert!(attention_mask.len() >= elements);
        debug_assert!(mm_token_type_ids.len() >= elements);
        let pad_id = self.profile.tokenizer.pad_token_id;
        let image_id = self.profile.tokenizer.image_token_id;
        let video_id = self.profile.tokenizer.video_token_id;

        for (row_index, ids) in plan.token_rows.iter().enumerate() {
            let start = row_index * plan.columns;
            let row_ids = &mut input_ids[start..start + plan.columns];
            let row_attention = &mut attention_mask[start..start + plan.columns];
            let row_modalities = &mut mm_token_type_ids[start..start + plan.columns];
            for (column, &id) in ids.iter().enumerate() {
                let id = i64::from(id);
                row_ids[column] = id;
                row_attention[column] = 1;
                row_modalities[column] = if id == image_id {
                    1
                } else if id == video_id {
                    2
                } else {
                    0
                };
            }
            for column in ids.len()..plan.columns {
                row_ids[column] = pad_id;
                row_attention[column] = 0;
                row_modalities[column] = 0;
            }
        }
    }

    pub(crate) fn image_token_count(&self, plan: &BatchTextPlan) -> usize {
        plan.count_token(self.profile.tokenizer.image_token_id)
    }

    fn prepare_unpadded(&self, batch: &[PlannedTextRequest<'_>]) -> Result<UnpaddedTextBatch> {
        let mut pending = Vec::with_capacity(batch.len());
        let mut token_rows = Vec::with_capacity(batch.len());
        let mut token_counts = Vec::with_capacity(batch.len());
        for (request_index, item) in batch.iter().enumerate() {
            validate_visual_plan(&item.request, item.visuals, request_index)?;
            let rendered_prompt = self.render_validated_request(&item.request, request_index)?;
            let (expanded_prompt, replacements) =
                expand_visuals(&self.profile, &rendered_prompt, item.visuals, request_index)?;
            let token_row = self.encode_prompt_once(&expanded_prompt, request_index)?;
            let count = u64::try_from(token_row.ids.len()).map_err(|_| {
                arithmetic("rendered token count does not fit parity arithmetic")
                    .with_context("request_index", request_index)
            })?;
            token_counts.push(count);
            pending.push(PendingTextRequest {
                rendered_prompt,
                expanded_prompt,
                replacements,
            });
            token_rows.push(token_row);
        }
        Ok(UnpaddedTextBatch {
            pending,
            token_rows,
            token_counts,
        })
    }

    fn materialize_text_batch(
        &self,
        pending: Vec<PendingTextRequest>,
        token_rows: Vec<EncodedTokenRow>,
        limits: ResourceLimits,
        additional_output_bytes: u64,
    ) -> Result<PreparedTextBatch> {
        let columns = token_rows
            .iter()
            .map(|row| row.ids.len())
            .max()
            .ok_or_else(|| {
                QwenError::new(
                    ErrorCategory::InternalInvariant,
                    "validated batch disappeared",
                )
            })?;
        let rows = token_rows.len();
        check_text_output_capacity(rows, columns, limits, additional_output_bytes)?;

        let capacity = rows.checked_mul(columns).ok_or_else(|| {
            arithmetic("text matrix element capacity overflowed")
                .with_context("rows", rows)
                .with_context("columns", columns)
        })?;
        let mut input_ids = Vec::with_capacity(capacity);
        let mut attention_mask = Vec::with_capacity(capacity);
        let mut mm_token_type_ids = Vec::with_capacity(capacity);
        let pad_id = self.profile.tokenizer.pad_token_id;
        let image_id = self.profile.tokenizer.image_token_id;
        let video_id = self.profile.tokenizer.video_token_id;
        let mut prepared_requests = Vec::with_capacity(rows);

        for (row_index, (row, request)) in token_rows.into_iter().zip(pending).enumerate() {
            let replacements =
                Self::locate_replacement_tokens(&row.offsets, request.replacements, row_index)?;
            let ids = row.ids;

            for &id in &ids {
                let id = i64::from(id);
                input_ids.push(id);
                attention_mask.push(1);
                mm_token_type_ids.push(if id == image_id {
                    1
                } else if id == video_id {
                    2
                } else {
                    0
                });
            }
            let padding = columns - ids.len();
            input_ids.extend(std::iter::repeat_n(pad_id, padding));
            attention_mask.extend(std::iter::repeat_n(0, padding));
            mm_token_type_ids.extend(std::iter::repeat_n(0, padding));
            prepared_requests.push(PreparedTextRequest {
                rendered_prompt: request.rendered_prompt,
                expanded_prompt: request.expanded_prompt,
                replacements,
            });
        }

        Ok(PreparedTextBatch {
            requests: prepared_requests,
            input_ids: Matrix::new(rows, columns, input_ids)?,
            attention_mask: Matrix::new(rows, columns, attention_mask)?,
            mm_token_type_ids: Matrix::new(rows, columns, mm_token_type_ids)?,
        })
    }

    fn locate_replacement_tokens(
        offsets: &[(usize, usize)],
        pending: Vec<PendingReplacement>,
        request_index: usize,
    ) -> Result<Vec<TextReplacement>> {
        let mut replacements = Vec::with_capacity(pending.len());
        let mut token_cursor = 0;
        for replacement in pending {
            let start = offsets
                .iter()
                .enumerate()
                .skip(token_cursor)
                .find_map(|(index, &(start, end))| {
                    (start == replacement.expanded_byte_start && end > start).then_some(index)
                })
                .ok_or_else(|| {
                    invariant("expanded replacement start was not found in full-prompt offsets")
                        .with_context("request_index", request_index)
                })?;
            let end = offsets
                .iter()
                .enumerate()
                .skip(start)
                .take_while(|(_, (start, _))| *start < replacement.expanded_byte_end)
                .filter(|(_, (start, end))| *end > *start && *end <= replacement.expanded_byte_end)
                .map(|(index, _)| index + 1)
                .last()
                .ok_or_else(|| {
                    invariant("expanded replacement end was not found in full-prompt offsets")
                        .with_context("request_index", request_index)
                })?;
            if offsets[end - 1].1 != replacement.expanded_byte_end {
                return Err(invariant(
                    "full-prompt token offsets do not cover the visual replacement exactly",
                )
                .with_context("request_index", request_index));
            }
            token_cursor = end;
            replacements.push(TextReplacement {
                modality: replacement.modality,
                rendered_code_points: replacement.rendered_code_points,
                expanded_code_points: replacement.expanded_code_points,
                expanded_tokens: CoordinateRange {
                    start: usize_to_i64(start, "replacement token start")?,
                    end: usize_to_i64(end, "replacement token end")?,
                },
            });
        }
        Ok(replacements)
    }

    fn encode_prompt_once(
        &self,
        expanded_prompt: &str,
        request_index: usize,
    ) -> Result<EncodedTokenRow> {
        let encoding = self.tokenizer.encode(expanded_prompt, true).map_err(|_| {
            QwenError::new(
                ErrorCategory::InternalInvariant,
                "pinned tokenizer failed to encode a validated prompt",
            )
            .with_context("request_index", request_index)
        })?;
        Ok(EncodedTokenRow {
            ids: encoding.get_ids().to_vec(),
            offsets: encoding.get_offsets().to_vec(),
        })
    }
}

fn check_text_output_capacity(
    rows: usize,
    columns: usize,
    limits: ResourceLimits,
    additional_output_bytes: u64,
) -> Result<()> {
    let rows_u64 = u64::try_from(rows)
        .map_err(|_| arithmetic("text batch rows do not fit parity arithmetic"))?;
    let columns_u64 = u64::try_from(columns)
        .map_err(|_| arithmetic("text batch columns do not fit parity arithmetic"))?;
    let one_matrix = checked_capacity_bytes(rows_u64, columns_u64, 8)?;
    let text_bytes = checked_mul("three text output matrices", one_matrix, 3)?;
    limits.check_materialized_output_bytes(crate::limits::checked_add(
        "composed materialized output bytes",
        text_bytes,
        additional_output_bytes,
    )?)
}

fn validate_rendered_visual_sequence(
    request: &Request<'_>,
    rendered: &str,
    request_index: usize,
) -> Result<()> {
    let mut expected = Vec::new();
    for message in request.messages {
        if let MessageContent::Items(items) = message.content {
            for item in items {
                match item {
                    ContentItem::Image(_) => expected.push(VisualModality::Image),
                    ContentItem::Video(_) => expected.push(VisualModality::Video),
                    ContentItem::Text(_) => {}
                }
            }
        }
    }

    let mut observed = rendered
        .match_indices(IMAGE_TOKEN)
        .map(|(offset, _)| (offset, VisualModality::Image))
        .chain(
            rendered
                .match_indices(VIDEO_TOKEN)
                .map(|(offset, _)| (offset, VisualModality::Video)),
        )
        .collect::<Vec<_>>();
    observed.sort_by_key(|(offset, _)| *offset);
    if observed.len() != expected.len() {
        return Err(
            invalid("rendered placeholder count does not match media occurrences")
                .with_context("request_index", request_index)
                .with_context("expected", expected.len())
                .with_context("actual", observed.len()),
        );
    }
    for (occurrence, (expected, (_, actual))) in expected.iter().zip(&observed).enumerate() {
        if expected != actual {
            return Err(
                invalid("rendered placeholder order does not match media traversal")
                    .with_context("request_index", request_index)
                    .with_context("occurrence", occurrence),
            );
        }
    }
    Ok(())
}

#[derive(Debug)]
struct UnpaddedTextBatch {
    pending: Vec<PendingTextRequest>,
    token_rows: Vec<EncodedTokenRow>,
    token_counts: Vec<u64>,
}

#[derive(Debug)]
struct PendingTextRequest {
    rendered_prompt: String,
    expanded_prompt: String,
    replacements: Vec<PendingReplacement>,
}

#[derive(Debug)]
struct PendingReplacement {
    modality: VisualModality,
    rendered_code_points: CoordinateRange,
    expanded_code_points: CoordinateRange,
    expanded_byte_start: usize,
    expanded_byte_end: usize,
}

#[derive(Debug)]
struct EncodedTokenRow {
    ids: Vec<u32>,
    offsets: Vec<(usize, usize)>,
}

fn validate_asset(profile: &Profile, directory: &Path, name: &'static str) -> Result<()> {
    let expected = profile.artifact_hash(name).ok_or_else(|| {
        profile_error("required tokenizer artifact is absent from the profile")
            .with_context("asset", name)
            .with_context("profile", profile.alias.as_str())
    })?;
    let bytes = fs::read(directory.join(name)).map_err(|_| {
        profile_error("required local tokenizer artifact is unavailable")
            .with_context("asset", name)
            .with_context("profile", profile.alias.as_str())
    })?;
    let actual = format!("{:x}", Sha256::digest(&bytes));
    if actual != expected {
        return Err(
            profile_error("local tokenizer artifact hash does not match profile")
                .with_context("asset", name)
                .with_context("profile", profile.alias.as_str())
                .with_context("expected_sha256", expected)
                .with_context("actual_sha256", actual),
        );
    }
    Ok(())
}

fn validate_template_asset(alias: ProfileAlias, path: &Path) -> Result<()> {
    let bytes = fs::read(path).map_err(|_| profile_error("chat template could not be read"))?;
    match alias {
        ProfileAlias::Qwen3Vl8b => {
            let value: serde_json::Value = serde_json::from_slice(&bytes)
                .map_err(|_| profile_error("chat template asset is malformed"))?;
            if !value
                .get("chat_template")
                .is_some_and(serde_json::Value::is_string)
            {
                return Err(profile_error("chat template JSON has no string template"));
            }
        }
        ProfileAlias::Qwen35_9b => {
            std::str::from_utf8(&bytes)
                .map_err(|_| profile_error("chat template asset is not UTF-8"))?;
        }
    }
    Ok(())
}

fn validate_tokenizer_config(profile: &Profile, path: &Path) -> Result<()> {
    let bytes =
        fs::read(path).map_err(|_| profile_error("tokenizer configuration could not be read"))?;
    let value: serde_json::Value = serde_json::from_slice(&bytes)
        .map_err(|_| profile_error("tokenizer configuration is malformed"))?;
    let model_max_length = value
        .get("model_max_length")
        .and_then(serde_json::Value::as_u64);
    let bos = value.get("bos_token").and_then(serde_json::Value::as_str);
    let eos = value.get("eos_token").and_then(serde_json::Value::as_str);
    let pad = value.get("pad_token").and_then(serde_json::Value::as_str);
    if model_max_length != Some(profile.tokenizer.model_max_length)
        || bos.is_some()
        || eos != Some("<|im_end|>")
        || pad != Some("<|endoftext|>")
    {
        return Err(
            profile_error("tokenizer configuration semantics do not match profile")
                .with_context("profile", profile.alias.as_str()),
        );
    }
    Ok(())
}

fn validate_tokenizer_semantics(profile: &Profile, tokenizer: &Tokenizer) -> Result<()> {
    let expected = [
        ("<|endoftext|>", profile.tokenizer.pad_token_id),
        ("<|im_end|>", profile.tokenizer.eos_token_id),
        (IMAGE_TOKEN, profile.tokenizer.image_token_id),
        (VIDEO_TOKEN, profile.tokenizer.video_token_id),
        (VISION_START_TOKEN, profile.tokenizer.vision_start_token_id),
        (VISION_END_TOKEN, profile.tokenizer.vision_end_token_id),
    ];
    for (token, expected_id) in expected {
        if tokenizer.token_to_id(token).map(i64::from) != Some(expected_id) {
            return Err(
                profile_error("tokenizer special-token identity does not match profile")
                    .with_context("profile", profile.alias.as_str())
                    .with_context("token", token)
                    .with_context("expected_id", expected_id),
            );
        }
    }
    if tokenizer.get_padding().is_some() || tokenizer.get_truncation().is_some() {
        return Err(profile_error(
            "tokenizer asset contains unexpected padding or truncation",
        ));
    }
    Ok(())
}

fn validate_visual_plan(
    request: &Request<'_>,
    visuals: &[VisualExpansion<'_>],
    request_index: usize,
) -> Result<()> {
    let mut expected = Vec::new();
    for message in request.messages {
        if let MessageContent::Items(items) = message.content {
            for item in items {
                match item {
                    ContentItem::Image(reference) => {
                        expected.push((VisualModality::Image, reference.input_index));
                    }
                    ContentItem::Video(reference) => {
                        expected.push((VisualModality::Video, reference.input_index));
                    }
                    ContentItem::Text(_) => {}
                }
            }
        }
    }
    if expected.len() != visuals.len() {
        return Err(
            invalid("visual plan occurrence count does not match request")
                .with_context("request_index", request_index)
                .with_context("expected", expected.len())
                .with_context("actual", visuals.len()),
        );
    }
    for (occurrence, (&(kind, input_index), &visual)) in expected.iter().zip(visuals).enumerate() {
        if visual.modality() != kind || visual.input_index() != input_index {
            return Err(
                invalid("visual plan occurrence does not match request traversal")
                    .with_context("request_index", request_index)
                    .with_context("occurrence", occurrence),
            );
        }
    }
    Ok(())
}

fn render_chat(profile: &Profile, request: &Request<'_>, request_index: usize) -> Result<String> {
    match profile.alias {
        ProfileAlias::Qwen3Vl8b => render_qwen3_vl(request),
        ProfileAlias::Qwen35_9b => render_qwen35(request, request_index),
    }
}

fn render_qwen3_vl(request: &Request<'_>) -> Result<String> {
    const TOOLS_HEADER: &str = "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>";
    const TOOLS_FOOTER: &str = "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n";

    let mut output = String::new();
    if request.options.tools.is_empty() {
        if request.messages[0].role == Role::System {
            output.push_str("<|im_start|>system\n");
            render_plain_content(&mut output, request.messages[0].content);
            output.push_str("<|im_end|>\n");
        }
    } else {
        output.push_str("<|im_start|>system\n");
        if request.messages[0].role == Role::System {
            render_plain_content(&mut output, request.messages[0].content);
            output.push_str("\n\n");
        }
        output.push_str(TOOLS_HEADER);
        for tool in request.options.tools {
            output.push('\n');
            output.push_str(&parse_ordered_json(tool.json, "tool definition")?.json_dumps());
        }
        output.push_str(TOOLS_FOOTER);
    }

    let mut image_count = 0_usize;
    let mut video_count = 0_usize;
    for (message_index, message) in request.messages.iter().enumerate() {
        match message.role {
            Role::System => {}
            Role::User => {
                output.push_str("<|im_start|>user\n");
                render_visual_content(
                    &mut output,
                    message.content,
                    request.options.add_vision_id,
                    &mut image_count,
                    &mut video_count,
                );
                output.push_str("<|im_end|>\n");
            }
            Role::Assistant => {
                output.push_str("<|im_start|>assistant\n");
                render_plain_content(&mut output, message.content);
                for (tool_call_index, tool_call) in message.tool_calls.iter().enumerate() {
                    if tool_call_index != 0 || content_is_truthy(message.content) {
                        output.push('\n');
                    }
                    let function = tool_call.function_call();
                    let arguments =
                        parse_ordered_json(function.arguments_json, "tool-call arguments")?;
                    write!(
                        output,
                        "<tool_call>\n{{\"name\": \"{}\", \"arguments\": {}}}\n</tool_call>",
                        function.name,
                        arguments.json_dumps()
                    )
                    .expect("writing to String cannot fail");
                }
                output.push_str("<|im_end|>\n");
            }
            Role::Tool => {
                if message_index == 0 || request.messages[message_index - 1].role != Role::Tool {
                    output.push_str("<|im_start|>user");
                }
                output.push_str("\n<tool_response>\n");
                render_visual_content(
                    &mut output,
                    message.content,
                    request.options.add_vision_id,
                    &mut image_count,
                    &mut video_count,
                );
                output.push_str("\n</tool_response>");
                if message_index + 1 == request.messages.len()
                    || request.messages[message_index + 1].role != Role::Tool
                {
                    output.push_str("<|im_end|>\n");
                }
            }
        }
    }
    if request.options.add_generation_prompt {
        output.push_str("<|im_start|>assistant\n");
    }
    Ok(output)
}

fn render_qwen35(request: &Request<'_>, request_index: usize) -> Result<String> {
    let mut output = render_qwen35_preamble(request)?;
    let last_query_index = qwen35_last_query_index(request, request_index)?;
    let mut image_count = 0_usize;
    let mut video_count = 0_usize;
    for (message_index, message) in request.messages.iter().enumerate() {
        let mut content = String::new();
        render_visual_content(
            &mut content,
            message.content,
            request.options.add_vision_id,
            &mut image_count,
            &mut video_count,
        );
        let mut content = content.trim().to_owned();
        match message.role {
            Role::System => {}
            Role::User => {
                write!(output, "<|im_start|>user\n{content}<|im_end|>\n")
                    .expect("writing to String cannot fail");
            }
            Role::Assistant => render_qwen35_assistant(
                &mut output,
                message,
                &mut content,
                message_index,
                last_query_index,
                request_index,
            )?,
            Role::Tool => {
                if message_index != 0 && request.messages[message_index - 1].role != Role::Tool {
                    output.push_str("<|im_start|>user");
                }
                output.push_str("\n<tool_response>\n");
                output.push_str(&content);
                output.push_str("\n</tool_response>");
                if message_index + 1 == request.messages.len()
                    || request.messages[message_index + 1].role != Role::Tool
                {
                    output.push_str("<|im_end|>\n");
                }
            }
        }
    }
    if request.options.add_generation_prompt {
        output.push_str("<|im_start|>assistant\n");
        if request.options.enable_thinking == Some(false) {
            output.push_str("<think>\n\n</think>\n\n");
        } else {
            output.push_str("<think>\n");
        }
    }
    Ok(output)
}

fn render_qwen35_preamble(request: &Request<'_>) -> Result<String> {
    const TOOLS_INSTRUCTIONS: &str = "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>";

    let mut output = String::new();
    if request.options.tools.is_empty() {
        if request.messages[0].role == Role::System {
            let content = plain_content(request.messages[0].content);
            write!(output, "<|im_start|>system\n{}<|im_end|>\n", content.trim())
                .expect("writing to String cannot fail");
        }
    } else {
        output.push_str(
            "<|im_start|>system\n# Tools\n\nYou have access to the following functions:\n\n<tools>",
        );
        for tool in request.options.tools {
            output.push('\n');
            output.push_str(&parse_ordered_json(tool.json, "tool definition")?.json_dumps());
        }
        output.push_str("\n</tools>");
        output.push_str(TOOLS_INSTRUCTIONS);
        if request.messages[0].role == Role::System {
            let content = plain_content(request.messages[0].content);
            if !content.trim().is_empty() {
                output.push_str("\n\n");
                output.push_str(content.trim());
            }
        }
        output.push_str("<|im_end|>\n");
    }
    Ok(output)
}

fn qwen35_last_query_index(request: &Request<'_>, request_index: usize) -> Result<usize> {
    request
        .messages
        .iter()
        .enumerate()
        .rev()
        .find_map(|(index, message)| {
            if message.role != Role::User {
                return None;
            }
            let content = preview_visual_content(message.content, request.options.add_vision_id);
            let content = content.trim();
            (!content.starts_with("<tool_response>") || !content.ends_with("</tool_response>"))
                .then_some(index)
        })
        .ok_or_else(|| {
            invalid("Qwen3.5 chat requires a user query")
                .with_context("request_index", request_index)
        })
}

fn render_qwen35_assistant(
    output: &mut String,
    message: &Message<'_>,
    content: &mut String,
    message_index: usize,
    last_query_index: usize,
    request_index: usize,
) -> Result<()> {
    let mut reasoning = message.reasoning_content.unwrap_or("").to_owned();
    if message.reasoning_content.is_none() && content.contains("</think>") {
        let before = content.split("</think>").next().unwrap_or_default();
        before
            .trim_end_matches('\n')
            .rsplit("<think>")
            .next()
            .unwrap_or_default()
            .trim_start_matches('\n')
            .clone_into(&mut reasoning);
        *content = content
            .rsplit("</think>")
            .next()
            .unwrap_or_default()
            .trim_start_matches('\n')
            .to_owned();
    }
    let reasoning = reasoning.trim();
    if message_index > last_query_index {
        write!(
            output,
            "<|im_start|>assistant\n<think>\n{reasoning}\n</think>\n\n{content}"
        )
        .expect("writing to String cannot fail");
    } else {
        write!(output, "<|im_start|>assistant\n{content}").expect("writing to String cannot fail");
    }

    for (tool_call_index, tool_call) in message.tool_calls.iter().enumerate() {
        if tool_call_index == 0 {
            if content.trim().is_empty() {
                output.push_str("<tool_call>\n");
            } else {
                output.push_str("\n\n<tool_call>\n");
            }
        } else {
            output.push_str("\n<tool_call>\n");
        }
        let function = tool_call.function_call();
        writeln!(output, "<function={}>", function.name).expect("writing to String cannot fail");
        let arguments = parse_ordered_json(function.arguments_json, "tool-call arguments")?;
        let OrderedJson::Object(arguments) = arguments else {
            return Err(invalid("Qwen3.5 tool-call arguments must be a JSON object")
                .with_context("request_index", request_index)
                .with_context("message_index", message_index)
                .with_context("tool_call_index", tool_call_index));
        };
        for (name, value) in arguments {
            write!(
                output,
                "<parameter={name}>\n{}\n</parameter>\n",
                value.jinja_string()
            )
            .expect("writing to String cannot fail");
        }
        output.push_str("</function>\n</tool_call>");
    }
    output.push_str("<|im_end|>\n");
    Ok(())
}

fn render_plain_content(output: &mut String, content: MessageContent<'_>) {
    match content {
        MessageContent::Text(text) => output.push_str(text),
        MessageContent::Items(items) => {
            for item in items {
                if let ContentItem::Text(text) = item {
                    output.push_str(text);
                }
            }
        }
    }
}

fn plain_content(content: MessageContent<'_>) -> String {
    let mut output = String::new();
    render_plain_content(&mut output, content);
    output
}

fn content_is_truthy(content: MessageContent<'_>) -> bool {
    match content {
        MessageContent::Text(text) => !text.is_empty(),
        MessageContent::Items(items) => !items.is_empty(),
    }
}

fn render_visual_content(
    output: &mut String,
    content: MessageContent<'_>,
    add_vision_id: bool,
    image_count: &mut usize,
    video_count: &mut usize,
) {
    match content {
        MessageContent::Text(text) => output.push_str(text),
        MessageContent::Items(items) => {
            for item in items {
                match item {
                    ContentItem::Text(text) => output.push_str(text),
                    ContentItem::Image(_) => {
                        *image_count += 1;
                        if add_vision_id {
                            write!(output, "Picture {}: ", *image_count)
                                .expect("writing to String cannot fail");
                        }
                        output.push_str(VISION_START_TOKEN);
                        output.push_str(IMAGE_TOKEN);
                        output.push_str(VISION_END_TOKEN);
                    }
                    ContentItem::Video(_) => {
                        *video_count += 1;
                        if add_vision_id {
                            write!(output, "Video {}: ", *video_count)
                                .expect("writing to String cannot fail");
                        }
                        output.push_str(VISION_START_TOKEN);
                        output.push_str(VIDEO_TOKEN);
                        output.push_str(VISION_END_TOKEN);
                    }
                }
            }
        }
    }
}

fn preview_visual_content(content: MessageContent<'_>, add_vision_id: bool) -> String {
    let mut output = String::new();
    let mut images = 0;
    let mut videos = 0;
    render_visual_content(
        &mut output,
        content,
        add_vision_id,
        &mut images,
        &mut videos,
    );
    output
}

fn expand_visuals(
    profile: &Profile,
    rendered: &str,
    visuals: &[VisualExpansion<'_>],
    request_index: usize,
) -> Result<(String, Vec<PendingReplacement>)> {
    let markers = find_visual_markers(rendered);
    if markers.len() != visuals.len()
        || markers
            .iter()
            .zip(visuals)
            .any(|((_, kind), visual)| *kind != visual.modality())
    {
        return Err(
            invalid("rendered visual placeholders do not match supplied occurrences")
                .with_context("request_index", request_index)
                .with_context("rendered_placeholders", markers.len())
                .with_context("visual_occurrences", visuals.len()),
        );
    }

    let mut expanded = String::with_capacity(rendered.len());
    let mut replacements = Vec::with_capacity(visuals.len());
    let mut last_byte = 0;
    for (occurrence, ((byte_start, modality), &visual)) in
        markers.into_iter().zip(visuals).enumerate()
    {
        let token = match modality {
            VisualModality::Image => IMAGE_TOKEN,
            VisualModality::Video => VIDEO_TOKEN,
        };
        let byte_end = byte_start + token.len();
        expanded.push_str(&rendered[last_byte..byte_start]);
        let replacement = visual_replacement(profile, visual, request_index, occurrence)?;
        let expanded_byte_start = expanded.len();
        let expanded_start = expanded.chars().count();
        expanded.push_str(&replacement);
        let expanded_byte_end = expanded.len();
        let expanded_end = expanded.chars().count();
        replacements.push(PendingReplacement {
            modality,
            rendered_code_points: CoordinateRange {
                start: usize_to_i64(
                    rendered[..byte_start].chars().count(),
                    "rendered span start",
                )?,
                end: usize_to_i64(rendered[..byte_end].chars().count(), "rendered span end")?,
            },
            expanded_code_points: CoordinateRange {
                start: usize_to_i64(expanded_start, "expanded span start")?,
                end: usize_to_i64(expanded_end, "expanded span end")?,
            },
            expanded_byte_start,
            expanded_byte_end,
        });
        last_byte = byte_end;
    }
    expanded.push_str(&rendered[last_byte..]);
    Ok((expanded, replacements))
}

fn find_visual_markers(text: &str) -> Vec<(usize, VisualModality)> {
    let mut markers = text
        .match_indices(IMAGE_TOKEN)
        .map(|(index, _)| (index, VisualModality::Image))
        .chain(
            text.match_indices(VIDEO_TOKEN)
                .map(|(index, _)| (index, VisualModality::Video)),
        )
        .collect::<Vec<_>>();
    markers.sort_by_key(|(index, _)| *index);
    markers
}

fn visual_replacement(
    profile: &Profile,
    visual: VisualExpansion<'_>,
    request_index: usize,
    occurrence: usize,
) -> Result<String> {
    let merge_area = checked_mul(
        "visual merge area",
        profile.visual.merge_size,
        profile.visual.merge_size,
    )?;
    match visual {
        VisualExpansion::Image { grid_thw, .. } => {
            let patches = checked_grid_product(grid_thw)?;
            if patches == 0 || patches % merge_area != 0 {
                return Err(invalid(
                    "image grid does not produce an integral positive token count",
                )
                .with_context("request_index", request_index)
                .with_context("occurrence", occurrence));
            }
            let count = usize::try_from(patches / merge_area)
                .map_err(|_| arithmetic("image placeholder count does not fit memory"))?;
            Ok(IMAGE_TOKEN.repeat(count))
        }
        VisualExpansion::Video {
            grid_thw,
            timestamps,
            ..
        } => {
            let frame_count = usize::try_from(grid_thw[0])
                .map_err(|_| arithmetic("video temporal grid does not fit memory"))?;
            if frame_count == 0 || timestamps.len() != frame_count {
                return Err(invalid("video timestamps do not match temporal grid")
                    .with_context("request_index", request_index)
                    .with_context("occurrence", occurrence)
                    .with_context("temporal_grid", grid_thw[0])
                    .with_context("timestamps", timestamps.len()));
            }
            let spatial = checked_mul("video spatial grid", grid_thw[1], grid_thw[2])?;
            if spatial == 0 || spatial % merge_area != 0 {
                return Err(invalid(
                    "video grid does not produce an integral positive frame token count",
                )
                .with_context("request_index", request_index)
                .with_context("occurrence", occurrence));
            }
            let tokens_per_frame = usize::try_from(spatial / merge_area)
                .map_err(|_| arithmetic("video placeholder count does not fit memory"))?;
            let mut output = String::new();
            for &timestamp in timestamps {
                if !timestamp.is_finite() {
                    return Err(invalid("video timestamp must be finite")
                        .with_context("request_index", request_index)
                        .with_context("occurrence", occurrence));
                }
                write!(output, "<{timestamp:.1} seconds>").expect("writing to String cannot fail");
                output.push_str(VISION_START_TOKEN);
                for _ in 0..tokens_per_frame {
                    output.push_str(VIDEO_TOKEN);
                }
                output.push_str(VISION_END_TOKEN);
            }
            Ok(output)
        }
    }
}

fn checked_grid_product(grid: [u64; 3]) -> Result<u64> {
    let temporal_height = checked_mul("visual temporal-height grid", grid[0], grid[1])?;
    checked_mul("visual grid patch count", temporal_height, grid[2])
}

#[derive(Clone, Debug, PartialEq)]
enum OrderedJson {
    Null,
    Bool(bool),
    Number(serde_json::Number),
    String(String),
    Array(Vec<Self>),
    Object(Vec<(String, Self)>),
}

impl OrderedJson {
    fn json_dumps(&self) -> String {
        match self {
            Self::Null => "null".to_owned(),
            Self::Bool(value) => value.to_string(),
            Self::Number(value) => python_json_number(value),
            Self::String(value) => serde_json::to_string(value).expect("a String is valid JSON"),
            Self::Array(values) => {
                let body = values
                    .iter()
                    .map(Self::json_dumps)
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("[{body}]")
            }
            Self::Object(values) => {
                let body = values
                    .iter()
                    .map(|(key, value)| {
                        format!(
                            "{}: {}",
                            serde_json::to_string(key).expect("a String is valid JSON"),
                            value.json_dumps()
                        )
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("{{{body}}}")
            }
        }
    }

    fn jinja_string(&self) -> String {
        match self {
            Self::Null => "None".to_owned(),
            Self::Bool(true) => "True".to_owned(),
            Self::Bool(false) => "False".to_owned(),
            Self::Number(value) => python_json_number(value),
            Self::String(value) => value.clone(),
            Self::Array(_) | Self::Object(_) => self.json_dumps(),
        }
    }
}

fn python_json_number(number: &serde_json::Number) -> String {
    if !number.is_f64() {
        return number.to_string();
    }
    let value = number
        .as_f64()
        .expect("an f64 JSON number has an f64 value");
    let magnitude = value.abs();
    if value != 0.0 && !(1.0e-4..1.0e16).contains(&magnitude) {
        let scientific = format!("{value:e}");
        let (mantissa, exponent) = scientific
            .split_once('e')
            .expect("lower-exponent formatting contains an exponent");
        let exponent = exponent
            .parse::<i32>()
            .expect("formatted exponent is an integer");
        let sign = if exponent < 0 { '-' } else { '+' };
        format!("{mantissa}e{sign}{:02}", exponent.unsigned_abs())
    } else {
        number.to_string()
    }
}

impl<'de> Deserialize<'de> for OrderedJson {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct OrderedJsonVisitor;

        impl<'de> Visitor<'de> for OrderedJsonVisitor {
            type Value = OrderedJson;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("a JSON value")
            }

            fn visit_bool<E>(self, value: bool) -> std::result::Result<Self::Value, E> {
                Ok(OrderedJson::Bool(value))
            }

            fn visit_i64<E>(self, value: i64) -> std::result::Result<Self::Value, E> {
                Ok(OrderedJson::Number(value.into()))
            }

            fn visit_u64<E>(self, value: u64) -> std::result::Result<Self::Value, E> {
                Ok(OrderedJson::Number(value.into()))
            }

            fn visit_f64<E>(self, value: f64) -> std::result::Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                serde_json::Number::from_f64(value)
                    .map(OrderedJson::Number)
                    .ok_or_else(|| E::custom("non-finite JSON number"))
            }

            fn visit_str<E>(self, value: &str) -> std::result::Result<Self::Value, E> {
                Ok(OrderedJson::String(value.to_owned()))
            }

            fn visit_string<E>(self, value: String) -> std::result::Result<Self::Value, E> {
                Ok(OrderedJson::String(value))
            }

            fn visit_none<E>(self) -> std::result::Result<Self::Value, E> {
                Ok(OrderedJson::Null)
            }

            fn visit_unit<E>(self) -> std::result::Result<Self::Value, E> {
                Ok(OrderedJson::Null)
            }

            fn visit_seq<A>(self, mut sequence: A) -> std::result::Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let mut values = Vec::new();
                while let Some(value) = sequence.next_element()? {
                    values.push(value);
                }
                Ok(OrderedJson::Array(values))
            }

            fn visit_map<A>(self, mut map: A) -> std::result::Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut values: Vec<(String, OrderedJson)> = Vec::new();
                while let Some((key, value)) = map.next_entry()? {
                    if let Some((_, old_value)) =
                        values.iter_mut().find(|(old_key, _)| old_key == &key)
                    {
                        *old_value = value;
                    } else {
                        values.push((key, value));
                    }
                }
                Ok(OrderedJson::Object(values))
            }
        }

        deserializer.deserialize_any(OrderedJsonVisitor)
    }
}

fn parse_ordered_json(json: &str, label: &'static str) -> Result<OrderedJson> {
    serde_json::from_str(json).map_err(|_| invalid(format!("malformed {label}")))
}

fn usize_to_i64(value: usize, operation: &'static str) -> Result<i64> {
    i64::try_from(value).map_err(|_| {
        arithmetic("coordinate does not fit int64").with_context("operation", operation)
    })
}

fn invalid(message: impl Into<String>) -> QwenError {
    QwenError::new(ErrorCategory::InvalidRequest, message)
}

fn profile_error(message: impl Into<String>) -> QwenError {
    QwenError::new(ErrorCategory::ProfileMismatch, message)
}

fn arithmetic(message: impl Into<String>) -> QwenError {
    QwenError::new(ErrorCategory::ArithmeticOverflow, message)
}

fn invariant(message: impl Into<String>) -> QwenError {
    QwenError::new(ErrorCategory::InternalInvariant, message)
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        path::{Path, PathBuf},
    };

    use super::{
        IMAGE_TOKEN, PlannedTextRequest, TextProcessor, VisualExpansion, VisualModality,
        expand_visuals, parse_ordered_json, render_chat,
    };
    use crate::{
        ErrorCategory, FunctionCall, ImageFormat, ImageInput, ImageOptions, ImageRef,
        LimitOverrides, Message, MessageContent, ProfileAlias, ProfileRegistry, Request,
        RequestOptions, ResourceLimits, Rgb8, Role, ToolCall, ToolDefinition, VideoInput,
        VideoOptions, VideoRef, request::ContentItem,
    };

    fn assets(alias: ProfileAlias) -> PathBuf {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../reference/.cache/huggingface");
        match alias {
            ProfileAlias::Qwen3Vl8b => root.join("models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"),
            ProfileAlias::Qwen35_9b => root.join("models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a"),
        }
    }

    fn processor(alias: ProfileAlias) -> TextProcessor {
        let directory = assets(alias);
        let registry = ProfileRegistry::bundled().expect("profiles");
        TextProcessor::from_local_assets(registry.get(alias), directory).expect("pinned assets")
    }

    fn text_message(role: Role, content: &str) -> Message<'_> {
        Message {
            role,
            content: MessageContent::Text(content),
            tool_calls: &[],
            reasoning_content: None,
        }
    }

    #[test]
    fn ordered_json_matches_python_tojson_spacing_and_order() {
        let value = parse_ordered_json(
            r#"{"z": 1, "a": [true, null, "naïve"], "nested": {"b": 2}}"#,
            "test",
        )
        .expect("JSON");
        assert_eq!(
            value.json_dumps(),
            r#"{"z": 1, "a": [true, null, "naïve"], "nested": {"b": 2}}"#
        );
        let floats = parse_ordered_json(
            "[1e-7, 1e-5, 1e-4, 1e15, 1e16, 1e20, 1.0, -0.0, 1.234e-7]",
            "test",
        )
        .expect("JSON floats");
        assert_eq!(
            floats.json_dumps(),
            "[1e-07, 1e-05, 0.0001, 1000000000000000.0, 1e+16, 1e+20, 1.0, -0.0, 1.234e-07]"
        );
    }

    #[test]
    fn chat_smoke_rendering_is_always_on_for_both_profiles() {
        let registry = ProfileRegistry::bundled().expect("profiles");
        let messages = [
            text_message(Role::System, "You are a concise visual assistant."),
            text_message(Role::User, "Reply with exactly: ready"),
        ];
        let request = Request {
            messages: &messages,
            images: &[],
            videos: &[],
            options: RequestOptions {
                add_generation_prompt: true,
                ..RequestOptions::default()
            },
        };
        assert_eq!(
            render_chat(registry.get(ProfileAlias::Qwen3Vl8b), &request, 0).expect("VL render"),
            "<|im_start|>system\nYou are a concise visual assistant.<|im_end|>\n<|im_start|>user\nReply with exactly: ready<|im_end|>\n<|im_start|>assistant\n"
        );
        assert_eq!(
            render_chat(registry.get(ProfileAlias::Qwen35_9b), &request, 0).expect("3.5 render"),
            "<|im_start|>system\nYou are a concise visual assistant.<|im_end|>\n<|im_start|>user\nReply with exactly: ready<|im_end|>\n<|im_start|>assistant\n<think>\n"
        );
    }

    #[test]
    fn tool_calls_consecutive_responses_and_reasoning_render_exactly() {
        let registry = ProfileRegistry::bundled().expect("profiles");
        let calls = [ToolCall::Function {
            id: None,
            function: FunctionCall {
                name: "weather",
                arguments_json: r#"{"city":"Montréal","units":["c","f"],"flag":true,"none":null}"#,
            },
        }];
        let vl_messages = [
            text_message(Role::User, "weather?"),
            Message {
                role: Role::Assistant,
                content: MessageContent::Text("checking"),
                tool_calls: &calls,
                reasoning_content: None,
            },
            text_message(Role::Tool, "first"),
            text_message(Role::Tool, "second"),
            text_message(Role::Assistant, "done"),
        ];
        let vl_request = Request {
            messages: &vl_messages,
            images: &[],
            videos: &[],
            options: RequestOptions {
                add_generation_prompt: true,
                ..RequestOptions::default()
            },
        };
        assert_eq!(
            render_chat(registry.get(ProfileAlias::Qwen3Vl8b), &vl_request, 0).expect("VL tools"),
            "<|im_start|>user\nweather?<|im_end|>\n<|im_start|>assistant\nchecking\n<tool_call>\n{\"name\": \"weather\", \"arguments\": {\"city\": \"Montréal\", \"units\": [\"c\", \"f\"], \"flag\": true, \"none\": null}}\n</tool_call><|im_end|>\n<|im_start|>user\n<tool_response>\nfirst\n</tool_response>\n<tool_response>\nsecond\n</tool_response><|im_end|>\n<|im_start|>assistant\ndone<|im_end|>\n<|im_start|>assistant\n"
        );

        let qwen35_messages = [
            text_message(Role::User, "weather?"),
            Message {
                role: Role::Assistant,
                content: MessageContent::Text("checking"),
                tool_calls: &calls,
                reasoning_content: None,
            },
            text_message(Role::Tool, "first"),
            text_message(Role::Tool, "second"),
            Message {
                role: Role::Assistant,
                content: MessageContent::Text("done"),
                tool_calls: &[],
                reasoning_content: Some(" compact hidden reasoning "),
            },
        ];
        let qwen35_request = Request {
            messages: &qwen35_messages,
            images: &[],
            videos: &[],
            options: RequestOptions {
                add_generation_prompt: true,
                enable_thinking: Some(false),
                ..RequestOptions::default()
            },
        };
        assert_eq!(
            render_chat(registry.get(ProfileAlias::Qwen35_9b), &qwen35_request, 0)
                .expect("3.5 tools"),
            "<|im_start|>user\nweather?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nchecking\n\n<tool_call>\n<function=weather>\n<parameter=city>\nMontréal\n</parameter>\n<parameter=units>\n[\"c\", \"f\"]\n</parameter>\n<parameter=flag>\nTrue\n</parameter>\n<parameter=none>\nNone\n</parameter>\n</function>\n</tool_call><|im_end|>\n<|im_start|>user\n<tool_response>\nfirst\n</tool_response>\n<tool_response>\nsecond\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\ncompact hidden reasoning\n</think>\n\ndone<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
    }

    #[test]
    fn visual_expansion_and_literal_placeholder_errors_are_always_on() {
        let registry = ProfileRegistry::bundled().expect("profiles");
        let items = [
            ContentItem::Image(ImageRef::default()),
            ContentItem::Text("X"),
            ContentItem::Video(VideoRef::default()),
        ];
        let messages = [Message {
            role: Role::User,
            content: MessageContent::Items(&items),
            tool_calls: &[],
            reasoning_content: None,
        }];
        let request = Request {
            messages: &messages,
            images: &[],
            videos: &[],
            options: RequestOptions {
                add_generation_prompt: true,
                add_vision_id: true,
                ..RequestOptions::default()
            },
        };
        let profile = registry.get(ProfileAlias::Qwen3Vl8b);
        let rendered = render_chat(profile, &request, 0).expect("render");
        let timestamps = [0.0];
        let visuals = [
            VisualExpansion::Image {
                input_index: 0,
                grid_thw: [1, 4, 4],
            },
            VisualExpansion::Video {
                input_index: 0,
                grid_thw: [1, 4, 4],
                timestamps: &timestamps,
            },
        ];
        let (expanded, replacements) =
            expand_visuals(profile, &rendered, &visuals, 0).expect("expansion");
        assert_eq!(
            expanded,
            "<|im_start|>user\nPicture 1: <|vision_start|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|vision_end|>XVideo 1: <|vision_start|><0.0 seconds><|vision_start|><|video_pad|><|video_pad|><|video_pad|><|video_pad|><|vision_end|><|vision_end|><|im_end|>\n<|im_start|>assistant\n"
        );
        assert_eq!(replacements.len(), 2);

        let error = expand_visuals(profile, "literal <|image_pad|>", &[], 0)
            .expect_err("literal placeholder must not masquerade as media");
        assert_eq!(error.category(), ErrorCategory::InvalidRequest);
    }

    #[test]
    fn qwen35_reasoning_extraction_and_thinking_rendering_are_always_on() {
        let registry = ProfileRegistry::bundled().expect("profiles");
        let messages = [
            text_message(Role::User, "hello"),
            text_message(Role::Assistant, "<think>\n internal \n</think>\nanswer"),
        ];
        let request = Request {
            messages: &messages,
            images: &[],
            videos: &[],
            options: RequestOptions {
                add_generation_prompt: true,
                ..RequestOptions::default()
            },
        };
        assert_eq!(
            render_chat(registry.get(ProfileAlias::Qwen35_9b), &request, 0)
                .expect("reasoning render"),
            "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n<think>\ninternal\n</think>\n\nanswer<|im_end|>\n<|im_start|>assistant\n<think>\n"
        );
        let request = Request {
            options: RequestOptions {
                add_generation_prompt: true,
                enable_thinking: Some(false),
                ..RequestOptions::default()
            },
            ..request
        };
        assert!(
            render_chat(registry.get(ProfileAlias::Qwen35_9b), &request, 0)
                .expect("disabled thinking")
                .ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n")
        );
    }

    #[test]
    #[ignore = "requires the hash-pinned local model snapshots under reference/.cache"]
    fn text_smoke_matches_both_committed_goldens() {
        let messages = [
            text_message(Role::System, "You are a concise visual assistant."),
            text_message(Role::User, "Reply with exactly: ready"),
        ];
        let request = Request {
            messages: &messages,
            images: &[],
            videos: &[],
            options: RequestOptions {
                add_generation_prompt: true,
                ..RequestOptions::default()
            },
        };
        let planned = [PlannedTextRequest {
            request,
            visuals: &[],
        }];

        {
            let processor = processor(ProfileAlias::Qwen3Vl8b);
            let output = processor
                .prepare_batch(&planned, ResourceLimits::default())
                .expect("Qwen3-VL text");
            assert_eq!(
                output.requests[0].rendered_prompt,
                "<|im_start|>system\nYou are a concise visual assistant.<|im_end|>\n<|im_start|>user\nReply with exactly: ready<|im_end|>\n<|im_start|>assistant\n"
            );
            assert_eq!(
                output.input_ids.as_slice(),
                [
                    151_644, 8948, 198, 2610, 525, 264, 63_594, 9124, 17_847, 13, 151_645, 198,
                    151_644, 872, 198, 20_841, 448, 6896, 25, 5527, 151_645, 198, 151_644, 77_091,
                    198
                ]
            );
        }

        {
            let processor = processor(ProfileAlias::Qwen35_9b);
            let output = processor
                .prepare_batch(&planned, ResourceLimits::default())
                .expect("Qwen3.5 text");
            assert_eq!(
                output.requests[0].rendered_prompt,
                "<|im_start|>system\nYou are a concise visual assistant.<|im_end|>\n<|im_start|>user\nReply with exactly: ready<|im_end|>\n<|im_start|>assistant\n<think>\n"
            );
            assert_eq!(
                output.input_ids.as_slice(),
                [
                    248_045, 8678, 198, 2523, 513, 264, 61_446, 8851, 17_313, 13, 248_046, 198,
                    248_045, 846, 198, 20_206, 440, 6681, 25, 5354, 248_046, 198, 248_045, 74_455,
                    198, 248_068, 198
                ]
            );
        }
    }

    #[test]
    #[ignore = "requires the hash-pinned local model snapshot under reference/.cache"]
    fn multimodal_smoke_matches_render_expansion_ids_and_ranges() {
        let processor = processor(ProfileAlias::Qwen3Vl8b);
        let image_bytes = [0_u8];
        let images = [ImageInput::Encoded {
            data: &image_bytes,
            format: ImageFormat::Jpeg,
        }];
        let items = [
            ContentItem::Image(ImageRef {
                input_index: 0,
                options: ImageOptions {
                    resized_height: Some(64),
                    resized_width: Some(64),
                    ..ImageOptions::default()
                },
            }),
            ContentItem::Text("Compare this image with the following image."),
        ];
        let messages = [Message {
            role: Role::User,
            content: MessageContent::Items(&items),
            tool_calls: &[],
            reasoning_content: None,
        }];
        let request = Request {
            messages: &messages,
            images: &images,
            videos: &[],
            options: RequestOptions {
                add_generation_prompt: true,
                add_vision_id: true,
                ..RequestOptions::default()
            },
        };
        let visuals = [VisualExpansion::Image {
            input_index: 0,
            grid_thw: [1, 4, 4],
        }];
        let output = processor
            .prepare_batch(
                &[PlannedTextRequest {
                    request,
                    visuals: &visuals,
                }],
                ResourceLimits::default(),
            )
            .expect("multimodal text");
        let prepared = &output.requests[0];
        assert_eq!(
            prepared.rendered_prompt,
            "<|im_start|>user\nPicture 1: <|vision_start|><|image_pad|><|vision_end|>Compare this image with the following image.<|im_end|>\n<|im_start|>assistant\n"
        );
        assert!(prepared.expanded_prompt.contains(&IMAGE_TOKEN.repeat(4)));
        assert_eq!(prepared.replacements.len(), 1);
        assert_eq!(prepared.replacements[0].modality, VisualModality::Image);
        assert_eq!(prepared.replacements[0].rendered_code_points.start, 44);
        assert_eq!(prepared.replacements[0].rendered_code_points.end, 57);
        assert_eq!(
            output
                .mm_token_type_ids
                .as_slice()
                .iter()
                .filter(|&&value| value == 1)
                .count(),
            4
        );
    }

    #[test]
    #[ignore = "requires the hash-pinned local model snapshots under reference/.cache"]
    #[allow(clippy::too_many_lines)]
    fn interleaved_image_video_expansion_matches_both_live_oracles() {
        let image_bytes = [0_u8];
        let images = [ImageInput::Encoded {
            data: &image_bytes,
            format: ImageFormat::Jpeg,
        }];
        let frame_bytes = [0_u8; 3];
        let frames = [Rgb8 {
            data: &frame_bytes,
            height: 1,
            width: 1,
            row_stride: 3,
        }];
        let videos = [VideoInput { frames: &frames }];
        let items = [
            ContentItem::Image(ImageRef {
                input_index: 0,
                options: ImageOptions {
                    resized_height: Some(64),
                    resized_width: Some(64),
                    ..ImageOptions::default()
                },
            }),
            ContentItem::Text("X"),
            ContentItem::Video(VideoRef {
                input_index: 0,
                options: VideoOptions {
                    resized_height: Some(64),
                    resized_width: Some(64),
                    ..VideoOptions::default()
                },
            }),
        ];
        let messages = [Message {
            role: Role::User,
            content: MessageContent::Items(&items),
            tool_calls: &[],
            reasoning_content: None,
        }];
        let request = Request {
            messages: &messages,
            images: &images,
            videos: &videos,
            options: RequestOptions {
                add_generation_prompt: true,
                add_vision_id: true,
                ..RequestOptions::default()
            },
        };
        let timestamps = [0.0];
        let visuals = [
            VisualExpansion::Image {
                input_index: 0,
                grid_thw: [1, 4, 4],
            },
            VisualExpansion::Video {
                input_index: 0,
                grid_thw: [1, 4, 4],
                timestamps: &timestamps,
            },
        ];
        let rendered_suffix = "<|im_start|>user\nPicture 1: <|vision_start|><|image_pad|><|vision_end|>XVideo 1: <|vision_start|><|video_pad|><|vision_end|><|im_end|>\n<|im_start|>assistant\n";
        let expanded_middle = "<|im_start|>user\nPicture 1: <|vision_start|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|vision_end|>XVideo 1: <|vision_start|><0.0 seconds><|vision_start|><|video_pad|><|video_pad|><|video_pad|><|video_pad|><|vision_end|><|vision_end|><|im_end|>\n<|im_start|>assistant\n";

        for alias in [ProfileAlias::Qwen3Vl8b, ProfileAlias::Qwen35_9b] {
            let processor = processor(alias);
            let output = processor
                .prepare_batch(
                    &[PlannedTextRequest {
                        request,
                        visuals: &visuals,
                    }],
                    ResourceLimits::default(),
                )
                .expect("interleaved visual text");
            let thinking_suffix = if alias == ProfileAlias::Qwen35_9b {
                "<think>\n"
            } else {
                ""
            };
            assert_eq!(
                output.requests[0].rendered_prompt,
                format!("{rendered_suffix}{thinking_suffix}")
            );
            assert_eq!(
                output.requests[0].expanded_prompt,
                format!("{expanded_middle}{thinking_suffix}")
            );
            assert_eq!(
                output.requests[0]
                    .replacements
                    .iter()
                    .map(|replacement| replacement.modality)
                    .collect::<Vec<_>>(),
                [VisualModality::Image, VisualModality::Video]
            );
            assert_eq!(
                output
                    .mm_token_type_ids
                    .as_slice()
                    .iter()
                    .filter(|&&value| value == 1)
                    .count(),
                4
            );
            assert_eq!(
                output
                    .mm_token_type_ids
                    .as_slice()
                    .iter()
                    .filter(|&&value| value == 2)
                    .count(),
                4
            );
        }
    }

    #[test]
    #[ignore = "requires the hash-pinned local model snapshot under reference/.cache"]
    fn qwen35_extracts_reasoning_and_honors_thinking_toggle_exactly() {
        let processor = processor(ProfileAlias::Qwen35_9b);
        let messages = [
            text_message(Role::User, "hello"),
            text_message(Role::Assistant, "<think>\n internal \n</think>\nanswer"),
        ];
        let request = Request {
            messages: &messages,
            images: &[],
            videos: &[],
            options: RequestOptions {
                add_generation_prompt: true,
                ..RequestOptions::default()
            },
        };
        let output = processor
            .prepare_batch(
                &[PlannedTextRequest {
                    request,
                    visuals: &[],
                }],
                ResourceLimits::default(),
            )
            .expect("reasoning extraction");
        assert_eq!(
            output.requests[0].rendered_prompt,
            "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n<think>\ninternal\n</think>\n\nanswer<|im_end|>\n<|im_start|>assistant\n<think>\n"
        );
        assert_eq!(
            output.input_ids.as_slice(),
            [
                248_045, 846, 198, 14_556, 248_046, 198, 248_045, 74_455, 198, 248_068, 198,
                10_168, 198, 248_069, 271, 8944, 248_046, 198, 248_045, 74_455, 198, 248_068, 198
            ]
        );

        let request = Request {
            options: RequestOptions {
                add_generation_prompt: true,
                enable_thinking: Some(false),
                ..RequestOptions::default()
            },
            ..request
        };
        let output = processor
            .prepare_batch(
                &[PlannedTextRequest {
                    request,
                    visuals: &[],
                }],
                ResourceLimits::default(),
            )
            .expect("disabled thinking");
        assert!(
            output.requests[0]
                .rendered_prompt
                .ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n")
        );
    }

    #[test]
    #[ignore = "requires the hash-pinned local model snapshot under reference/.cache"]
    fn right_padding_and_masks_use_profile_ids() {
        let processor = processor(ProfileAlias::Qwen3Vl8b);
        let short_messages = [text_message(Role::User, "")];
        let long_messages = [text_message(Role::User, "a longer Unicode prompt 👩🏽‍💻")];
        let short = Request {
            messages: &short_messages,
            images: &[],
            videos: &[],
            options: RequestOptions::default(),
        };
        let long = Request {
            messages: &long_messages,
            images: &[],
            videos: &[],
            options: RequestOptions::default(),
        };
        let output = processor
            .prepare_batch(
                &[
                    PlannedTextRequest {
                        request: short,
                        visuals: &[],
                    },
                    PlannedTextRequest {
                        request: long,
                        visuals: &[],
                    },
                ],
                ResourceLimits::default(),
            )
            .expect("padded batch");
        let columns = output.input_ids.columns();
        let short_mask = &output.attention_mask.as_slice()[..columns];
        let first_zero = short_mask
            .iter()
            .position(|&value| value == 0)
            .expect("padding");
        assert!(short_mask[..first_zero].iter().all(|&value| value == 1));
        assert!(short_mask[first_zero..].iter().all(|&value| value == 0));
        assert!(
            output.input_ids.as_slice()[first_zero..columns]
                .iter()
                .all(|&value| value == 151_643)
        );
        assert_eq!(
            output.input_ids.byte_strides().expect("strides"),
            [(columns * 8) as u64, 8]
        );
    }

    #[test]
    #[ignore = "requires the hash-pinned local model snapshot under reference/.cache"]
    fn literal_visual_tokens_and_plan_mismatches_are_rejected() {
        let processor = processor(ProfileAlias::Qwen3Vl8b);
        let messages = [text_message(Role::User, "literal <|image_pad|>")];
        let request = Request {
            messages: &messages,
            images: &[],
            videos: &[],
            options: RequestOptions::default(),
        };
        let error = processor
            .prepare_batch(
                &[PlannedTextRequest {
                    request,
                    visuals: &[],
                }],
                ResourceLimits::default(),
            )
            .expect_err("literal placeholder");
        assert_eq!(error.category(), ErrorCategory::InvalidRequest);
    }

    #[test]
    fn changed_or_missing_assets_are_profile_mismatches() {
        let registry = ProfileRegistry::bundled().expect("profiles");
        let error = TextProcessor::from_local_assets(
            registry.get(ProfileAlias::Qwen3Vl8b),
            Path::new("this-directory-does-not-exist"),
        )
        .err()
        .expect("local-only failure");
        assert_eq!(error.category(), ErrorCategory::ProfileMismatch);

        let directory =
            std::env::temp_dir().join(format!("qwen-mm-changed-tokenizer-{}", std::process::id()));
        fs::create_dir(&directory).expect("unique temporary directory");
        fs::write(directory.join("tokenizer.json"), b"{}").expect("temporary changed tokenizer");
        let error =
            TextProcessor::from_local_assets(registry.get(ProfileAlias::Qwen3Vl8b), &directory)
                .err()
                .expect("hash mismatch");
        fs::remove_dir_all(&directory).expect("temporary cleanup");
        assert_eq!(error.category(), ErrorCategory::ProfileMismatch);
        assert_eq!(error.context()["asset"].to_string(), "tokenizer.json");
    }

    #[test]
    #[ignore = "requires the hash-pinned local model snapshot under reference/.cache"]
    fn rendered_token_limits_fail_with_the_stable_category() {
        let processor = processor(ProfileAlias::Qwen3Vl8b);
        let messages = [text_message(Role::User, "two tokens are already too many")];
        let request = Request {
            messages: &messages,
            images: &[],
            videos: &[],
            options: RequestOptions::default(),
        };
        let limits = ResourceLimits::default()
            .lowered(LimitOverrides {
                rendered_tokens_per_request: Some(1),
                ..LimitOverrides::default()
            })
            .expect("lowered limit");
        let error = processor
            .prepare_batch(
                &[PlannedTextRequest {
                    request,
                    visuals: &[],
                }],
                limits,
            )
            .expect_err("token limit");
        assert_eq!(error.category(), ErrorCategory::ResourceLimit);
    }

    #[test]
    fn tools_are_rendered_without_jinja_or_python() {
        let messages = [text_message(Role::User, "weather?")];
        let tools = [ToolDefinition {
            json: r#"{"type":"function","function":{"name":"weather","parameters":{"type":"object"}}}"#,
        }];
        let request = Request {
            messages: &messages,
            images: &[],
            videos: &[],
            options: RequestOptions {
                tools: &tools,
                ..RequestOptions::default()
            },
        };
        let registry = ProfileRegistry::bundled().expect("profiles");
        let rendered =
            render_chat(registry.get(ProfileAlias::Qwen3Vl8b), &request, 0).expect("tool render");
        assert!(rendered.contains(
            r#"{"type": "function", "function": {"name": "weather", "parameters": {"type": "object"}}}"#
        ));
    }
}

#[cfg(test)]
#[path = "text_conformance_tests.rs"]
mod conformance_tests;
