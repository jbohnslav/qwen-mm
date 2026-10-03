"""Independent fixture/oracle checks; public native parity runs video_oracle.py."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

SCRIPTS = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("video_oracle", SCRIPTS / "video_oracle.py")
assert SPEC is not None and SPEC.loader is not None
oracle = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = oracle
SPEC.loader.exec_module(oracle)


class VideoOracleTests(unittest.TestCase):
    def test_fixture_distinguishes_temporal_channels_and_axes(self) -> None:
        first = oracle.make_frames(5, 128, 192)
        np.testing.assert_array_equal(first, oracle.make_frames(5, 128, 192))
        self.assertEqual(first.dtype, np.uint8)
        self.assertEqual(first.shape, (5, 128, 192, 3))
        self.assertFalse(np.array_equal(first[0], first[1]))
        self.assertFalse(np.array_equal(first[..., 0], first[..., 1]))
        self.assertFalse(np.array_equal(first[:, 20, 30], first[:, 30, 20]))

    def test_list_oracle_preserves_legacy_padding_and_double_resize(self) -> None:
        from PIL import Image

        frames = [Image.fromarray(frame) for frame in oracle.make_frames(3, 63, 95)]
        data, metadata, route = oracle.reference_video(frames, {})
        self.assertEqual(route, "qwen-vl-utils-list")
        self.assertEqual(tuple(data.shape), (4, 3, 320, 448))
        self.assertEqual(metadata["frames_indices"], [0, 1, 2, 3])
        self.assertEqual(metadata["fps"], 2)
        np.testing.assert_array_equal(data[-1].numpy(), data[-2].numpy())

    def test_decoded_odd_clip_preserves_original_metadata_before_padding(self) -> None:
        frames = oracle.make_frames(3)
        metadata = {"fps": 29.97, "frames_indices": [7, 15, 31], "total_num_frames": 90}
        data, expected, _ = oracle.reference_video(
            (torch.from_numpy(frames).permute(0, 3, 1, 2), metadata),
            {"min_pixels": 128 * 192, "max_pixels": 128 * 192},
        )
        self.assertEqual(len(data), 3)
        self.assertEqual(expected, metadata)
        self.assertIsNot(expected, metadata)
        self.assertEqual(metadata["frames_indices"], [7, 15, 31])

    def test_ready_qwen_tuple_has_no_second_resize(self) -> None:
        data = torch.from_numpy(oracle.make_frames(4)).permute(0, 3, 1, 2).float()
        metadata = {"fps": 12, "frames_indices": [0, 5, 11, 17], "total_num_frames": 18}
        result, info, route = oracle.reference_video((data, metadata), {})
        self.assertEqual(route, "transformers-presampled")
        self.assertEqual(result.data_ptr(), data.data_ptr())
        self.assertEqual(info, metadata)

    def test_temporal_layout_witness_recovers_each_official_frame(self) -> None:
        native, official = oracle.load_processors(
            "qwen3-vl-8b", oracle.ROOT / "reference/.cache/huggingface", 1
        )
        del native
        rgb = oracle.make_frames(3)
        outputs = official.video_processor(
            videos=[torch.from_numpy(rgb).permute(0, 3, 1, 2)],
            do_sample_frames=False,
            do_resize=False,
            return_tensors="np",
        )
        represented = oracle.unpatchify_video(
            outputs["pixel_values_videos"], outputs["video_grid_thw"][0]
        )
        expected = np.concatenate((rgb, rgb[-1:]), axis=0).transpose(0, 3, 1, 2)
        np.testing.assert_array_equal(represented, expected)
        np.testing.assert_allclose(
            oracle.canonical_video_frames(represented),
            outputs["pixel_values_videos"],
            rtol=0,
            atol=oracle.VIDEO_ATOL,
        )

    def test_generated_cases_cover_both_temporal_and_batch_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cases = oracle.cases(Path(temporary), include_files=False)
        names = {case.name for case in cases}
        self.assertEqual(len(names), len(cases))
        self.assertTrue(
            {
                "pillow_single",
                "pillow_odd",
                "numpy_single",
                "framebatch_pts",
                "decoded_total_pixel_budget",
                "mixed_media_order",
                "indexed_repeated_video",
                "qwen_utils_prepared_tuple",
                "heterogeneous_left_padding",
                "heterogeneous_right_padding",
                "torch_contiguous_spatial_resize",
                "decoded_single_spatial_resize",
            }
            <= names
        )

    def test_rounding_regression_cases_preserve_source_geometry_and_torch_layout(self) -> None:
        if importlib.util.find_spec("av") is None:
            self.skipTest("optional PyAV unavailable")
        with tempfile.TemporaryDirectory() as temporary:
            generated = {
                case.name: case for case in oracle.cases(Path(temporary), include_files=True)
            }
            for name, count in (
                ("decoded_spatial_resize", 4),
                ("torch_contiguous_spatial_resize", 4),
                ("decoded_single_spatial_resize", 1),
                ("mp4_spatial_resize", 4),
            ):
                with self.subTest(case=name):
                    item = generated[name].requests[0]["messages"][0]["content"][0]
                    self.assertEqual((item["resized_height"], item["resized_width"]), (96, 160))
                    source = item["video"]
                    if name == "mp4_spatial_resize":
                        source, _ = oracle.decode_mp4(source)
                        np.testing.assert_array_equal(
                            source.numpy(),
                            oracle.make_frames(count, 177, 259).transpose(0, 3, 1, 2),
                        )
                    elif name == "torch_contiguous_spatial_resize":
                        self.assertTrue(source.is_contiguous())
                    data, _, _ = oracle.reference_video(source, item)
                    self.assertEqual(tuple(data.shape), (count, 3, 96, 160))

    def test_mp4_fixture_is_lossless_and_clipping_preserves_absolute_indices(self) -> None:
        if importlib.util.find_spec("av") is None:
            self.skipTest("optional PyAV unavailable")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "clip with spaces.mp4"
            rgb = oracle.make_frames(36)
            oracle.write_mp4(path, rgb, fps=12)
            data, info = oracle.decode_mp4(path)
            np.testing.assert_array_equal(data.numpy(), rgb.transpose(0, 3, 1, 2))
            self.assertEqual(info, {"fps": 12, "total_num_frames": 36})
            selected, metadata, route = oracle.reference_video(
                path.as_uri(),
                {
                    "video_start": 0.5,
                    "video_end": 2.0,
                    "nframes": 6,
                    "min_pixels": 128 * 192,
                    "max_pixels": 128 * 192,
                },
            )
            self.assertEqual(route, "pyav+pinned-qwen-sampling")
            self.assertEqual(metadata["frames_indices"], [6, 10, 13, 17, 20, 24])
            np.testing.assert_array_equal(
                selected.numpy(), data[metadata["frames_indices"]].numpy()
            )

    def test_long_clip_sampling_matches_pinned_torch_float32_linspace(self) -> None:
        from qwen_mm._video import _sample_indices

        scenarios = (
            (15001, 30.0, {}),
            (180000, 30.0, {}),
            (1000001, 29.97, {}),
            (180000, 30.0, {"nframes": 768, "video_start": 2.3, "video_end": 900.4}),
            (1000001, 29.97, {"nframes": 768, "video_start": 1440.04, "video_end": 3000.2}),
        )
        for total, fps, options in scenarios:
            with self.subTest(total=total, fps=fps, options=options):
                start, end, count = oracle.calculate_video_frame_range(options, total, fps)
                wanted_count = oracle.smart_nframes(options, count, fps)
                expected = (
                    torch.linspace(start, end, wanted_count, dtype=torch.float32)
                    .round()
                    .long()
                    .tolist()
                )
                self.assertEqual(_sample_indices(total, fps, options, {}), expected)


if __name__ == "__main__":
    unittest.main()
