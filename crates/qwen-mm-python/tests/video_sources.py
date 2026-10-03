"""Source adapter checks without model downloads or a video decoder dependency."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from qwen_mm import InvalidRequestError, MediaGeometryError, ResourceLimitError
from qwen_mm._media import normalize_requests
from qwen_mm._video import _sample_indices


class VideoSources(unittest.TestCase):
    def setUp(self):
        self.clip = np.arange(3 * 32 * 64 * 3, dtype=np.uint8).reshape(3, 32, 64, 3)

    def normalize(self, source, *, limits=None, **options):
        return normalize_requests(
            [
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "video", "video": source, **options},
                            ],
                        }
                    ]
                }
            ],
            limits=limits or {},
        )[0]

    def test_decoded_array_is_preserved_and_not_resampled(self):
        request = self.normalize(self.clip, sample_fps=3.5)
        np.testing.assert_array_equal(np.stack(request["videos"][0]["frames"]), self.clip)
        self.assertNotIn("image_frames", request["videos"][0])
        self.assertEqual(request["messages"][0]["content"][0]["options"]["sample_fps"], 3.5)
        with self.assertRaises(InvalidRequestError):
            self.normalize(self.clip, fps=2)

    def test_image_lists_and_decoded_descriptors_have_distinct_paths(self):
        request = self.normalize(list(self.clip))
        self.assertTrue(request["videos"][0]["image_frames"])
        request = self.normalize({"frames": list(self.clip), "image_frames": False})
        self.assertNotIn("image_frames", request["videos"][0])

    def test_prepared_utils_float_tuple_preserves_metadata(self):
        metadata = {"fps": 30, "frames_indices": [0, 7, 12], "total_num_frames": 30}
        request = self.normalize((self.clip.astype(np.float32), metadata))
        video = request["videos"][0]
        self.assertEqual(video["frames_indices"], [0, 7, 12])
        self.assertEqual(video["fps"], 30)
        self.assertNotIn("image_frames", video)

    def test_frame_batch_preserves_pts(self):
        batch = SimpleNamespace(data=self.clip, pts_seconds=np.array([0.1, 0.8, 1.25]))
        self.assertEqual(self.normalize(batch)["videos"][0]["timestamps"], [0.1, 0.8, 1.25])

    def test_decoder_requests_only_selected_frames(self):
        requests = []
        clip = self.clip

        class Decoder:
            metadata = SimpleNamespace(average_fps=30, num_frames=30, height=32, width=64)

            def get_frames_at(self, *, indices):
                requests.append(indices)
                return SimpleNamespace(data=np.stack([clip[0]] * len(indices)))

        result = self.normalize(Decoder(), nframes=4)
        self.assertEqual(requests, [[0, 10, 19, 29]])
        self.assertEqual(result["videos"][0]["frames_indices"], requests[0])

    def test_limits_precede_decoder_retrieval(self):
        class Decoder:
            metadata = SimpleNamespace(average_fps=30, num_frames=30, height=32, width=64)

            def get_frames_at(self, *, indices):
                raise AssertionError("over-limit video must not decode")

        with self.assertRaises(ResourceLimitError):
            self.normalize(Decoder(), nframes=4, limits={"raw_frames_per_video": 2})
        with self.assertRaises(ResourceLimitError):
            self.normalize(Decoder(), nframes=4, limits={"decoded_edge_length": 16})

    def test_frame_ceiling_precedes_tensor_host_copy(self):
        class OverLimitTensor:
            shape = (769, 3, 32, 32)

            def detach(self):
                raise AssertionError("over-limit tensor must not be copied")

            def cpu(self):
                raise AssertionError("over-limit tensor must not be copied")

            def numpy(self):
                raise AssertionError("over-limit tensor must not be copied")

        with self.assertRaises(ResourceLimitError):
            self.normalize(OverLimitTensor(), limits={"raw_frames_per_video": 2})

    def test_unknown_file_frame_count_checks_dimensions_before_scan(self):
        stream = SimpleNamespace(
            average_rate=12, height=64, width=64, frames=0, codec_context=SimpleNamespace()
        )

        class Container:
            streams = SimpleNamespace(video=[stream])

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def decode(self, *args):
                raise AssertionError(
                    "over-limit file dimensions must not trigger a frame-count scan"
                )

        with patch.dict("sys.modules", {"av": SimpleNamespace(open=lambda source: Container())}):
            with self.assertRaises(ResourceLimitError):
                self.normalize(b"encoded", video_backend="pyav", limits={"decoded_edge_length": 16})

    def test_timestamp_and_dtype_validation(self):
        for times in ([0, 2, 1], [0, float("nan"), 2], [0, 1], [0, True, 2]):
            with self.subTest(times=times), self.assertRaises(MediaGeometryError):
                self.normalize({"frames": self.clip, "timestamps": times})
        for rate in (0, -1, True, float("inf")):
            with self.subTest(rate=rate), self.assertRaises(MediaGeometryError):
                self.normalize(self.clip, raw_fps=rate)

    def test_source_and_nested_options_are_not_silently_dropped(self):
        with self.assertRaises(InvalidRequestError):
            self.normalize(self.clip, made_up=True)
        with self.assertRaises(InvalidRequestError):
            self.normalize(self.clip, raw_fps=2, options={"raw_fps": 3})
        with self.assertRaises(InvalidRequestError):
            self.normalize({"frames": self.clip, "metadata": []})

    def test_references_are_reused_and_unreferenced_sources_fail(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": 0},
                    {"type": "video", "video": 0},
                ],
            }
        ]
        request = normalize_requests([{"messages": messages, "videos": [self.clip]}], limits={})[0]
        self.assertEqual(len(request["videos"]), 1)
        self.assertEqual([i["input_index"] for i in request["messages"][0]["content"]], [0, 0])
        with self.assertRaises(InvalidRequestError):
            normalize_requests(
                [{"messages": messages, "videos": [self.clip, self.clip]}], limits={}
            )

    def test_pinned_sampling_and_clips(self):
        self.assertEqual(_sample_indices(30, 30, {"nframes": 4}, {}), [0, 10, 19, 29])
        self.assertEqual(
            _sample_indices(30, 30, {"video_start": 0.2, "video_end": 0.6, "nframes": 4}, {}),
            [6, 10, 14, 18],
        )
        self.assertEqual(_sample_indices(1, 30, {}, {}), [0])
        with self.assertRaises(InvalidRequestError):
            _sample_indices(30, 30, {"nframes": 4, "fps": 2}, {})

    def test_reference_cache_preserves_timestamp_precision_and_option_types(self):
        first = np.array([0.0, 0.3 - 1e-10, 0.7])
        second = np.array([0.0, 0.3 + 1e-10, 0.7])
        self.assertEqual(repr(first), repr(second))
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": 0, "timestamps": first},
                    {"type": "video", "video": 0, "timestamps": second},
                ],
            }
        ]
        request = normalize_requests([{"messages": messages, "videos": [self.clip]}], limits={})[0]
        self.assertEqual(len(request["videos"]), 2)
        self.assertNotEqual(request["videos"][0]["timestamps"], request["videos"][1]["timestamps"])
        messages[0]["content"] = [
            {"type": "video", "video": 0, "raw_fps": 1},
            {"type": "video", "video": 0, "raw_fps": True},
        ]
        with self.assertRaises(MediaGeometryError):
            normalize_requests([{"messages": messages, "videos": [self.clip]}], limits={})


if __name__ == "__main__":
    unittest.main()
