"""Tests for Ryan surface iteration finalize (no collaborator subprocess)."""

import json
import os
import sys
import tempfile
import unittest

import numpy as np

CODES = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if CODES not in sys.path:
	sys.path.insert(0, CODES)
import ryan_v2_finalize_iter_surface as rv


class TestFinalizeIter(unittest.TestCase):
	def test_resolve_uses_iteration_log_when_predictions_exist(self):
		with tempfile.TemporaryDirectory() as td:
			for kk in (0, 1, 2):
				os.makedirs(os.path.join(td, "iter_%02d" % kk), exist_ok=True)
				np.savez_compressed(
					os.path.join(td, "iter_%02d" % kk, "predictions.npz"),
					X_fill=np.zeros((10, 2)),
					mu=np.zeros(10),
					std=np.ones(10),
				)
			with open(rv.iteration_log_path(td), "w", encoding="utf-8") as f:
				f.write(json.dumps({"iteration": 2, "gp_tag": "coco_k02"}) + "\n")
			self.assertEqual(rv.resolve_last_iteration_index(td), 2)

	def test_resolve_falls_back_when_log_points_missing_npz(self):
		with tempfile.TemporaryDirectory() as td:
			os.makedirs(os.path.join(td, "iter_07"), exist_ok=True)
			np.savez_compressed(
				os.path.join(td, "iter_07", "predictions.npz"),
				X_fill=np.zeros((5, 2)),
				mu=np.arange(5, dtype=float),
				std=np.ones(5),
			)
			with open(rv.iteration_log_path(td), "w", encoding="utf-8") as f:
				f.write(json.dumps({"iteration": 99}) + "\n")
			idx = rv.resolve_last_iteration_index(td)
			self.assertEqual(idx, 7)

	def test_load_final_surface_arrays(self):
		with tempfile.TemporaryDirectory() as td:
			os.makedirs(os.path.join(td, "iter_01"), exist_ok=True)
			X_fill = np.array([[1.0, 2.0], [3.0, -1.5]], dtype=np.float64)
			np.savez_compressed(
				os.path.join(td, "iter_01", "predictions.npz"),
				X_fill=X_fill,
				mu=np.array([10.0, 20.0]),
				mu_raw=np.array([11.0, 21.0]),
				std=np.array([1.5, 2.5]),
			)
			with open(rv.iteration_log_path(td), "w", encoding="utf-8") as f:
				f.write(json.dumps({"iteration": 1}) + "\n")
			k, x1, x2, mu, std, rec = rv.load_final_surface_arrays(td, mu_key="mu_raw", std_key="std")
			self.assertEqual(k, 1)
			self.assertEqual(rec["iteration"], 1)
			np.testing.assert_array_almost_equal(mu, np.array([11.0, 21.0]))
			np.testing.assert_array_almost_equal(x1, X_fill[:, 0])
			np.testing.assert_array_almost_equal(x2, X_fill[:, 1])

	def test_default_workspace(self):
		p = rv.default_ryan_surface_workspace(os.path.join("Out", "AT2017gfo"))
		self.assertEqual(p[-len("ryan_surface_iterations") :], "ryan_surface_iterations")

	def test_load_pinned_iteration_prefers_that_iter(self):
		with tempfile.TemporaryDirectory() as td:
			os.makedirs(os.path.join(td, "iter_00"), exist_ok=True)
			os.makedirs(os.path.join(td, "iter_07"), exist_ok=True)
			np.savez_compressed(
				os.path.join(td, "iter_00", "predictions.npz"),
				X_fill=np.zeros((3, 2)),
				mu=np.ones(3),
				std=np.ones(3),
			)
			np.savez_compressed(
				os.path.join(td, "iter_07", "predictions.npz"),
				X_fill=np.zeros((5, 2)),
				mu=np.arange(5, dtype=float),
				std=np.ones(5),
			)
			with open(rv.iteration_log_path(td), "w", encoding="utf-8") as f:
				f.write(json.dumps({"iteration": 0, "gp_tag": "k00"}) + "\n")

			k, _x1, _x2, mu, _std, rec = rv.load_final_surface_arrays(td, iteration=7)
			self.assertEqual(k, 7)
			np.testing.assert_array_almost_equal(mu, np.arange(5, dtype=float))
			self.assertIsNone(rec)

	def test_load_pinned_iteration_matching_log_record(self):
		with tempfile.TemporaryDirectory() as td:
			os.makedirs(os.path.join(td, "iter_02"), exist_ok=True)
			X_fill = np.array([[1.0, 2.0], [3.0, -1.5]], dtype=np.float64)
			np.savez_compressed(
				os.path.join(td, "iter_02", "predictions.npz"),
				X_fill=X_fill,
				mu=np.array([1.0, 2.0]),
				std=np.array([0.1, 0.2]),
				std_raw=np.array([0.2, 0.3]),
			)
			with open(rv.iteration_log_path(td), "w", encoding="utf-8") as f:
				f.write(json.dumps({"iteration": 2, "gp_tag": "surf_k02"}) + "\n")

			_k, _, _, _, std, rec = rv.load_final_surface_arrays(td, iteration=2, std_key="std_raw")
			np.testing.assert_array_almost_equal(std, np.array([0.2, 0.3]))
			self.assertIsNotNone(rec)
			self.assertEqual(rec.get("gp_tag"), "surf_k02")

	def test_iteration_record_for_index(self):
		with tempfile.TemporaryDirectory() as td:
			p = rv.iteration_log_path(td)
			with open(p, "w", encoding="utf-8") as f:
				f.write(json.dumps({"iteration": 0, "tag": "a"}) + "\n")
				f.write(json.dumps({"iteration": 5, "tag": "mid"}) + "\n")
				f.write(json.dumps({"iteration": 5, "tag": "five_b"}) + "\n")
			r = rv.iteration_record_for_index(td, 5)
			self.assertEqual(r.get("tag"), "five_b")


if __name__ == "__main__":
	unittest.main()
