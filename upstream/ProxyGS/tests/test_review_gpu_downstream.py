"""Offline fail-closed artifact review tests, without Torch/CUDA dependencies."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

MODULE = Path(__file__).resolve().parents[1] / 'tools' / 'review_gpu_downstream.py'
SPEC = importlib.util.spec_from_file_location('review_gpu_downstream', MODULE)
reviewer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reviewer)


class ReviewerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.legacy = self.root / 'legacy.json'
        self.legacy.write_text('{"status":"qualified"}')
        cameras = [f'camera{i}' for i in range(32)]
        for scene in reviewer.SCENES:
            folder = self.root / scene / 'formal'
            folder.mkdir(parents=True)
            contract = {'formal': True, 'camera_ids': cameras, 'modes': reviewer.MODES,
                        'stream_policy': 'default_stream_serialized_gpu_stages', 'overlap_claim': False,
                        'cache_policy': {'refresh_period': 2, 'max_source_age': 1, 'capacity_rows': 6826846, 'payload_dtype': 'float32'}}
            qualification, performance = {}, {}
            for mode in reviewer.MODES:
                record = {'mode': mode, 'scene': scene, 'frame_count': 32, 'wall_ms': 100.,
                          'selection_oracle_exact': True, 'max_resident_rows': 3, 'timing_role': 'diagnostic_only',
                          'selection_records': [{'frame': i, 'camera': cameras[i], 'oracle_exact': True,
                                                 'depth_d2h_bytes': 0, 'selected_ids_h2d_bytes': 0} for i in range(32)],
                          'deadline_records': [{'pair': i, 'late': False} for i in range(16)],
                          'deadline_misses': 0, 'fallback_pairs': [],
                          'schedule_lead_pairs': 3 if mode == 'cpu_scheduled_cache_q2_mesh32' else (2 if 'q2' in mode else (1 if 'q1' in mode else 0))}
                if mode != 'cpu_serial_fresh_mesh32':
                    record['quality'] = {'records': [{'frame': i, 'camera': cameras[i], 'metrics': {'reuse_minus_fresh_gt': {'psnr': 0., 'ssim': 0., 'lpips': 0.}}, 'depth': {'relative_mae': 0.}} for i in range(32)]}
                if mode == 'gpu_serial_fresh_mesh32':
                    record['fresh_exact'] = {'passed': True, 'records': [{'frame': i, 'camera': cameras[i], 'image': True, 'render_depth': True, 'render_alpha': True} for i in range(32)]}
                qualification[mode] = record
                performance[mode] = [dict(copy.deepcopy(record), timing_role='performance') for _ in range(3)]
            summary = {'status': 'pass', 'frame_count': 32, 'qualification': qualification, 'performance_median_wall_ms': {m: 100. for m in reviewer.MODES}}
            for name, value in [('contract', contract), ('summary', summary), ('performance', performance), ('status', {'state': 'complete', 'status': 'pass'})]:
                (folder / f'{name}.json').write_text(json.dumps(value))

    def result(self):
        return reviewer.review(self.root, 'formal', self.legacy)

    def mutate(self, filename, callback):
        path = self.root / 'rome' / 'formal' / f'{filename}.json'
        value = json.loads(path.read_text())
        callback(value)
        path.write_text(json.dumps(value))

    def assert_rejected(self, substring):
        result = self.result()
        self.assertEqual(result['status'], 'failed')
        self.assertTrue(any(substring in f for f in result['failures']), result['failures'])

    def test_complete_matrix(self):
        result = self.result()
        self.assertEqual(result['status'], 'pass', result['failures'])
        self.assertEqual(result['paired_wall_ms']['gpu_serial_fresh_mesh32'], 800.)

    def test_qualification_scope_cannot_claim_full_matrix(self):
        result = reviewer.review(self.root, 'formal', self.legacy, scene_names=('amsterdam',))
        self.assertEqual(result['status'], 'pass', result['failures'])
        self.assertEqual(result['scope'], 'qualification_subset')
        self.assertEqual(result['expected_scene_count'], 1)
        self.assertEqual(result['expected_frames_per_mode_repeat'], 32)

    def test_missing_frame(self):
        self.mutate('performance', lambda v: v['gpu_serial_cache_mesh32'][0]['selection_records'].pop())
        self.assert_rejected('frame order')

    def test_missing_mode(self):
        self.mutate('performance', lambda v: v.pop('gpu_scheduled_cache_q1_mesh32'))
        self.assert_rejected('performance repeats')

    def test_missing_repeat(self):
        self.mutate('performance', lambda v: v['gpu_serial_cache_mesh32'].pop())
        self.assert_rejected('performance repeats')

    def test_bad_quality(self):
        self.mutate('summary', lambda v: v['qualification']['gpu_serial_cache_mesh32']['quality']['records'][0]['depth'].update(relative_mae=.06))
        self.assert_rejected('depth quality failed')

    def test_fresh_exact_failure_cannot_use_cache_tolerance(self):
        self.mutate('summary', lambda v: v['qualification']['gpu_serial_fresh_mesh32']['fresh_exact']['records'][0].update(image=False))
        self.assert_rejected('not exactly equal')

    def test_fresh_exact_missing(self):
        self.mutate('summary', lambda v: v['qualification']['gpu_serial_fresh_mesh32'].pop('fresh_exact'))
        self.assert_rejected('fresh exact gate missing')

    def test_gpu_q2_must_not_inherit_historical_lead3(self):
        self.mutate('performance', lambda v: v['gpu_scheduled_cache_q2_mesh32'][0].update(schedule_lead_pairs=3))
        self.assert_rejected('schedule lead policy differs')

    def test_diagnostic_timing_cannot_be_performance(self):
        self.mutate('performance', lambda v: v['gpu_serial_fresh_mesh32'][0].update(timing_role='diagnostic_only'))
        self.assert_rejected('timing role mismatch')

    def test_transfer_evidence_missing(self):
        self.mutate('performance', lambda v: v['gpu_serial_fresh_mesh32'][0]['selection_records'][0].pop('depth_d2h_bytes'))
        self.assert_rejected('transfer evidence')


if __name__ == '__main__':
    unittest.main(verbosity=2)
