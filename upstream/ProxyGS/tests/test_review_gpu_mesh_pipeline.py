"""Offline mesh-pipeline artifact acceptance and evidence failures."""
import importlib.util
import json
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

fixture = load('v1_review_fixture_for_pipeline', HERE / 'test_review_gpu_downstream.py')
reviewer = load('pipeline_reviewer', HERE.parent / 'tools/review_gpu_mesh_pipeline.py')
fixture.reviewer = reviewer

class MeshReviewerTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixture.ReviewerTests('test_complete_matrix')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root, self.legacy = self.fixture.root, self.fixture.legacy
        for scene in reviewer.SCENES:
            contract_path = self.root / scene / 'formal/contract.json'
            contract = json.loads(contract_path.read_text())
            contract['protocol'] = 'proxygs-gpu-cpu-mesh-pipeline-v2'
            contract['retained_profile'] = {'workers': 24 if scene == 'barcelona' else 32}
            contract['pipeline_modes'] = {'gpu_cpu_mesh_pipeline_window_retained': {'future_camera_scope': 'frozen_known_32_camera_window'}}
            contract_path.write_text(json.dumps(contract))
            for filename in ('summary', 'performance'):
                path = self.root / scene / 'formal' / f'{filename}.json'
                data = json.loads(path.read_text())
                modes = data['qualification'] if filename == 'summary' else data
                for mode in reviewer.MODES:
                    if not mode.startswith('gpu_cpu_mesh_pipeline_'):
                        continue
                    q, workers = ((15, contract['retained_profile']['workers']) if mode == 'gpu_cpu_mesh_pipeline_window_retained' else ((1, 2) if 'q1_pool2' in mode else (2, 4)))
                    for record in ([modes[mode]] if filename == 'summary' else modes[mode]):
                        for selection in record['selection_records']:
                            selection.update(mesh_workers=workers, mesh_native_query_threads=1)
                        record['mesh_pipeline'] = {'lookahead_pairs': q, 'pool_workers': workers,
                            'actual_mesh_workers': workers, 'native_query_threads': 1, 'startup_mesh_wait_ms': 1., 'total_mesh_wait_ms': 16.,
                            'cpu_mesh_gpu_stage_host_overlap_ms': 5.,
                            'wait_records': [{'pair': p, 'wait_ms': 1., 'submitted_pairs': list(range(min(p + q + 1, 16))), 'pending_pairs': list(range(p, min(p + q + 1, 16))), 'lookahead_pairs': q, 'pool_workers': workers} for p in range(16)]}
                        if mode == 'gpu_cpu_mesh_pipeline_window_retained':
                            record['mesh_pipeline'].update(future_camera_scope='frozen_known_32_camera_window', worker_policy='frozen_retained_profile', submission_policy='online_full_window_pairwise_consumption')
                path.write_text(json.dumps(data))

    def check_rejected(self, field, value, expected):
        path = self.root / 'rome/formal/performance.json'
        data = json.loads(path.read_text())
        data['gpu_cpu_mesh_pipeline_q1_pool2'][0]['mesh_pipeline'][field] = value
        path.write_text(json.dumps(data))
        result = reviewer.review(self.root, 'formal', self.legacy)
        self.assertEqual(result['status'], 'failed')
        self.assertTrue(any(expected in f for f in result['failures']), result['failures'])

    def test_complete_pipeline_matrix(self):
        result = reviewer.review(self.root, 'formal', self.legacy)
        self.assertEqual(result['status'], 'pass', result['failures'])
        self.assertEqual(result['expected_frames_per_mode_repeat'], 256)

    def test_missing_wait_records(self):
        self.check_rejected('wait_records', [], 'mesh wait denominator')

    def test_changed_pool(self):
        self.check_rejected('pool_workers', 32, 'mesh pool or lead')

    def test_known_window_scope_required(self):
        path = self.root / 'rome/formal/contract.json'
        data = json.loads(path.read_text())
        data['pipeline_modes']['gpu_cpu_mesh_pipeline_window_retained'].pop('future_camera_scope')
        path.write_text(json.dumps(data))
        result = reviewer.review(self.root, 'formal', self.legacy)
        self.assertEqual(result['status'], 'failed')
        self.assertTrue(any('future camera scope' in f for f in result['failures']))

    def test_retained_worker_profile_required(self):
        path = self.root / 'barcelona/formal/contract.json'
        data = json.loads(path.read_text())
        data['retained_profile']['workers'] = 32
        path.write_text(json.dumps(data))
        result = reviewer.review(self.root, 'formal', self.legacy)
        self.assertEqual(result['status'], 'failed')
        self.assertTrue(any('retained worker profile' in f for f in result['failures']))

    def test_native_threads_changed(self):
        self.check_rejected('native_query_threads', 4, 'actual mesh worker evidence')

    def test_missing_overlap_evidence(self):
        self.check_rejected('cpu_mesh_gpu_stage_host_overlap_ms', None, 'absent/invalid')

    def test_wait_sum_mismatch(self):
        self.check_rejected('total_mesh_wait_ms', 2., 'mesh wait sum mismatch')

    def test_unbounded_queue(self):
        records = [{'pair': p, 'wait_ms': 1., 'submitted_pairs': list(range(16)), 'pending_pairs': list(range(p,16)), 'lookahead_pairs': 1, 'pool_workers': 2} for p in range(16)]
        self.check_rejected('wait_records', records, 'mesh pending queue bound')

if __name__ == '__main__':
    unittest.main(verbosity=2)
