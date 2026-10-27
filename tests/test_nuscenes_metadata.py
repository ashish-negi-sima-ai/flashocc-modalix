import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from nuscenes_metadata import records, select_samples, load_selected


class MetadataTests(unittest.TestCase):
    def test_stream_boundaries_and_invalid_input(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'table.json'
            values = [{'token': 'a', 'value': 'é,[]"' * 20}, {'token': 'b', 'number': 12345}]
            path.write_text(json.dumps(values))
            self.assertEqual(list(records(path, chunk_size=3)), values)
            for invalid in ['[{"x":1}', '[{"x":1},]', '[{"x":1}] extra']:
                path.write_text(invalid)
                with self.assertRaises(ValueError):
                    list(records(path, chunk_size=2))

    def test_quotas_never_duplicate_samples(self):
        scenes = [{'name': 'short', 'token': 's', 'first_sample_token': 'a'},
                  {'name': 'long', 'token': 'l', 'first_sample_token': 'b'}]
        samples = {'a': {'token': 'a', 'scene_token': 's', 'next': ''},
                   'b': {'token': 'b', 'scene_token': 'l', 'next': 'c'},
                   'c': {'token': 'c', 'scene_token': 'l', 'next': 'd'},
                   'd': {'token': 'd', 'scene_token': 'l', 'next': ''}}
        selected = select_samples(scenes, samples, ['short', 'long'], 4)
        self.assertEqual({r['token'] for r in selected}, set(samples))
        with self.assertRaises(ValueError):
            select_samples(scenes, samples, ['short', 'long'], 5)
        with self.assertRaises(ValueError):
            select_samples(scenes, samples, ['short', 'short'], 1)

    def test_split_overlap_rejected_before_reading(self):
        with self.assertRaisesRegex(ValueError, 'disjoint'):
            load_selected('/missing', 'v1.0-trainval', ['scene-a'], ['scene-a'], 1, 1, [])

    def test_trainval_keeps_only_selected_keyframe_records(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / 'v1.0-trainval'
            data.mkdir()
            tables = {
                'scene': [{'name': 'train', 'token': 's1', 'first_sample_token': 'a'},
                          {'name': 'val', 'token': 's2', 'first_sample_token': 'b'}],
                'sample': [{'token': 'a', 'scene_token': 's1', 'next': ''},
                           {'token': 'b', 'scene_token': 's2', 'next': ''}],
                'sensor': [{'token': 'camera', 'channel': 'CAM_FRONT'}],
                'calibrated_sensor': [{'token': 'cal', 'sensor_token': 'camera'}],
                'ego_pose': [{'token': 'used'}, {'token': 'unused'}],
                'sample_data': [{'token': 'selected', 'sample_token': 'b', 'is_key_frame': True,
                                 'calibrated_sensor_token': 'cal', 'ego_pose_token': 'used'},
                                {'token': 'sweep', 'sample_token': 'b', 'is_key_frame': False,
                                 'calibrated_sensor_token': 'cal', 'ego_pose_token': 'unused'}]}
            for name, values in tables.items():
                (data / (name + '.json')).write_text(json.dumps(values))
            tables, selected = load_selected(directory, 'v1.0-trainval', ['train'], ['val'],
                                              0, 1, ['CAM_FRONT'])
            self.assertEqual([s['token'] for _, s in selected], ['b'])
            self.assertEqual(set(tables['sample_data']), {'selected'})
            self.assertEqual(set(tables['ego_pose']), {'used'})
            with self.assertRaises(ValueError):
                load_selected(directory, 'v1.0-trainval', ['train'], ['val'], 0, 1,
                              ['CAM_FRONT'], [{'split': 'calibration', 'token': 'b'}])


if __name__ == '__main__':
    unittest.main()
