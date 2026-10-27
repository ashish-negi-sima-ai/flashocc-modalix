import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from runtime_config import defaults


class RuntimeConfigTests(unittest.TestCase):
    def test_environment_overrides_local_settings_without_stale_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'runtime.json'
            path.write_text(json.dumps({'device': 'sima@old', 'insight_api': 'https://sdk:9900'}))
            with patch.dict(os.environ, {'FLASHOCC_CONFIG': str(path), 'FLASHOCC_DEVICE': 'sima@new'}, clear=True):
                self.assertEqual(defaults()['device'], 'sima@new')
                self.assertEqual(defaults()['insight_api'], 'https://sdk:9900')
            with patch.dict(os.environ, {'FLASHOCC_CONFIG': str(path.parent / 'missing')}, clear=True):
                self.assertIsNone(defaults()['device'])
                self.assertIsNone(defaults()['insight_api'])

    def test_bad_local_configuration_is_not_silently_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'runtime.json'
            for bad in [{'devcie': 'typo'}, {'device': ''}, {'device': 123}, []]:
                path.write_text(json.dumps(bad))
                with patch.dict(os.environ, {'FLASHOCC_CONFIG': str(path)}, clear=True):
                    with self.assertRaises(ValueError):
                        defaults()


if __name__ == '__main__':
    unittest.main()
