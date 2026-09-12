import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('preserve', Path(__file__).resolve().parents[1] / 'tools/preserve_experiment.py')
preserve = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preserve)


class BackupTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.project = root / 'project'
        self.project.mkdir()
        for args in (['init', '-q'], ['config', 'user.name', 'Test'], ['config', 'user.email', 'test@example.invalid']):
            subprocess.run(['git', '-C', str(self.project), *args], check=True)
        (self.project / 'model.py').write_text('model_version = 1\n')
        subprocess.run(['git', '-C', str(self.project), 'add', 'model.py'], check=True)
        subprocess.run(['git', '-C', str(self.project), 'commit', '-qm', 'fixture'], check=True)
        self.run = self.project / 'exp/suite/full_train'
        self.run.mkdir(parents=True)
        for name in ('best.pt', 'last.pt', 'train.csv', 'val.csv'):
            (self.run / name).write_bytes(b'fixture data\n')
        (self.run / 'perf_summary.json').write_text('{"returncode": 0}')
        config = {}
        for key in ('dnet_ckpt', 'fnet_ckpt', 'magnet_ckpt', 'raw_wh_json', 'train_split', 'val_split'):
            source = self.project / key
            source.write_bytes((key + '\n').encode())
            config[key] = str(source)
        (self.run / 'config.json').write_text(json.dumps(config))
        self.destination = root / 'backups'

    def test_copy_and_tamper_detection(self):
        original = preserve.sha256(self.run / 'best.pt')
        backup = preserve.preserve(self.project, self.run, self.destination, 'HEAD')
        self.assertFalse((backup / 'INCOMPLETE').exists())
        self.assertEqual(preserve.sha256(self.run / 'best.pt'), original)
        self.assertTrue((backup / 'code.tar.gz').is_file())
        self.assertTrue((backup / 'dependencies/dnet_ckpt/dnet_ckpt').is_file())
        preserve.verify(backup)
        (backup / 'experiment/full_train/best.pt').write_bytes(b'corrupt')
        with self.assertRaises(RuntimeError):
            preserve.verify(backup)

    def test_reject_recursive_destination(self):
        with self.assertRaises(ValueError):
            preserve.preserve(self.project, self.run, self.run / 'backups', 'HEAD')

    def test_reject_failed_run(self):
        (self.run / 'perf_summary.json').write_text('{"returncode": 1}')
        with self.assertRaises(ValueError):
            preserve.preserve(self.project, self.run, self.destination, 'HEAD')


if __name__ == '__main__':
    unittest.main()
