import gzip
import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from viewer.server import HEADER,VOXELS,Server,SnapshotStore,make_handler,validate_snapshot

def snapshot(sequence=1,label=4):
    return HEADER.pack(b'FOCC3D01',sequence,sequence%40,1792056000000+sequence,b'a'*32)+bytes([label])*VOXELS

class SnapshotTests(unittest.TestCase):
    def test_invalid_snapshot_rejected(self):
        for data in [b'',snapshot()[:-1],b'badmagic'+snapshot()[8:],snapshot(label=18)]:
            with self.assertRaises(ValueError):validate_snapshot(data)
    def test_atomic_replace_and_etag(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'latest.occ';path.write_bytes(snapshot())
            store=SnapshotStore(path);first=store.get()
            self.assertEqual(gzip.decompress(first[1]),first[0]);self.assertEqual(store.get()[2],first[2])
            temporary=path.with_suffix('.tmp');temporary.write_bytes(snapshot(2,7));temporary.replace(path)
            second=store.get();self.assertNotEqual(first[2],second[2]);self.assertEqual(second[3]['sequence'],2)
            self.assertEqual(first[0][-1],4);self.assertEqual(second[0][-1],7)
    def test_http_delivery_and_waiting(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'latest.occ';server=Server(('127.0.0.1',0),make_handler(SnapshotStore(path)))
            worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
            base=f'http://127.0.0.1:{server.server_port}'
            try:
                with urllib.request.urlopen(base+'/api/config') as response:self.assertEqual(json.load(response)['shape'],[200,200,16])
                with self.assertRaises(urllib.error.HTTPError) as caught:urllib.request.urlopen(base+'/api/frame')
                self.assertEqual(caught.exception.code,503)
                path.write_bytes(snapshot())
                request=urllib.request.Request(base+'/api/frame',headers={'Accept-Encoding':'gzip'})
                with urllib.request.urlopen(request) as response:
                    etag=response.headers['ETag'];self.assertEqual(gzip.decompress(response.read()),snapshot())
                request=urllib.request.Request(base+'/api/frame',headers={'If-None-Match':etag})
                with self.assertRaises(urllib.error.HTTPError) as caught:urllib.request.urlopen(request)
                self.assertEqual(caught.exception.code,304)
            finally:server.shutdown();server.server_close();worker.join()

if __name__=='__main__':unittest.main()
