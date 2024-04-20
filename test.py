#!/usr/bin/env python3

from functools import reduce
import json
from operator import add
import os
import pytest
import subprocess
import tomllib


with open('test.toml', 'rb') as f:
    data = tomllib.load(f)

@pytest.mark.parametrize("args", data['integration'])
def test_integration(args, tmp_path):
    markers, in_fn, offs_s = args['markers'], args['input'], args['offs_s']
    _, ext = os.path.splitext(in_fn)
    ms = reduce(add, zip(('-b', '-e'), tuple(markers)), ())
    out_fn = tmp_path / f'clean{ext}'
    json_fn = tmp_path / 'dump.json'
    trash_dir = tmp_path / 'del'
    subprocess.check_call(('./cutbynoise.py',) + ms
                          + (in_fn, '-o', out_fn, '--trash', trash_dir, '--json', json_fn, '-v', '--window')
                          + tuple(args.get('flags', [])) )
    with open(json_fn) as f:
        d = json.load(f)
    assert d['offs_s'] == pytest.approx(offs_s, abs=0.2)
