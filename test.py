#!/usr/bin/env python3

from functools import reduce
import json
from operator import add
import os
import pytest
import subprocess
import tomllib
import pathlib


from cutbynoise import ffprobe


with open('test.toml', 'rb') as f:
    data = tomllib.load(f)

@pytest.mark.parametrize("args", data['integration'])
@pytest.mark.parametrize("flags", ( (), ('--window',), ) )
def test_integration(args, flags, tmp_path):
    markers, in_fn, offs_s = args['markers'], args['input'], args['offs_s']
    _, ext = os.path.splitext(in_fn)
    ms = reduce(add, zip(('-b', '-e'), tuple(markers)), ())
    out_fn = tmp_path / f'clean{ext}'
    json_fn = tmp_path / 'dump.json'
    trash_dir = tmp_path / 'del'
    subprocess.check_call(('./cutbynoise.py',) + ms
                          + (in_fn, '-o', out_fn, '--trash', trash_dir, '--json', json_fn, '-v')
                          + flags
                          + tuple(args.get('flags', [])) )
    with open(json_fn) as f:
        d = json.load(f)
    assert d['offs_s'] == pytest.approx(offs_s, abs=0.4)

    idur = float(ffprobe(in_fn)['format']['duration'])
    odur = float(ffprobe(out_fn)['format']['duration'])
    tdur = 0
    for p in pathlib.Path(trash_dir).iterdir():
        tdur += float(ffprobe(str(p))['format']['duration'])
    assert idur == pytest.approx(odur + tdur, abs=0.3)

    fns = list(pathlib.Path(trash_dir).iterdir())
    fns.sort()
    assert len(fns) * 2 == len(args['offs_s'])
    for fn, i in zip(fns, range(0, len(args['offs_s']), 2)):
        dur = float(args['offs_s'][i+1]) - float(args['offs_s'][i])
        tdur = float(ffprobe(fn)['format']['duration'])
        assert dur == pytest.approx(tdur, abs=0.4)
