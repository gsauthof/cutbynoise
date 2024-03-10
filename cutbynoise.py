#!/usr/bin/env python3


# cutbynoise - cut sample enclosed noise regions out of sound files
#
# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: © 2024 Georg Sauthoff <mail@gms.tf>


import argparse
import itertools
import json
import logging
import numpy as np
import ffmpeg
import os
import scipy.signal as signal
import sys
import tempfile


log = logging.getLogger(__name__)


def setup_logging(level):
    old_f = logging.getLogRecordFactory()
    def f(*xs, **kw):
        r = old_f(*xs, **kw)
        r.epoch = r.relativeCreated / 1000.0
        level_dict = { 10 : 'DBG',  20 : 'INF', 30 : 'WRN', 40 : 'ERR', 50 : 'CRI' }
        r.lvl = level_dict[r.levelno]
        return r
    logging.setLogRecordFactory(f)
    log_format = '{epoch:05.1f} {lvl} {message}    [{name}]'
    logging.basicConfig(format=log_format, level=level, style='{')

    np.set_printoptions(linewidth=np.inf, threshold=np.inf)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('input',           help='input audio filename')
    p.add_argument('--output',  '-o', metavar='FILENAME', help='trimmed output filename')
    p.add_argument('--begin',   '-b', required=True, help='sample that starts a noise region')
    p.add_argument('--end',     '-e', help='sample that ends a noise region')
    p.add_argument('--thresh',  '-t', default=0.69, type=float, help='correlation threshold (default: %(default)f)')
    p.add_argument('--trash',         metavar='DIRECTORY', help='store noise snippets in directory (default: they are discarded)')
    p.add_argument('--cut-file',      metavar='FILENAME', help='filename for ffmpeg input list (default: temporary file)')
    p.add_argument('--overlap',       default=True, action='store_true', help='use faster overlap-add method (default: on)')
    p.add_argument('--std',           dest='overlap', action='store_false', help='use standard fft cross-correlation (default: --overlap)')
    p.add_argument('--length',  '-l', default=400, type=int, help='maximum lenghth of a noisy region in seconds (default: %(default)d)')
    p.add_argument('--verbose', '-v', action='store_true', help='print verbose messages')
    p.add_argument('--debug',   '-d', action='store_true', help='print debug messages')
    p.add_argument('--down',          action='store_true', default=True, help='down-sample inputs to 4 kHz to speed up processing (default: yes)')
    p.add_argument('--hq',            dest='down', action='store_false', help='read inputs at full sample rate (default: --down)')
    # for some samples/inputs, 4000 or even 2000 is good enough
    p.add_argument('--hz',            default=13750, type=int, help='sample rate to use when --down is specified (default: %(default)d)')
    p.add_argument('--json',          metavar='FILENAME', help='write positions to json file')

    args = p.parse_args()

    args.marks = [ args.begin ]
    args.level = logging.WARNING
    if args.verbose:
        args.level = logging.INFO
    if args.debug:
        args.level = logging.DEBUG
    if args.end:
        args.marks.append(args.end)

    return args


def read_wav(filename, target_rate=None):
    h = ffmpeg.probe(filename)
    rate = int(h['streams'][0]['sample_rate'])
    extra = {}
    if target_rate and str(target_rate) != str(rate):
        extra['ar'] = str(target_rate)
    bs, err = (
            ffmpeg.input(filename)
            .output('pipe:',
                    format='wav',
                    acodec='pcm_s16le',
                    map_channel='0.0.0',  # only left channel
                    **extra)
            .run(capture_stdout=True, capture_stderr=True)
            )
    wav = np.frombuffer(bs, np.int16)
    wav = wav.astype('float32', casting='safe')
    return wav, rate


# copied from https://github.com/scipy/scipy/blob/4edfcaa3ce8a387450b6efce968572def71be089/scipy/signal/_signaltools.py#L1101
# cf. https://github.com/scipy/scipy/blob/4edfcaa3ce8a387450b6efce968572def71be089/scipy/signal/_signaltools.py#L243
def _reverse_and_conj(x):
    """
    Reverse array `x` in all dimensions and perform the complex conjugate
    """
    reverse = (slice(None, None, -1),) * x.ndim
    return x[reverse].conj()


def align(hay, needle, rate, thresh, first, prev=None, overlap=False):
    y = np.dot(needle, needle)
    if overlap:
        xs = signal.oaconvolve(hay, _reverse_and_conj(needle), mode='full')
        yP = np.max(signal.oaconvolve(needle, _reverse_and_conj(needle), mode='full'))
    else:
        xs = signal.correlate(hay, needle, mode='full', method='fft')
        yP = np.max(signal.correlate(needle, needle, mode='full', method='fft'))
    log.debug('Self correlation of sample: {}'.format(abs(1 - y / yP)))
    assert abs(1 - y / yP) < 1e-5
    h = thresh * y
    ks, hs = signal.find_peaks(xs, height=h, distance=10*rate)
    if first:
        ks[::2] -= needle.size
    offs_s = ks / rate

    # might be true when end sample is used elsewhere, besides pairings with the begin sample ...
    if prev is not None and prev.shape < offs_s.shape:
        rs = []
        for p in prev:
            t = [ x for x in offs_s if x > p ]
            if t:
                k = min(t)
                rs.append(k)
        log.info(f'Found more end positions than beginnings ({offs_s}) - filtering to: {rs}')
        offs_s = np.array(rs)

    return offs_s

def merge(rs):
    if len(rs) == 1:
        return rs[0]
    xs, ys = rs
    if xs.size != ys.size:
        raise RuntimeError(f'Not all markers pair: {xs.size} beginnings vs. {ys.size} ends')
    v = np.empty((xs.size + ys.size,), dtype=xs.dtype)
    v[0::2] = xs
    v[1::2] = ys
    return v

def write_noise(ifilename, off_pairs, odir):
    os.makedirs(odir, exist_ok=True)
    _, ext = os.path.splitext(ifilename)
    for i, (b, e) in enumerate(off_pairs, 1):
        ofilename = f'{odir}/{i:02}{ext}'
        log.info(f'Writing noise to {ofilename} ...')
        out, err = (
                ffmpeg.input(ifilename, ss=str(b), to=str(e))
                .output(ofilename, acodec='copy')
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
                )

def write_output(f, ifilename, offs_s, ofilename):
    ls = itertools.cycle(('outpoint', 'inpoint'))
    fn = os.path.abspath(ifilename)
    print(f"file '{fn}'", file=f)
    for l, off in zip(ls, offs_s):
        print(f'{l} {off}', file=f)
        if l == 'outpoint':
            print(f"file '{fn}'", file=f)
    f.close()
    out, err = (
            ffmpeg.input(f.name, format='concat', safe=0)
            .output(ofilename, acodec='copy')
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
            )


def write_json(offs_s, filename):
    with open(filename, 'w') as f:
        d = { 'offs_s': list(offs_s) }
        json.dump(d, f)


def main():
    args = parse_args()
    setup_logging(args.level)

    if args.down:
        hay, hay_rate = read_wav(args.input, args.hz)
        hay_rate = args.hz
    else:
        hay, hay_rate = read_wav(args.input)

    rs = []
    for i, needle_fn in enumerate(args.marks):
        needle, needle_rate = read_wav(needle_fn, hay_rate)
        if hay_rate != needle_rate:
            log.warning(f'Sample rate mismatch: {hay_rate} ({args.input}) vs. {needle_rate} ({needle_fn})')
        log.info(f'Aligning {os.path.basename(needle_fn)} with {os.path.basename(args.input)} (overlap={args.overlap}) ...')
        offs_s = align(hay, needle, hay_rate, args.thresh, first=(i==0), prev=(None if i==0 else rs[-1]), overlap=args.overlap)
        rs.append(offs_s)
        log.info(f'Template matches at: {offs_s} (s)')
    offs_s = merge(rs)
    log.info(f'All template match positions: {offs_s} (s)')
    if offs_s.size == 0:
        raise RuntimeError("Didn't find any noise regions")
    if offs_s.size % 2 != 0:
        raise RuntimeError('Found unbalanced markers')
    if offs_s.size > 10:
        raise RuntimeError(f'Found too many regions: {offs_s.size/2}')
    off_pairs = offs_s.reshape((-1, 2))
    log.info(f'Noisy regions: {str(off_pairs).replace("\n", " ")} (s)')
    ds = np.diff(off_pairs, axis=1).squeeze()
    log.info(f'Duration(s) of noisy regions: {ds} (s)')
    if np.max(ds) > args.length:
        raise RuntimeError(f'Noise regions are longer than expected: {ds}')
    if args.trash:
        write_noise(args.input, off_pairs, args.trash)
    if args.json:
        write_json(offs_s, args.json)
    if args.output:
        log.info(f'Writing cut {args.output} ...')
        if args.cut_file:
            with open(args.cut_file, 'w') as f:
                write_output(f, args.input, offs_s, args.output)
        else:
            with tempfile.NamedTemporaryFile('w', delete_on_close=False) as f:
                write_output(f, args.input, offs_s, args.output)


if __name__ == '__main__':
    sys.exit(main())

