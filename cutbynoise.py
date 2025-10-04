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
import os
import scipy.signal as signal
import subprocess
import sys
import tempfile
import traceback


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
    p.add_argument('--down',          action='store_true', default=True, help='down-sample inputs to 13.75 kHz to speed up processing (default: yes)')
    p.add_argument('--hq',            dest='down', action='store_false', help='read inputs at full sample rate (default: --down)')
    # for some samples/inputs, 4000 or even 2000 is good enough
    p.add_argument('--hz',            default=13750, type=int, help='sample rate to use when --down is specified (default: %(default)d)')
    p.add_argument('--json',          metavar='FILENAME', help='write positions to json file')
    p.add_argument('--window',  '-w', action='store_true', help='limit memory usage by aligning in a moving window')
    p.add_argument('--silence',       nargs='?', const=1.0, metavar='SECONDS', type=float, help='search for n second silence as end marker (default: 1 s if specified)')

    args = p.parse_args()

    args.marks = [ args.begin ]
    args.level = logging.WARNING
    if args.verbose:
        args.level = logging.INFO
    if args.debug:
        args.level = logging.DEBUG
    if args.end:
        args.marks.append(args.end)
    if args.silence:
        if args.end:
            raise RuntimeError('--end conflicts with --silence')

    return args


def ffprobe(filename):
    r = subprocess.run(['ffprobe', '-loglevel', 'warning', '-show_format', '-show_streams',
                        '-of', 'json', filename],
                   capture_output=True, text=True, check=True)
    d = json.loads(r.stdout)
    return d


def probe_wav(filename):
    h = ffprobe(filename)
    rate = int(h['streams'][0]['sample_rate'])
    return rate


def ffmpeg_read(filename, rate=None):
    ms = []
    if rate is not None:
        ms.extend(['-ar', str(rate)])
    r = subprocess.run(( ['ffmpeg', '-loglevel', 'warning', '-vn', '-i', filename, '-f', 'wav', '-acodec', 'pcm_s16le']
                       + ms
                       # only left channel
                       + ['-filter', 'pan=1c|c0=c0', 'pipe:'] ),
                       capture_output=True, check=True)
    return r.stdout


def read_wav(filename, target_rate=None):
    rate = probe_wav(filename)
    extra = {}
    if target_rate and str(target_rate) != str(rate):
        extra['rate'] = str(target_rate)
    bs = ffmpeg_read(filename, **extra)
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



def pair_positions(prev, offs_s):
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


def align(hay, needle, rate, thresh, pos, prev=None, overlap=False):
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
    if pos[0] == 0:
        if pos[1] == 1:
            ks[::2] -= needle.size
        else: # i.e. pos[1] == 2
            ks -= needle.size

    offs_s = ks / rate

    offs_s = pair_positions(prev, offs_s)

    return offs_s


def yield_window(filename, window_size, overlap_size, rate):
    def to_wav(bs):
        bale = np.frombuffer(bs, np.int16)
        bale = bale.astype('float32', casting='safe')
        return bale

    # NB: each sample is 2 byte big ...
    n, k = window_size * 2, overlap_size * 2
    bs = bytearray(n + k)
    xs = memoryview(bs)[k:]
    ys = memoryview(bs)[-k:]
    zs = memoryview(bs)[:k]
    off = 0
    off_inc = n // 2
    with subprocess.Popen(['ffmpeg', '-loglevel', 'warning', '-vn', '-i', filename,
                           # only left channel
                           '-filter', 'pan=1c|c0=c0', '-f', 'wav',
                           '-acodec', 'pcm_s16le', '-ar', str(rate), 'pipe:'],
                          stdout=subprocess.PIPE, stdin=subprocess.DEVNULL) as p:
        while True:
            l = p.stdout.readinto(xs)
            if off == 0:
                yield off, to_wav(xs)
                off = (n - k) // 2
            elif l < len(xs):
                yield off, to_wav(memoryview(bs)[:k + l])
                break
            else:
                yield off, to_wav(bs)
                off += off_inc
            zs[:] = ys
        # NB: context exit also waits but without a timeout
        p.wait(60)


def find_silence(xs, needle, rate):
    ys = np.square(xs)
    n = len(needle)
    # NB: if we scale ys by `/n` and `np.sqrt()` zs
    #     we would get the the sliding RMS of the input signal
    zs = signal.correlate(ys, needle, mode='full', method='fft')
    t = 0.01 * np.max(zs) # multiply by 0.05 or so when using the RMS
    # i.e. filter out  bottoms that lie below the threshold t
    # and stretch over at least n samples, i.e. our silence
    rs = np.nonzero(np.diff(np.nonzero(zs > t)[0]) > n)[0]
    rs += n
    return rs


def align_window(hay_fn, needles, rate, thresh):
    ys = [ np.dot(needle, needle) for needle in needles ]
    #n = needle.size * 100
    k = max(int(needle * rate) if type(needle) is float else needle.size for needle in needles)
    n = k * 50
    ysP = [ np.max(signal.correlate(needle, needle, mode='full', method='fft')) for needle in needles ]
    assert abs(1 - ys[0] / ysP[0]) < 1e-5
    if len(needles) > 1:
        assert abs(1 - ys[1] / ysP[1]) < 1e-5
    hs = [ thresh * y for y in ys ]
    kks = [ [ np.zeros(0, dtype='int64') ] for _ in needles ]
    if len(needles) == 2 and type(needles[1]) is float:
        silent_needle = np.ones(int(needles[1] * rate))
    for off, bale in yield_window(hay_fn, n, k, rate):
        for i, needle in enumerate(needles):
            if type(needle) is float:
                ks = find_silence(bale, silent_needle, rate)
                if ks.size:
                    kks[i].append(ks + off)
            else:
                xs = signal.correlate(bale, needle, mode='full', method='fft')
                ks, _ = signal.find_peaks(xs, height=hs[i], distance=10*rate)
                if ks.size:
                    kks[i].append(ks + off)

    for i in range(len(kks)):
        kks[i] = np.concatenate(kks[i])
        if i == 0:
            if len(kks) == 1:
                kks[i][::2] -= needles[0].size
            else:
                kks[i] -= needles[0].size
        kks[i] = kks[i] / rate

    if len(needles) == 1:
        return kks

    kks[1] = pair_positions(kks[0], kks[1])

    return kks


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


def ffmpeg_cutout(filename, begin, end, ofilename):
    # XXX `-acodec copy` basically equivalent to `-c copy` when dealing with podcast files, right ...
    r = subprocess.run(['ffmpeg', '-loglevel', 'warning',
                        '-ss', str(begin), '-to', str(end), '-i', filename,
                        '-acodec', 'copy', ofilename, '-y'],
                       capture_output=True, text=True, check=True)


def write_noise(ifilename, off_pairs, odir):
    os.makedirs(odir, exist_ok=True)
    _, ext = os.path.splitext(ifilename)
    for i, (b, e) in enumerate(off_pairs, 1):
        ofilename = f'{odir}/{i:02}{ext}'
        log.info(f'Writing noise to {ofilename} ...')
        ffmpeg_cutout(ifilename, b, e, ofilename)


def ffmpeg_punchout(filename, ofilename):
    # XXX `-acodec copy` basically equivalent to `-c copy` when dealing with podcast files, right ...
    r = subprocess.run(['ffmpeg', '-loglevel', 'warning', '-f', 'concat', '-safe', '0',
                        '-i', filename,
                        '-acodec', 'copy', ofilename, '-y'],
                       capture_output=True, text=True, check=True)


def write_output(f, ifilename, offs_s, ofilename):
    ls = itertools.cycle(('outpoint', 'inpoint'))
    fn = os.path.abspath(ifilename)
    print(f"file '{fn}'", file=f)
    for l, off in zip(ls, offs_s):
        print(f'{l} {off}', file=f)
        if l == 'outpoint':
            print(f"file '{fn}'", file=f)
    if sys.version_info < (3, 12):
        f.flush()
    else:
        f.close()
    ffmpeg_punchout(f.name, ofilename)


def write_json(offs_s, filename):
    with open(filename, 'w') as f:
        d = { 'offs_s': list(offs_s) }
        json.dump(d, f)


def main():
    args = parse_args()
    setup_logging(args.level)

    if args.window:
        if args.down:
            hay_rate = args.hz
        else:
            hay_rate = probe_wav(args.input)
        needles = []
        for i, needle_fn in enumerate(args.marks):
            needle, needle_rate = read_wav(needle_fn, hay_rate)
            if not args.down and hay_rate != needle_rate:
                log.warning(f'Sample rate mismatch: {hay_rate} ({args.input}) vs. {needle_rate} ({needle_fn})')
            needles.append(needle)
        if args.silence:
            needles.append(args.silence)
        rs = align_window(args.input, needles, hay_rate, args.thresh)
        log.info(f'Template matches at: {rs} (s)')
    else:
        if args.down:
            hay, hay_rate = read_wav(args.input, args.hz)
            hay_rate = args.hz
        else:
            hay, hay_rate = read_wav(args.input)
        rs = []
        n = len(args.marks)
        for i, needle_fn in enumerate(args.marks):
            needle, needle_rate = read_wav(needle_fn, hay_rate)
            if not args.down and hay_rate != needle_rate:
                log.warning(f'Sample rate mismatch: {hay_rate} ({args.input}) vs. {needle_rate} ({needle_fn})')
            log.info(f'Aligning {os.path.basename(needle_fn)} with {os.path.basename(args.input)} (overlap={args.overlap}) ...')
            offs_s = align(hay, needle, hay_rate, args.thresh, pos=(i, n), prev=(None if i==0 else rs[-1]), overlap=args.overlap)
            rs.append(offs_s)
            log.info(f'Template matches at: {offs_s} (s)')
        if args.silence:
            silent_needle = np.ones(int(args.silence * hay_rate))
            ks = find_silence(hay, silent_needle, hay_rate)
            if ks.size:
                offs_s = ks / hay_rate
                rs.append(offs_s)
                log.info(f'Silence matched at: {offs_s} (s)')

    offs_s = merge(rs)
    log.info(f'All template match positions: {offs_s} (s)')
    if offs_s.size == 0:
        raise RuntimeError("Didn't find any noise regions")
    if offs_s.size % 2 != 0:
        raise RuntimeError('Found unbalanced markers')
    if offs_s.size > 10:
        raise RuntimeError(f'Found too many regions: {offs_s.size/2}')
    off_pairs = offs_s.reshape((-1, 2))
    pp = str(off_pairs).replace('\n', ' ')
    log.info(f'Noisy regions: {pp} (s)')
    ds = np.diff(off_pairs, axis=1).squeeze()
    log.info(f'Duration(s) of noisy regions: {ds} (s)')
    if np.max(ds) > args.length:
        raise RuntimeError(f'Noise regions are longer than expected: {ds}')
    if np.min(ds) < 0:
        raise RuntimeError(f'Noise markers switched: {ds}')
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
            if sys.version_info < (3, 12):
                kw = {}
            else:
                kw = { 'delete_on_close': False }
            with tempfile.NamedTemporaryFile('w', **kw) as f:
                write_output(f, args.input, offs_s, args.output)


def mainP():
    try:
        return main()
    except subprocess.CalledProcessError as e:
        print(f'Command {e.cmd} failed ({e.returncode}): {e.stderr}', file=sys.stderr)
        traceback.print_exception(e, file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(mainP())

