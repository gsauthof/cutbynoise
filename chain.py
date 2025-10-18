#!/usr/bin/env python3


# chain - chain filter commands together
#
# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: © 2025 Georg Sauthoff <mail@gms.tf>


# Example call:
#
#     chain somepodcastepisode.mp3 clean.mp3 -t c  --  cutbynoise -v --window %in -o %out \
#         --  -e c  --  --bof                    --end boff-start.flac  \
#         --  -e c  --  --begin boff-fin.flac    --eof                  \
#         --  -e c  --  --begin boff-begin.flac  --end boff-end.flac    \
#         --  -e c  --  --begin boff-end.flac                           \
#         --  -e c  --  --begin boff-end.flac --eof
#
# In this command, `chain` invokes `cutbynoise` five times with different
# command line options. Each time the input marker is replaced, i.e. either
# with the chain input filename or the output of the last successful previous command.
# The output marker is replaced with a temporary filename that is located in a temporary
# directory in the parent directory of the chain output file.
# Temporary files are removed when they aren't needed, anymore, by default.


import argparse
import logging
import os
import pathlib
import subprocess
import sys
import tempfile


log = logging.getLogger(__name__)


def setup_logging():
    logging.addLevelName(logging.DEBUG  , 'DBG')
    logging.addLevelName(logging.INFO   , 'INF')
    logging.addLevelName(logging.WARNING, 'WRN')
    logging.addLevelName(logging.ERROR  , 'ERR')
    logging.basicConfig(format='{asctime} {levelname}  {message} [{name}]',
                        style='{',
                        datefmt='%Y-%m-%d %H:%M:%S')


def add_arg_group(p):
    g = p.add_mutually_exclusive_group()
    g.add_argument('--template', '-t', help='next argument group is a template')
    g.add_argument('--append', '-a', action='store_true', help='append command to chain')
    g.add_argument('--extend', '-e', help='append command to chain, references template')


def find(xs, x):
    try:
        return xs.index(x)
    except ValueError:
        return len(xs)

def split_args(ys):
    i = find(ys, '--')
    xs = ys[:i]
    ys = ys[i+1:]
    return xs, ys


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--in-mark', default='%in', help='string to replace with current input (default: %(default)s)')
    p.add_argument('--out-mark', default='%out', help='string to replace with current output (default: %(default)s)')
    p.add_argument('--strict', action='store_true', help='fail on first error in chain (default is to fail if all commands in chain fail)')
    p.add_argument('--verbose', '-v', action='store_true', help='enable verbose logging')
    p.add_argument('--debug', '-d', action='store_true', help='enable debug logging')
    p.add_argument('--keep', '-k', action='store_true', help='keep temporary directory and files (default: auto-delete)')
    p.add_argument('input', help='input filename to put into the chain')
    p.add_argument('output', help='ouput filename to put into the chain')
    add_arg_group(p)
    q = argparse.ArgumentParser()
    add_arg_group(q)

    xs, ys = split_args(sys.argv[1:])
    args = p.parse_args(xs)
    zs = args

    ts = {}
    cs = []
    while True:
        xs, ys = split_args(ys)

        if zs.template:
            ts[zs.template] = xs
        elif zs.extend:
            cmd = ts[zs.extend].copy()
            cmd.extend(xs)
            cs.append(cmd)
        elif zs.append:
            cs.append(xs)

        if ys:
            xs, ys = split_args(ys)
            zs = q.parse_args(xs)
        else:
            break

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    log.debug(f'parsed templates: {ts}')
    log.debug(f'parsed commands: {cs}')

    return args, cs


def inc_filename(ext, k):
    k += 1
    ofn = f'o{k}{ext}'
    return k, ofn



def main():
    setup_logging()
    args, cs = parse_args()

    ext    = pathlib.Path(args.output).suffix
    obase  = pathlib.Path(args.output).parent
    ifn    = args.input
    k, ofn = inc_filename(ext, 0)
    ofns = []

    last_out = None
    with tempfile.TemporaryDirectory(dir=obase, delete=(not args.keep)) as d:
        for cmd in cs:
            new_out = False
            while args.in_mark in cmd:
                cmd[cmd.index(args.in_mark)] = ifn
            while args.out_mark in cmd:
                cmd[cmd.index(args.out_mark)] = f'{d}/{ofn}'
                new_out = True

            log.info(f'Executing: {" ".join(cmd)}')
            r = subprocess.run(cmd)
            log.info(f'done: {r.returncode}')

            if r.returncode == 0:
                if new_out:
                    ifn      = f'{d}/{ofn}'
                    last_out = ifn
                    k, ofn   = inc_filename(ext, k)
                    if not args.keep:
                        ofns.append(ifn)
                        if len(ofns) == 2:
                            log.debug(f'Removing {ofns[0]}')
                            os.unlink(ofns.pop(0))
            elif args.strict:
                raise RuntimeError('invocation failed')

        if last_out is None:
            raise RuntimeError('all invocations failed')
        else:
            pathlib.Path(last_out).rename(args.output)



if __name__ == '__main__':
    sys.exit(main())

