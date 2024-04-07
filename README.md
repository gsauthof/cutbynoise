This repository contains cutbynoise - a tool for cutting regions
that are delimited by certain samples (a.k.a. jingles, markers or
templates) out of larger sound files.

For example, you can use it on a multi-hour long podcast episode
MP3 file where the producer inserts audio advertisements and
marks such inserts with static samples.

That way you can easily listen again and again to an
advertisement you found interesting.

Alternatively, you can listen to an episode without having to
manually pause and skip over advertisements you have grown to
hate and whose markers have conditioned you to do so, already.

Example usage:

```
$ ./cutbynoise.py -b ldn-begin-short.flac -o ldn372-clean.mp3 ldn372.mp3 --trash del -v
004.7 INF Aligning ldn-begin-short.flac with ldn372.mp3 (overlap=True) ...    [__main__]
006.5 INF Template matches at: [1918.26029091 1994.97614545 4029.22829091 4112.20814545] (s)    [__main__]
006.5 INF All template match positions: [1918.26029091 1994.97614545 4029.22829091 4112.20814545] (s)    [__main__]
006.5 INF Noisy regions: [[1918.26029091 1994.97614545]  [4029.22829091 4112.20814545]] (s)    [__main__]
006.5 INF Duration(s) of noisy regions: [76.71585455 82.97985455] (s)    [__main__]
006.5 INF Writing noise to del/01.mp3 ...    [__main__]
006.9 INF Writing noise to del/02.mp3 ...    [__main__]
007.1 INF Writing cut ldn372-clean.mp3 ...    [__main__]
```

NB: In this case the begin template also matches the end one.
In case the region is delimited by truly different templates you
can specify the second one via `-e`.


## How it works

Cutbynoise [cross-correlates][cc] each template with the input
sound file. To speed this up and reduce memory usage, only the
mono signal is cross-correlated, the signals are down-sampled,
the cross-correlation is computed via [convolution][conv], using
[FFT][fft] and the [overlap-add method][oa].

In a direct implementation, [cross-correlating][cc] a shorter
signal with a longer one basically means computing the
dot product over a sliding window.
Peaks in the resulting vector are possible template matches.
Thus, it can be seen as a variant of the sequence alignment
problem.

For the heavy-lifting it uses [ffmpeg][ffmpeg] (via
[ffmpeg-python][ffmpegp]), i.e. for reading and resampling the
waveform signals and cutting the input file, and [SciPy][scipy]
for the cross-correlation.

Optionally, cutbynoise is able to correlate the templates in a
moving window (cf. `--window`). This reduces the memory usage
further (e.g. 145 Mib RSS instead of 4.5 GiB RSS, on a 4 h long
mp3 file) and is thus recommended on memory constrained systems.
Note that the window mode trades memory savings against an
increased runtime (e.g. 28 s instead of 23 s, on a 2023 laptop
CPU, using the same example input).


## Related Methods

In case the begin and end templates of regions aren't very static
one can look into standard methods used in speech recognition,
i.e. especially [Dynamic Time Warping (DTW)][dtw]. See also e.g.
the [Dynamic Time Warping Suite][dtws] which provides packages
for Python and R (FWIW, I haven't tested them, yet).

When it comes to podcasts with ads, some of them provide a
premium feed that also contains all the episodes without any ads.
Thus, if it fits your budget and you want to support the podcast
hosts you can consider to simply pay for such a premium feed on a
regular basis, instead of fiddling with cutbynoise.

However, not all podcasts (understandably) can be bothered to
provide multiple feeds. For example, AFAICS, the c't uplink
podcast is only available with ads, i.e. even if you have a c't
Plus paid subscription.

Also, at some point, everyone's budget is limited, i.e. one can
only financially support so many projects.

As always, one can also simply unsubscribe podcasts that contain
too many annoying ads. Arguably, there are enough podcasts
available that don't contain any advertisements.


## Getting Started

Cutbynoise can be installed via pip or directly be executed,
given that the required dependencies (cf. `pyproject.toml`) are installed.

On Fedora, you can install the dependencies system-wide like this:

```
dnf install python3-ffmpeg-python python3-scipy python3-numpy ffmpeg
```

For creating suitable templates (samples) that delimit your
regions you can use your favourite audio editor. For example,
you can use the open-source [Audacity][audacity] for this task
(cf.  File -> Export -> Export Selected Audio ...), which is
packaged by many Linux distributions.


## Limitations

As is, cutbynoise doesn't take extra care of additional meta-data
contained in the input file.
Especially chapter marks aren't adjusted, i.e. in the output
file some chapter marks might be off by whatever amount was cut.
Also, embedded images might not remain in the output file.

FWIW, both limitations might be fixed by moderate extra effort,
e.g. by finding the right ffmpeg options and possibly adjusting
chapter marks programmatically.

OTOH, the ad business is very dynamic and wants to target
victims as much as possible, most ads are often automatically
inserted on-the-fly for each download. That means depending on
the date, ad-auction outcomes and phase of the moon, a certain
podcast episode download link might contain quite different ads
at quite different locations, including none.
Since developers working in the ad industry allegedly couldn't care
less, their ad pasting code likely doesn't adjust any chapter
marks, in the first place. Thus, when simply cutting out the ad
noise _not_ adjusting any existing chapter marks might be the
right thing and actually improve the chapter marks accuracy.


## License

Cutbynoise is licensed under the [GPL version 3 or later][gpl].


## See Also

Cutbynoise can be integrated with [castproxy](https://github.com/gsauthof/feed-util#castproxypy),
Atom feed generator that aggregates audiocasts (podcasts).


[cc]: https://en.wikipedia.org/wiki/Cross-correlation
[conv]: https://en.wikipedia.org/wiki/Cross-correlation#Properties
[fft]: https://en.wikipedia.org/wiki/Fast_Fourier_transform
[oa]: https://en.wikipedia.org/wiki/Overlap%E2%80%93add_method
[ffmpeg]: https://en.wikipedia.org/wiki/FFmpeg
[ffmpegp]: https://github.com/kkroening/ffmpeg-python
[scipy]: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.correlate.html
[gpl]: https://en.wikipedia.org/wiki/GNU_General_Public_License
[dtw]: https://en.wikipedia.org/wiki/Dynamic_time_warping
[dtws]: https://dynamictimewarping.github.io/
[audacity]: https://en.wikipedia.org/wiki/Audacity_(audio_editor)

