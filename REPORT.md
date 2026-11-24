# Reproducing the Open ASR Leaderboard

I reproduced the [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) on AMI, GigaSpeech and SPGISpeech with five systems:

* Whisper large-v3, 
* Qwen3-ASR-1.7B, 
* Parakeet-TDT-0.6B-v2, 
* Deepgram Nova-3 (raw and formatted). 

I scored 400 clips per dataset (1,200 clips, 2.2 h), then re-ran the three open models on all 65,914 clips of those splits (144 h) to
validate the sampling approach. Setup and rerun instructions are in
[README.md](README.md); every number below comes from a CSV in
`outputs/scores/` or `outputs-full/scores/`.

To tell a real gap from noise I redrew the clip list at random 2,000 times and
measured how far each WER moved, using the same re-drawn list for both systems
in a comparison.

The sample was sized for pooled comparisons, and that is where it holds: at
1,194 clips it detects gaps of 0.54 WER. Sliced to one dataset it leaves ~400
clips and detects 0.83 on GigaSpeech, 0.69 on SPGISpeech and only 1.57 on AMI,
whose short clips make per-clip error variance large. Resolving 0.5 WER on AMI
would need about 3,900 clips, so 400 was the wrong size for an AMI ranking —
the design error was claiming per-dataset results from a sample built for
pooled ones.

So per-dataset comparisons between systems below come from the full splits
(AMI: 7,715 clips, error bar 0.4). Two per-dataset results are from the
1,200-clip sample, because Nova-3 and the metadata were only run there; both
are flagged in place and stated as directional rather than as rankings.

## Findings

**The reproduction is close but uniformly high.** On the full splits all nine
system × dataset numbers land above published, by +0.18 to +0.78, averaging
+0.45. The published ordering reproduces exactly.

| System | AMI | GigaSpeech | SPGISpeech |
|---|---|---|---|
| parakeet-tdt-0.6b-v2 | 9.51 (+0.42) | 8.83 (+0.68) | 2.12 (+0.18) |
| Qwen3-ASR-1.7B | 8.75 (+0.44) | 7.84 (+0.62) | 2.82 (+0.25) |
| whisper-large-v3 | 14.10 (+0.47) | 9.13 (+0.78) | 2.94 (+0.23) |

The gap isn't sampling (I scored every clip), isn't batching (0.01 WER), and
isn't one broken system, since three different models on three code paths all
shifted by nearly the same amount. The likeliest cause is decoding
configuration — I run greedy, batch 32, fp16, English forced, while the board
uses per-model scripts. The sample itself held up: the full-split WER fell
inside the 1,200-clip error bar in all 9 cells.

Two things worth flagging about the benchmark. Four of its nine datasets aren't
reproducible (two private, two absent from the public repo), so the headline
"Average WER" can't be checked by anyone outside Hugging Face. And the source
files are sorted longest-audio-first, so reading the first N rows gives you
only the longest clips — which matters because clip length is the largest
single driver of WER here.

**One normalization rule decides who wins AMI.** The leaderboard's normalizer
deletes fillers ("um", "uh", "hmm") before scoring. I chose to make a normalizer that keeps them, knowing that AMI is full of these types of disfluencies. Some models benefit from this kind of normalization than others, like Whisper, so looked at this specifically. The systems disagree about whether fillers are words at all: AMI's references contain 37.4 fillers per 1,000 words, and Parakeet writes 35.0 of them while Whisper writes 3.6 and Qwen3 2.2 — a training-data difference, not an accuracy
one. Deleting fillers is therefore worth 2.8 WER to Whisper and 3.1 to Qwen3,
but only 0.5 to Parakeet. On every scorable AMI clip:

| Normalizer | Parakeet | Qwen3 | Winner |
|---|---|---|---|
| leaderboard's (deletes fillers) | 9.51 | 8.75 | Qwen3, by 0.75 |
| mine (keeps fillers) | 11.22 | 13.91 | Parakeet, by 2.69 |

Whether you want this type of filler removal normalization tends to be task specific decision. For something like meeting notes, you probably don't care about how many times someone said "um". For legal / medical note taking, you probably want the transcript to be as close as possible to the true. 

**Formatting is penalised as if it were error.** Nova-3 with default
formatting on scores +0.56 worse under the leaderboard's normalizer and +1.36
worse under mine, same model and same audio. The normalizer grew up around
open models that emit bare lowercase text, so an API returning
production-ready punctuation and numerals pays for it — more than the gap
between ranks 17 and 53 on these datasets.

**Equal WERs hide unequal failure modes.** Parakeet, Nova-3 raw and Qwen3 sit
at 6.04 / 6.05 / 6.08 — a three-way tie. Whisper's 7.53 is almost entirely one
column: it deletes 3.11 words per 100 against roughly 1.6 for everyone else,
while its substitution rate is normal. Whisper tends to have a problem of missing transcript portions, apparently worse than other models used. 

**The metric I added: a long-output rate**, the share of clips where a system
emits at least 1.2x the reference's word count — a cheap screen for a decoder
running past the end of the speech. Qwen3 scores 0.59 against 1.68 to 2.01 for
the others and is the only system that separates from the field, while its WER
sits inside the three-way tie. That is the whole result: the gaps among the
other four (0.25 to 0.34, p = 0.37 to 0.51) are well below what 1,194 clips
resolve, so they can't be ranked.

Reading the flagged clips also narrows what the metric means. Of Parakeet's 24,
most are speech captured at the clip edge that AMI's reference truncates away,
full words written where the reference has stutter fragments (`i d` → `i do
not`), or digit strings written as spoken words; two or three are substitution
cascades on bad audio and none is a runaway decoder. So on pre-segmented
conversational speech this metric largely measures verbatim style and reference
truncation, and only a terseness outlier like Qwen3 stands clear of that. It
would do its intended job better on long-form audio, where runaway decoding
actually occurs.

**Some of a vendor's score belongs to the benchmark.** Errors cluster at clip
edges for every system, because every clip was cut out of a longer recording.
Adding 0.5 s of digital silence to each end of Nova-3's AMI clips — which adds
no information — drops its WER from 9.97 to 9.23 and its edge deletions from
6.52 to 4.38, while middle deletions don't move.

This is 394 AMI clips, since Nova-3 was never run on the full splits, so treat
the direction as the result and not the magnitude. It is a paired before/after
on identical audio, which has far lower variance than comparing two systems,
and the effect points the same way in five of six dataset × condition
cells — but I ran eight versions of the test and the strongest alone would
come up by chance about 1 time in 70, which out of eight tries is roughly what
noise gives. What I'd defend: pre-segmentation costs a production endpointer
real WER on this corpus. What I would not: the 0.74-point figure, or the
derived claim that it accounts for half of Nova-3's AMI gap. Confirming the
size needs Nova-3 on the full 7,805-clip AMI split, which is 8 hours of audio.

**Clip properties move WER more than model choice does.** Breaking results
down by per-clip metadata, the spread from worst to best bin is 7.4 WER for
duration, 7.0 for disfluency rate, 6.8 for word count, 4.5 for numeric
content, 3.9 for speech rate, 1.6 for SNR and 1.2 for voiced fraction. The
spread across all five systems is 1.5, and across the two normalizers 1.9.
Five of seven clip properties matter more than which system you pick.

That partly dissolves "AMI is the hard corpus." AMI's clips are much shorter
(124 scored clips under two seconds, against 36 of GigaSpeech's 400).
Reweighting AMI to GigaSpeech's length distribution removes 1.8–2.4 WER points,
which is the entire AMI–GigaSpeech gap for four of the five systems. This is
also the 1,200-clip sample, so the per-system numbers carry ~1.6 WER of
uncertainty each and I wouldn't rank systems by them; what makes the result
credible is that five independently trained systems agree on the size to within
0.7 points, which noise would not produce. Whisper is the exception and the important one: its AMI gap
is 5.08 and length only takes it to 3.32, so length explains about a third of
the result that "AMI is hard" is built on.

**Not all of these fields are equally trustworthy, and the ones carrying the
findings are the solid ones.** Duration and word count are counted directly
from the audio and the transcript. Noise and speech rate are estimates, so I
checked them: two independent estimators agree on the ordering of clips
(0.91–0.93), but against manufactured ground truth both overstate SNR by up to
12.1 dB, which is why AMI's median reads as a nonsensical 35.6 dB. They rank
clips correctly (0.92–0.98) and nothing more, so noise enters only as an
ordering and never as a dB figure. Voiced fraction has no second estimator at
all and is the weakest column in the table.

## Limitations

- Three of nine datasets, with no noisy or accented set.
- Sizing, as above: built for pooled comparisons, so anything Deepgram or
  metadata-based is stuck at ~400 clips per dataset. I first read Parakeet as
  beating Qwen3 on AMI by 0.93 — below that resolution, and the full split
  reversed the sign.
- Decoding configuration is my explanation for the +0.45 gap by elimination,
  not by demonstration. Replicating one official script would settle it.
- The long-output metric uses length as a proxy, so it also misses an equally
  long fabrication and false-positives on a two-for-one substitution.
- The filler, length and noise results aren't controlled experiments. What
  generalises from the filler finding is the mechanism, not the 2.7 points.

## How I'd automate this

The pipeline already separates transcription from scoring, so re-scoring under
a new normalizer or metric takes seconds rather than GPU-hours. For continuous
use I'd pin dataset, model, normalizer and leaderboard versions in config and
fail a nightly job on drift rather than silently rescoring against a moved
target; publish the sampled clip IDs as a fixed artifact so the sample doesn't
depend on a seed and an upstream sort order; run a cheap sample nightly with
the full splits on release candidates and reconcile the two
(`src/eval/subset_validation.py` checks that the full number lands inside the
sample's error bar and attributes any difference to batching, sampling or
config); carry the normalizer version on every result, since a normalization
change can decide which model wins; alarm on paired comparisons rather than
point estimates, because most movement on a board this tight is noise; never
report a rule's effect averaged across datasets, which understated the filler
rule's AMI effect several times over when I first computed it; and keep pipeline
health (empty outputs, API errors, metadata checks) alarming separately from
model quality.

## Where I used AI, and where I didn't

Claude (via Claude Code) wrote most of the code: the four system backends, the
data plumbing, the docstrings, and first drafts of this report. It also looked
up which of the leaderboard's datasets are gated.

I chose the leaderboard and the three datasets, designed the sample once I
found the length sort, picked which normalization rules to disagree on, and
decided what the results do and don't support.

Checking the generated code took a specific approach, because reading it did
not work. Three number-handling bugs in my normalizer passed a careful read,
and so did a rule that closed up hyphens, which turned `forty-two-year-old`
into `fortytwoyearold` and added about 1.8 points to every WER I reported. I
found them by making the code print its output on deliberately awkward inputs
and reading that instead.

The costliest mistake was not a bug. My rule-by-rule table was correct, but it
averaged across datasets, so AMI's filler effect got diluted by SPGISpeech,
which contains two filler words in 948,632. The table reported 0.65 where the
real effect was 2.4 to 3.1, large enough to change which model wins. No test
would catch that, because nothing was broken. I noticed because I expected
Whisper to gain from dropping fillers and the number said otherwise.
