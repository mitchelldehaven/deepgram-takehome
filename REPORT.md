# What a leaderboard WER actually measures

A reproduction of the [🤗 Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard)
(English short-form), on 1200 utterances (2.18 h) drawn from three of its
datasets, across five system configurations.

Setup and rerun instructions are in [README.md](README.md). Every number below
comes from `outputs/scores/`, and every run stamps its git hash into `logs/`.

---

## Summary

1. **The reproduction holds.** Across 9 system×dataset cells, our WER lands a
   mean **+0.58** above published, and the published number falls inside our
   95% CI in **7 of 9**. The bias is one-directional, which points at
   configuration rather than noise.
2. **Normalization decides what you may conclude, not who is ahead.** Under
   the leaderboard's normalizer, Deepgram Nova-3 with default formatting beats
   Whisper large-v3 by 0.92 WER and the paired bootstrap calls it (p<0.001).
   Under a second defensible normalizer the ordering is the same but the gap
   is 0.28 and **unresolvable** (p=0.36). Same audio, same models, same
   metric — a conclusion that exists under one set of text conventions and
   not the other.
3. **Output formatting is worth up to 1.6 WER on its own.** The *same*
   Nova-3 model on the *same* audio scores 0.56 (leaderboard normalizer) to
   1.62 (ours) worse purely by leaving the API's punctuation and numeral
   formatting on. Nothing about the recognition changed.
4. **The top three are statistically indistinguishable.** Parakeet (6.04),
   Nova-3 raw (6.05) and Qwen3-ASR (6.08) are separated by 0.04 WER where the
   paired bootstrap resolves ±0.35. The leaderboard presents them at ranks
   36, unranked, and 17.
5. **WER hides what kind of error, and three added metrics disagree with it
   about who wins.** Whisper's WER is driven by *deletions* (3.11 per 100
   words, double everyone else's). Qwen3-ASR hallucinates three times less
   than anything else and is separated from the whole field at p<0.001, while
   sitting third on WER. Parakeet is first on WER and **last** on
   hallucination.
6. **Part of a vendor's WER is the benchmark's segmentation, not the model.**
   Nova-3's edge deletions on AMI drop by a third when the audio is padded
   with silence, which adds no information and so cannot be a recognition
   gain. That closes 44% of its gap to Parakeet.
7. **Our own metadata failed its first validation.** The SNR estimators are
   biased by +0.8 to +12.1 dB against injected ground truth. Their *ranking*
   is sound (ρ 0.92–0.98), so tercile breakdowns are defensible and absolute
   dB values are not.

---

## 1. Recreating it

### The subset

Three datasets spanning the board's difficulty range: **AMI-Cleaned**
(meetings, spontaneous), **GigaSpeech-Cleaned** (podcast/YouTube) and
**SPGISpeech** (earnings calls). 400 utterances each, 1200 total, 22,439
reference words.

The upstream parquets are **sorted by audio length, globally and descending**.
Reading the first shard or the first N rows therefore returns only the longest
utterances, and duration correlates with WER. The sampler instead takes shards
at even intervals across the sorted order and samples rows within each, which
spans the duration distribution by construction. Resulting spread:

| Dataset | n | duration min/median/max | ref words |
|---|---|---|---|
| AMI-Cleaned | 400 | 1.0 / 3.0 / 17.4 s | 4,297 |
| GigaSpeech-Cleaned | 400 | 1.1 / 6.5 / 19.1 s | 8,334 |
| SPGISpeech | 400 | 5.0 / 8.9 / 15.0 s | 9,808 |

**Why 400 is enough** is not an assertion, it is a measurement. The paired
bootstrap resolves system differences down to about **±0.35 WER** on this
sample. That is sufficient to separate Whisper from the rest (gaps of 1.5–2.5)
and insufficient to separate the top three (gaps of 0.04). Both of those are
reported as such below rather than papered over. Resolving the top three would
need roughly 100× the data, which is a real statement about the leaderboard,
not a defect in the sample.

### Reproduction vs published

Leaderboard normalizer, as published:

| System | Dataset | Ours | Published | Gap |
|---|---|---|---|---|
| whisper-large-v3 | AMI-Cleaned | 14.64 | 13.63 | +1.01 |
| whisper-large-v3 | GigaSpeech-Cleaned | 9.49 | 8.35 | +1.14 |
| whisper-large-v3 | SPGISpeech | 2.84 | 2.71 | +0.13 |
| Qwen3-ASR-1.7B | AMI-Cleaned | 9.27 | 8.31 | +0.96 |
| Qwen3-ASR-1.7B | GigaSpeech-Cleaned | 8.03 | 7.22 | +0.81 |
| Qwen3-ASR-1.7B | SPGISpeech | 3.05 | 2.57 | **+0.48** |
| parakeet-tdt-0.6b-v2 | AMI-Cleaned | 8.34 | 9.09 | −0.75 |
| parakeet-tdt-0.6b-v2 | GigaSpeech-Cleaned | 9.20 | 8.15 | +1.05 |
| parakeet-tdt-0.6b-v2 | SPGISpeech | 2.37 | 1.94 | **+0.43** |

Bold = published value outside our 95% CI.

**Why they differ.** In rough order of contribution:

- **Subset, not the full split.** We score 400 of 7,805 (AMI) to ~39,300
  (SPGI) utterances. This is the dominant term and it is symmetric noise —
  but our CIs already account for it, and 7 of 9 published values land inside.
- **Decoding configuration.** We run greedy, batch 32, fp16, `language="en"`
  forced. The leaderboard's per-model scripts differ in batching and
  precision, and batching changes padding, which changes Whisper output.
- **Model version drift.** `nvidia/parakeet-tdt-0.6b-v2` is a moving
  reference; we pin the leaderboard version (`20-11-2026`) but cannot pin
  their inference commit.
- **Our AMI result is *better* than published for Parakeet (−0.75).** A
  one-directional bias with one exception is more consistent with sampling
  plus config than with a systematic error in our pipeline.

**The headline "Average WER" is not reproducible by anyone outside Hugging
Face.** Two of its nine default columns are private datasets, and a third
(`Earnings22-Cleaned-AA-chunked`) is scored by concatenating chunks against a
separately hosted full-transcript set. We compare per-dataset numbers, which
are reproducible. This is worth stating plainly: the single number at the top
of the board cannot be independently audited.

### A note on who is and isn't ranked

Deepgram appears **nowhere** on this leaderboard — not in any of its 32
versions back to 2026-05-20, not in `api/providers/` on GitHub. Meanwhile 22
competitor APIs have submitted integrations, including AssemblyAI (#18),
ElevenLabs (#8), Speechmatics, Rev, Soniox, Gladia, Azure (#4), OpenAI and
Zoom (#1). Proprietary systems get listed by the vendor submitting a provider
file, so presence is opt-in and absence is a decision.

I can observe the absence; I can't observe the reason, and this report doesn't
guess at one. The consequence is what matters: **there is no published number
for Deepgram on the benchmark its competitors are ranked on**, so the only way
to get a comparable figure is to run it through the identical pipeline. That
is what `deepgram_nova_3_raw` is.

---

## 2. Testing the normalization

The leaderboard uses Whisper's `EnglishTextNormalizer` (vendored here at
pinned commit `c469619`). It makes choices nobody publishes alongside the
numbers: it deletes filler words, expands contractions, strips apostrophes,
maps British to American spelling, and canonicalises number words to digits.

Our alternative (`config/normalization/custom.yaml`) disagrees on six
specific points, each independently togglable. These are not *better* choices
— they are *defensible* ones, which is the whole point. (It disagreed on
seven until one of them turned out to be indefensible; see *The rule that did
not survive the diff tool*, below.)

### Ranking under each normalizer

| | Leaderboard normalizer | | Our normalizer | |
|---|---|---|---|---|
| | WER | 95% CI | WER | 95% CI |
| parakeet-tdt-0.6b-v2 | **6.04** | [5.49, 6.66] | **7.62** | [7.01, 8.30] |
| deepgram-nova-3 (raw) | 6.05 | [5.46, 6.68] | 7.73 | [7.10, 8.39] |
| Qwen3-ASR-1.7B | 6.08 | [5.54, 6.66] | 8.04 | [7.43, 8.71] |
| deepgram-nova-3 (formatted) | 6.62 | [6.04, 7.26] | 9.35 | [8.63, 10.08] |
| whisper-large-v3 | 7.54 | [6.91, 8.22] | 9.63 | [8.92, 10.40] |

**The orderings agree.** Every system sits at the same rank under both
normalizers. An earlier version of this table had two rank flips, and they
were an artifact of a bad rule of ours rather than a property of the
normalizers — see below, because how that was found matters more than the
flips did.

What does not survive the change of normalizer is the *significance*. The
load-bearing row is the last two:

| Nova-3 (formatted) vs whisper-large-v3 | Δ WER | 95% CI | p |
|---|---|---|---|
| leaderboard normalizer | **−0.92** | [−1.36, −0.47] | <0.001 |
| ours | −0.28 | [−0.93, +0.38] | 0.36 |

Both normalizers put Nova-3 ahead. Only one of them entitles you to say so.
A buyer reading "Nova-3 beats Whisper by 0.9 WER" is reading a fact about the
normalizer as much as about the models: under text conventions that are no
less defensible, the same 1,200 utterances cannot tell the two apart. The
claim that moves is not the ranking, it is the confidence — which is the
claim a procurement decision actually rests on.

### Which rule does the work

Mean |ΔWER| across systems from flipping one rule:

| Rule | Δ WER | What it does |
|---|---|---|
| `remove_fillers` | 0.65 | deletes um/uh/mm before scoring |
| `spell_out_numbers` | 0.42 | digits ↔ words |
| `expand_contractions` | 0.17 | "don't" → "do not" |
| `keep_apostrophes` | 0.07 | |
| `british_to_american` | 0.04 | |
| `normalize_currency` | 0.01 | near-no-op: no currency symbols in these refs |
| `strip_non_speech` | 0.00 | these refs carry no `[laughter]` markers |

One rule dominates, and it is the one to worry about:
**deleting disfluencies before scoring is a 0.65 WER subsidy**, and it
is not neutral between systems — it rewards models that don't emit fillers and
costs nothing to models that do. On spontaneous speech that is a meaningful
thumb on the scale, and it is invisible in the published number.

### The rule that did not survive the diff tool

A number in the ablation table says a rule matters but not *why*, so
`src/eval/compare.py` re-scores with one rule flipped and prints the
utterances that moved. Run against `collapse_hyphenation` — then the largest
line in the table, 0.93 WER on Nova-3 alone — the mechanism was immediate:

```
ref   ... as a fortytwoyearold that i have one more year to get all my selfcontrol together
A     ... as a [+forty] [+two] [+year] {fortytwoyearold>old} ... all my [+self] {selfcontrol>control} together
B     ... as a forty two year old that i have one more year to get all my self control together
```

The rule closed hyphens up: `re-run` → `rerun`, but also
`forty-two-year-old` → `fortytwoyearold`. The second is a token no recognizer
will ever emit, and the metric charged four errors for it. Nobody utters a
hyphen; a speaker says four words and every system writes four.

The fix is to treat a hyphen as the word boundary it is. That turned out to
delete the rule rather than repair it, because **the leaderboard's normalizer
already does exactly that** — a hyphen is Unicode category `Pd`, and its
`remove_symbols_and_diacritics` maps every `P` character to a space. Once ours
agreed, there was nothing left to disagree about, so the toggle is gone and
the custom normalizer is down to six rules.

Three things followed, and the third is the one that matters:

1. Every custom-normalizer WER fell by 1.6–2.1 points.
2. `spell_out_numbers` rose from 0.27 to 0.42, because it had been partly
   masked: spelled-out numbers carry hyphens, so the collapse rule was
   absorbing some of its effect.
3. **Both rank flips disappeared.** The earlier table had Qwen3 above Nova-3
   raw, and Whisper above Nova-3 formatted, under our normalizer only. Neither
   was a fact about normalization. Both were our own rule inflating WER
   unevenly — it punishes systems in proportion to how often their output
   meets a hyphenated reference, which is a property of the reference text,
   not of the recognizer.

So the headline claim of this section got *weaker and truer at the same
time*. "The ranking moves when the normalization changes" was the more
dramatic finding and it was partly an artifact of a bug in the thing doing the
measuring. What is left — the ordering holds, the significance does not — is a
smaller claim that is actually load-bearing, and it is the one that survives
scrutiny.

The general lesson is the uncomfortable one. This rule had a docstring, a
passing test, a config toggle and a number in an ablation table. It was the
*largest* number in that table, which made it look like the most important
finding in the section rather than the biggest error in it. Nothing about
reading the code revealed the problem. Printing one utterance did.

### Output formatting is a measurement artifact

Nova-3 twice, same model, same audio, differing only in API flags
(`punctuate`, `smart_format`, `numerals`):

| Normalizer | raw | formatted | Δ | p |
|---|---|---|---|---|
| leaderboard | 6.05 | 6.62 | **+0.56** | <0.001 |
| ours | 7.73 | 9.35 | **+1.62** | <0.001 |

The open models the board ranks emit bare lowercase text; the normalizer grew
up around that convention and handles it cleanly. A commercial API that
returns production-ready formatted text is penalised for it. **Anyone
benchmarking a formatting API against leaderboard numbers must turn formatting
off first, or they will measure the wrong thing by up to 1.6 WER** — larger
than the gap between ranks 17 and 53 on these datasets.

---

## 3. Looking past the metric

### What WER misses

WER weights every word equally and every error type equally. Two systems at
the same WER can be very different products. The sub/del/ins split, leaderboard
normalizer:

| System | WER | sub | del | ins |
|---|---|---|---|---|
| parakeet-tdt-0.6b-v2 | 6.04 | 3.22 | 1.57 | 1.25 |
| deepgram-nova-3 (raw) | 6.05 | 3.08 | 1.59 | 1.38 |
| Qwen3-ASR-1.7B | 6.08 | 3.05 | 1.62 | 1.41 |
| whisper-large-v3 | 7.54 | 3.18 | **3.11** | 1.24 |

Whisper's substitution rate is *normal*. Its deletion rate is **double**
everyone else's, and that single component is essentially its entire gap from
the field. Whisper isn't mishearing more — it's dropping content, consistent
with its known tendency to truncate. For captioning that's a silent
correctness failure; for a voice agent extracting a confirmation number,
dropping is far worse than substituting, because a substitution is often
detectable downstream and a deletion is not.

### The three metrics I'd add

All three answer the same question from different directions: *what does an
average hide?* Each is a function `(references, hypotheses) -> per-utterance
(numerator, denominator)`, which is what lets all four share one bootstrap, one
significance test and one breakdown path. Thresholds are in
`config/eval/openasr.yaml` so a reader can move them and rescore in seconds.

| Metric | Definition | Denominator | Whose decision |
|---|---|---|---|
| **Catastrophic failure** | utterances whose own WER > 50% | utterances | staffing a human review queue, writing an SLA |
| **Hallucination** | hypothesis ≥ 1.2× reference length | utterances | anyone for whom invented text is worse than missing text — legal, medical, voice agents |
| **Edge error** | errors in the first/last 2 reference words | edge word slots | anyone benchmarking a prerecorded API on pre-segmented audio; and the benchmark's own designers |

The denominators differ on purpose, and the difference is the point. On this
subset WER divides by 22,891 reference words, the two rate metrics by 1,194
utterances, and the edge metric by 4,204 edge word slots. A rate over words and
a rate over utterances answer different questions, and a system can look fine
on one while being unshippable on the other.

### Result: the four metrics disagree about who wins

Leaderboard normalizer, all 1,200 utterances:

| System | WER | Catastrophic | Hallucination | Edge error |
|---|---|---|---|---|
| parakeet-tdt-0.6b-v2 | **6.04** | 2.26 | 2.01 | **8.82** |
| deepgram-nova-3 (raw) | 6.05 | 2.51 | 1.76 | 10.28 |
| Qwen3-ASR-1.7B | 6.08 | **2.01** | **0.59** | 9.04 |
| deepgram-nova-3 (formatted) | 6.62 | 2.51 | 1.84 | 11.04 |
| whisper-large-v3 | 7.54 | 3.02 | 1.68 | 12.25 |

**Qwen3-ASR hallucinates three times less than anything else, and that is the
one result here the sample resolves cleanly.** It is third on WER — inside a
three-way tie — but its hallucination rate of 0.59 separates it from all four
other configurations at p<0.001 (gaps of 1.17 to 1.42, every interval clear of
zero). Four separated pairs out of ten, and all four are Qwen against the
field.

Meanwhile **Parakeet is first on WER and last on hallucination.** The system
the leaderboard would hand you is the one most likely to invent material. For a
voice agent reading back an order number, that is the wrong trade, and no
amount of staring at the published column reveals it.

**Whisper is second-best on hallucination despite being worst on WER.** That is
consistent with its error composition above: it deletes, it does not invent.
Whichever of those two failures you can tolerate decides whether Whisper's
7.54 or Parakeet's 6.04 is the better number for you.

### Where the catastrophic tail lives

| System | AMI | GigaSpeech | SPGISpeech |
|---|---|---|---|
| whisper-large-v3 | **7.61** | 1.50 | 0.00 |
| deepgram-nova-3 (raw) | 5.84 | 1.50 | 0.25 |
| parakeet-tdt-0.6b-v2 | 5.58 | 1.25 | 0.00 |
| Qwen3-ASR-1.7B | 5.08 | 1.00 | 0.00 |

Unusable utterances are ~20× more common on meeting audio than on earnings
calls, for every system. A pooled 2–3% is the average of "one in thirteen" and
"never", which is not a number anyone can staff against.

**Catastrophic failure is the one metric this sample cannot support.** Only 1 of
10 system pairs separates. At a ~2.5% rate over 1,200 utterances there are
roughly 30 events per system, so the metric is badly underpowered and I report
it as directional only. That is a sample-size problem, not a reason to drop the
metric — it is the first thing I would spend more utterances on, because it is
also the metric with the most direct operational consequence.

### Edge error, and why it is a measurement artifact as much as a result

Errors concentrate at utterance boundaries for every system, because every
utterance here is a slice cut out of a longer recording. Decomposed on AMI, per
100 edge slots:

| System | edge sub | edge del | edge ins | mid del |
|---|---|---|---|---|
| parakeet-tdt-0.6b-v2 | 5.27 | 2.59 | 3.39 | 1.88 |
| deepgram-nova-3 (raw) | 5.00 | **6.52** | 3.75 | 2.71 |
| Qwen3-ASR-1.7B | 4.64 | 4.91 | 2.14 | 3.59 |
| whisper-large-v3 | 6.70 | **11.43** | 2.14 | 7.90 |

Nova-3's deficit against Parakeet at the edges is **+3.93 deletions** with
substitutions and insertions essentially level (−0.27, +0.36). It is omission,
not invention.

So I tested whether it was recoverable. `src/eval/padding_probe.py` re-runs
Nova-3 with 0.5 s and 1.0 s of digital silence prepended and appended —
which adds **no information**, so any improvement cannot be a recognition
gain:

| Condition | AMI WER | edge del | edge ins |
|---|---|---|---|
| unpadded | 9.99 | 6.52 | 3.75 |
| +0.5 s silence | **9.27** | **4.38** | 2.50 |
| +1.0 s silence | 9.48 | 4.64 | 2.41 |

Edge deletions fall by a third, middle deletions do not move (2.71 → 2.78),
and the WER change on AMI is −0.72 [−1.35, −0.09], p=0.022. Edge *insertions*
fall too, so padding suppressed hallucination rather than inducing it over the
new silence — the opposite of what I expected, and plausibly because an abrupt
truncation mid-utterance is itself what reads as continuing speech.

Three caveats, because this is the most over-claimable result in the report:

1. **Eight tests were run; the Bonferroni threshold is 0.00625 and none
   survive it.** The deletion effect is credible because it is large and
   uniform in direction across all six dataset×condition cells, not because of
   any single p-value.
2. **It is not monotone.** 0.5 s beats 1.0 s, which argues against a simple
   "the endpointer needs runway" story.
3. **On SPGISpeech, deletions fell 52% but WER did not move**, because
   insertions rose to compensate. My prediction that the clean corpus would be
   an inert control was wrong: padding changes edge behaviour everywhere, it
   only *helps* where deletions dominate.

Padding closes about 44% of Nova-3's AMI gap to Parakeet (+1.65 → +0.93), so
segmentation explains part of it and transcription conventions explain the
rest. The decision this informs is concrete: **anyone benchmarking a
prerecorded API against leaderboard numbers on pre-segmented corpora should
pad the audio, or they are partly measuring the endpointer instead of the
recognizer.**

---

## 4. Characterizing the data

Per-utterance metadata in `src/data/metadata.py`: duration, two SNR estimates,
two speech-rate estimates, voiced fraction, disfluency rate, numeric rate.

### Testing the metadata before trusting it

Two estimators per acoustic field, chosen not to share a failure mode:

| Field | Spearman | bin agreement | κ | verdict |
|---|---|---|---|---|
| `snr_db` | +0.931 | 0.834 | +0.751 | OK |
| `speech_rate` | +0.908 | 0.780 | +0.670 | OK |

But agreement between two estimators is weak evidence — they can share an
assumption and be wrong together. So `src/data/validate_metadata.py`
manufactures ground truth: take the cleanest 25% of utterances, inject white
and babble noise at known SNR (0–20 dB), and measure recovery.

| Estimator | Noise | Bias | MAE | Spearman w/ truth |
|---|---|---|---|---|
| WADA | white | +0.8 dB | 1.6 dB | +0.972 |
| WADA | babble | +5.4 dB | 5.5 dB | +0.931 |
| percentile | white | +7.0 dB | 7.0 dB | +0.980 |
| percentile | babble | **+12.1 dB** | 12.1 dB | +0.921 |

**This changes what the metadata may be used for.** The absolute values are
badly biased — the percentile estimator overstates SNR by 7–12 dB, which is
why AMI's median reads as a nonsensical 35.6 dB. But rank correlation with
truth is 0.92–0.98 across the board. So **tercile bins are trustworthy and
absolute dB figures are not**, and nothing in this report quotes an SNR in dB
as if it were calibrated.

This check also caught a real bug. My first WADA implementation inverted the
model analytically and saturated at the grid maximum for all 1200 utterances,
which produced a confident "SNR is unreliable, do not slice on it" verdict
that was entirely an artifact of my own code. Two-estimator agreement alone
would have surfaced it as a property of the data. Ground truth surfaced it as
a bug.

### Breakdowns

WER by speech rate (tercile), leaderboard normalizer:

| System | slow | mid | fast |
|---|---|---|---|
| whisper-large-v3 | 6.87 | 5.12 | **10.37** |
| Qwen3-ASR-1.7B | 5.53 | 4.24 | 8.25 |
| parakeet-tdt-0.6b-v2 | 5.48 | 4.24 | 8.19 |
| deepgram-nova-3 (raw) | 5.33 | 4.50 | 8.06 |

The pattern is U-shaped, and the fast tercile is where systems separate: the
spread across systems roughly doubles from mid (0.88) to fast (2.31). Whisper
degrades hardest. Slow speech is also worse than mid — these are the sparse,
hesitant, disfluent utterances rather than carefully articulated ones.

WER by SNR tercile (ordering trustworthy, values not) runs 8.42 → 7.28 → 6.97
for Whisper and similarly for the rest: monotone, as expected, and a smaller
effect than speech rate.

By dataset, the systems do **not** agree on which corpus is hard:

| System | AMI | GigaSpeech | SPGI |
|---|---|---|---|
| whisper-large-v3 | 14.64 | 9.49 | 2.84 |
| Qwen3-ASR-1.7B | 9.27 | 8.03 | 3.05 |
| parakeet-tdt-0.6b-v2 | **8.34** | **9.20** | 2.37 |
| deepgram-nova-3 (raw) | 9.99 | 8.06 | 2.66 |

Parakeet is best on AMI and worst-but-one on GigaSpeech. A macro-average over
datasets, which is what the board reports, erases this completely — and which
corpus resembles your traffic is the only question a buyer actually has.

---

## Limitations

- **Three of nine datasets.** No URGENT24 (noisy), no Voice Arena (accented),
  no Earnings22-chunked, and the two private sets are unavailable. Our
  conclusions about ranking sensitivity hold on the three we ran; the full
  board mixes in conditions we did not test, and URGENT24 in particular would
  likely stress the SNR breakdown much harder.
- **400 utterances per dataset** resolves ~±0.35 WER. Insufficient for the
  top three. Stated throughout rather than worked around.
- **Single run per system.** Greedy decoding is deterministic, so this costs
  nothing for the open models, but the Deepgram API could change under us
  without notice. No version pinning is available there.
- **Decoding config is ours, not theirs.** We did not replicate each model's
  per-script batching and precision. This is the most likely cause of the
  one-directional +0.58 reproduction gap and the first thing I would chase
  with more time.
- **Catastrophic failure rate is underpowered here.** At ~2.5% over 1,200
  utterances it rests on ~30 events per system and separates 1 of 10 pairs. It
  is reported as directional. More utterances would fix this, and it is where
  I would spend them first.
- **The hallucination metric uses length as a proxy.** A hypothesis 1.2x the
  reference length is usually invention, but a system that substitutes two
  words for one trips it without inventing anything, and one that replaces a
  sentence with an equally long fabrication does not trip it at all. It is a
  screen, not a verdict. It is also sensitive on short utterances, where a
  3-word hypothesis against a 2-word reference is 50% over.
- **An earlier keyword error rate was dropped rather than reported.** Its
  vocabulary was defined in-set over 1,200 references, which is a noisy rarity
  estimate, and every confidence interval overlapped every other, so it could
  not support the reordering it appeared to show.
- **Injected noise is friendlier than real noise.** Additive white and babble
  is not reverberation or overlapping speech, which is what actually makes AMI
  hard. The recovery figures are an upper bound on estimator quality.
- **`spell_out_numbers` is still imperfect.** `$5.50` becomes "five point
  five dollars" rather than "five fifty". It is applied identically to
  reference and hypothesis so it does not bias the comparison, but it is not
  how anyone speaks.
- **Our normalizer's currency path is effectively untested.** These three
  reference sets contain **zero** currency symbols and **one** comma-grouped
  number across 1,200 utterances, so `normalize_currency` ablates to 0.01 and
  the thousands-separator handling fires once. Three real bugs lived in that
  code until I spot-checked it on handwritten sentences (see *Where I used
  AI*); on financial or medical audio, where number handling matters most,
  this is the part of the normalizer I would least trust without first
  writing it a proper test set.
- **No multi-seed subset.** One sample at seed 42. Re-drawing the subset would
  give a cleaner estimate of sampling variance than the bootstrap alone.

## How I'd automate this

The pipeline is already structured for it — raw hypotheses are cached
separately from scoring, so normalization and metric changes are seconds, not
GPU-hours. To run it continuously:

1. **Pin everything, fail loudly on drift.** Dataset revision, model revision,
   normalizer commit and leaderboard version are all config today. A nightly
   job should diff them and fail rather than silently rescore against a moved
   reference. `dirtygit` already ties every result to a commit.
2. **Make the subset a fixed, versioned artifact.** Publish the sampled
   utterance IDs as a dataset so the sample stops being a function of a seed
   and a sort order that upstream may change.
3. **Treat normalization as a tracked dimension, not a constant.** Every
   result should carry its normalizer hash. The scoring grid is already
   system × dataset × normalizer; a regression dashboard should show all
   three, because a normalizer change can decide whether a model change is
   reportable — here it is the difference between p<0.001 and p=0.36 on the
   same 1,200 utterances.
4. **Gate on CIs, not point estimates.** A nightly "model A beat model B"
   alert should fire on the paired bootstrap, not on the means. Most of the
   movement on a leaderboard this tight is noise.
5. **Scale the sample until the comparison you care about is resolvable.**
   Measure the gap you need to detect, then size the sample to it. 400/dataset
   detects 0.35 WER; detecting 0.05 is a different exercise, and knowing which
   one you need is the first question.
6. **Separate pipeline health from model quality.** Empty-hypothesis counts,
   API error rates and metadata reliability should alarm independently. Three
   of my five systems returned a handful of empty transcripts; that must never
   be silently absorbed into a WER.

## Where I used AI, and where I didn't

Claude (via Claude Code) was used heavily throughout. The split below is
specific rather than a blanket disclaimer, because which parts were generated
turned out to matter for how the work had to be checked.

**AI did most of the typing**: boilerplate across the four system backends,
the parquet/shard plumbing, dataframe wrangling, docstrings, and the first
drafts of this report. It also did the API spelunking that established the
leaderboard's current shape — that the results repos are gated, that the
public Gradio endpoint serves the same table, and that Deepgram has never
appeared in any version.

**I made the judgment calls**: choosing this leaderboard and these three
datasets; the systematic-sampling design once the length-sort was discovered;
which seven normalization rules to disagree on and why; using a *paired*
bootstrap rather than comparing independent CIs; the decision to validate
metadata against injected ground truth rather than settle for two estimators
agreeing; and every claim in this report about what is and isn't statistically
supported.

**Four times, the useful finding came from distrusting the AI-written code
rather than reading it.** The WADA-SNR implementation looked plausible and was
wrong — it saturated at the grid maximum for every utterance and produced a
confident, entirely spurious "metadata is unreliable" finding. The keyword
metric's original capitalization-based vocabulary also looked plausible and
silently excluded a third of the data, because GigaSpeech references are
lowercase -- that metric was later dropped for unrelated reasons, but the
bug is why I stopped trusting plausible-looking code here. The custom
normalizer held three number-handling bugs, including
one where the same quantity tokenized differently depending on whether the
source wrote `twenty-five` or `25` — which defeats the entire purpose of
canonicalizing numbers, and which moved the top rank under our own normalizer
once fixed. And `collapse_hyphenation` welded `forty-two-year-old` into a
single token, inflating every custom-normalizer WER by about 1.8 points and
manufacturing two rank flips that this report would otherwise have presented
as its headline finding.

That last one is the sharpest version of the problem. It was not an obscure
helper: it was the *largest number in the ablation table*, which is to say it
looked like the most important result in the section rather than the biggest
error in it. A bug that makes a finding look stronger is the one least likely
to be investigated.

Each was found the same way: by making the code emit output a human could
eyeball, on inputs chosen to be awkward, and reading it. None was found by
code review — all four passed a careful read. That is the main thing I'd
carry into doing this work at scale. Generated analysis code fails in ways
that produce *publishable-looking numbers*, and the numbers are the artifact
nobody checks, because checking them is the work the numbers were supposed to
replace.
