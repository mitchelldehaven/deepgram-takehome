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
2. **Normalization decides the ranking.** Under the leaderboard's normalizer,
   Deepgram Nova-3 with default formatting beats Whisper large-v3 by 0.92 WER
   (p<0.001). Under a second defensible normalizer, that gap vanishes
   (+0.11, p=0.78). Same audio, same models, same metric — different text
   conventions.
3. **Output formatting is worth up to 1.6 WER on its own.** The *same*
   Nova-3 model on the *same* audio scores 0.56 (leaderboard normalizer) to
   1.63 (ours) worse purely by leaving the API's punctuation and numeral
   formatting on. Nothing about the recognition changed.
4. **The top three are statistically indistinguishable.** Parakeet (6.04),
   Nova-3 raw (6.05) and Qwen3-ASR (6.08) are separated by 0.04 WER where the
   paired bootstrap resolves ±0.35. The leaderboard presents them at ranks
   36, unranked, and 17.
5. **WER hides what kind of error.** Whisper's WER is driven by *deletions*
   (3.11 per 100 words, double everyone else's). A product team should read
   that very differently from an equal-WER system that substitutes.
6. **Our own metadata failed its first validation.** The SNR estimators are
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

Our alternative (`config/normalization/custom.yaml`) disagrees on seven
specific points, each independently togglable. These are not *better* choices
— they are *defensible* ones, which is the whole point.

### Ranking under each normalizer

| | Leaderboard normalizer | | Our normalizer | |
|---|---|---|---|---|
| | WER | 95% CI | WER | 95% CI |
| parakeet-tdt-0.6b-v2 | **6.04** | [5.49, 6.66] | 8.59 | [7.92, 9.31] |
| deepgram-nova-3 (raw) | 6.05 | [5.46, 6.68] | 8.63 | [7.99, 9.34] |
| Qwen3-ASR-1.7B | 6.08 | [5.54, 6.66] | **8.59** | [7.93, 9.26] |
| deepgram-nova-3 (formatted) | 6.62 | [6.04, 7.26] | 10.26 | [9.51, 11.01] |
| whisper-large-v3 | 7.54 | [6.91, 8.22] | 10.15 | [9.40, 10.98] |

The load-bearing row is the last two. Under the leaderboard's normalizer,
Nova-3 with default formatting **beats** Whisper by 0.92 WER, and the paired
bootstrap says that is real (p<0.001). Under ours, the gap is +0.11 and
unresolvable (p=0.78). A buyer reading "Nova-3 beats Whisper" is reading a
fact about the normalizer as much as about the models.

### Which rule does the work

Mean |ΔWER| across systems from flipping one rule:

| Rule | Δ WER | What it does |
|---|---|---|
| `collapse_hyphenation` | 0.75 | "double-click" → "doubleclick" |
| `remove_fillers` | 0.63 | deletes um/uh/mm before scoring |
| `expand_contractions` | 0.19 | "don't" → "do not" |
| `spell_out_numbers` | 0.17 | digits ↔ words |
| `keep_apostrophes` | 0.07 | |
| `british_to_american` | 0.04 | |
| `normalize_currency` | 0.00 | no effect on this data |
| `strip_non_speech` | 0.00 | these refs carry no `[laughter]` markers |

Two rules account for most of it, and `remove_fillers` is the one to worry
about: **deleting disfluencies before scoring is a 0.63 WER subsidy**, and it
is not neutral between systems — it rewards models that don't emit fillers and
costs nothing to models that do. On spontaneous speech that is a meaningful
thumb on the scale, and it is invisible in the published number.

### Output formatting is a measurement artifact

Nova-3 twice, same model, same audio, differing only in API flags
(`punctuate`, `smart_format`, `numerals`):

| Normalizer | raw | formatted | Δ | p |
|---|---|---|---|---|
| leaderboard | 6.05 | 6.62 | **+0.56** | <0.001 |
| ours | 8.63 | 10.26 | **+1.63** | <0.001 |

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

### The metric I'd add: keyword error rate

**Definition.** Over reference occurrences of *rare content words* — words
appearing in ≤3 of the 1200 references, length ≥4 — the fraction the
hypothesis fails to produce. Rarity is measured in-set, so it is
casing-independent and works across all three reference conventions.
Vocabulary: 2,896 words, covering 1,019 of 1,200 utterances.

**Whose decision it informs.** A buyer choosing ASR for an application where
*which* words are wrong matters more than *how many*: contact-centre analytics
keyed on product and account names, medical or legal transcription, meeting
notes, voice agents doing entity capture. For them a WER tie is not a tie.

**Result** (leaderboard normalizer):

| System | WER | KER | KER 95% CI |
|---|---|---|---|
| deepgram-nova-3 (raw) | 6.05 | **5.91** | [5.06, 6.77] |
| deepgram-nova-3 (formatted) | 6.62 | 6.08 | [5.20, 6.97] |
| Qwen3-ASR-1.7B | 6.08 | 6.19 | [5.39, 7.03] |
| parakeet-tdt-0.6b-v2 | **6.04** | 6.46 | [5.61, 7.31] |
| whisper-large-v3 | 7.54 | 6.60 | [5.80, 7.49] |

The ordering changes. Parakeet is first on WER and fourth on KER; Nova-3 is
third on WER and first on KER; Whisper is last on WER but mid-pack on KER,
because its deletions fall disproportionately on common words.

**This reordering is not statistically resolved.** Every KER interval overlaps
every other. The honest claim is that KER *orders the systems differently from
WER*, that the difference is in a direction that matters commercially, and
that confirming it needs more data. Reporting it as "Nova-3 wins on keywords"
would be overreach.

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
- **The keyword metric's vocabulary is defined in-set.** Document frequency
  over 1,200 utterances is a noisy rarity estimate; a real deployment would
  define the keyword list from the customer's own domain (product catalogue,
  account names), which is both more accurate and more useful.
- **Injected noise is friendlier than real noise.** Additive white and babble
  is not reverberation or overlapping speech, which is what actually makes AMI
  hard. The recovery figures are an upper bound on estimator quality.
- **`spell_out_numbers` is imperfect.** `$5.50` becomes "five point five
  dollars" rather than "five fifty". It is applied identically to reference
  and hypothesis so it does not bias the comparison, but it is not how anyone
  speaks.
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
   three, because a normalizer change can move a ranking as much as a model
   change.
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

Claude (via Claude Code) was used heavily and is credited as co-author on the
commits.

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

**Two findings came from distrusting the AI-written code rather than reading
it.** The WADA-SNR implementation looked plausible and was wrong — it
saturated at the grid maximum for every utterance and produced a confident,
entirely spurious "metadata is unreliable" finding. The keyword metric's
original capitalization-based vocabulary also looked plausible and silently
excluded a third of the data, because GigaSpeech references are lowercase.
Both were caught by checking outputs against expectations, not by code review.
That is the main thing I'd carry into doing this work at scale: generated
analysis code fails in ways that produce *publishable-looking numbers*, so the
validation has to be adversarial and empirical.
