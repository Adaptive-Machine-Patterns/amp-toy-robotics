# Provenance

Where the code in this repository comes from, for anyone checking that it
is safe to use.

- **Authorship.** Written for Adaptive Machine Patterns Ltd by Andy Whyte,
  with AI coding assistants: OpenAI ChatGPT (an early version of the arm
  geometry), Qwen 3.6 and 3.8 (first version of
  the GUI), and Anthropic Claude (polish, GUI development and production).
  The output of these assistants belongs to the user under their
  providers' terms and model licences. All of it was reviewed and is
  published under Apache-2.0 by the copyright holder.
- **No third-party source code is included.** Before publication, the
  code was searched for licence headers, attribution comments and
  verbatim copies of the libraries it was developed alongside (numpy,
  Pillow, PyTorch, lerobot, transformers, diffusers, Vosk, Piper). None
  were found in this package.
- **`geometry.py`** was rewritten for publication from the original
  project's renderer. It keeps only the arm and object geometry, with the
  arithmetic in the same order so that every seed gives the same scene as
  before (`tests/test_parity.py`).
- **The object colours** are ColorBrewer's Set2 palette (see NOTICE).
- **The task, the teacher and the training-data generator** come from
  AMP's original project (`robot-control-gui`, commit 3f4197c). The
  grounding tests come from that project's working tree on 2026-10-14,
  with two tests added (instruction swap, blank instruction).
- **The joint speed limit** (0.2.0) is that project's commit b3bec5b, line
  for line; the parity tests pass against it.
