# Parser and scoring behavior is unpinned until re-baseline

Extraction and scoring behavior is design-in-progress: it is being
hand-tuned, and it changes freely under the working profile IDs. There
are deliberately no byte-identity fixtures or pinned corpus baselines in
the test suite — do not add them. Tests express *intended* behavior and
are edited when intent changes; their job is catching accidental changes,
not enshrining history. The corruption-corpus generator exists to make
deterministic tuning datasets on demand (recipes + seeds), not to define
a canonical dataset.

The versioning discipline snaps back into force at re-baseline: the
moment experiment outcomes start being recorded against a profile ID,
that ID freezes, and behavior changes require a new version. Until then,
recorded results never constrain behavior or schemas — they are archived
(moved aside, retained, referenced by no code), not deleted and not
carried forward.
