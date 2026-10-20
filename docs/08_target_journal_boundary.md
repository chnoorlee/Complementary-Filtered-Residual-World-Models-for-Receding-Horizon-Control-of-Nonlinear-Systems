# Target-journal boundary

## Working candidate and official scope check

The working candidate is *Journal of Intelligent & Robotic Systems* (JIRS,
Springer Nature). Its official journal page was fetched on 2026-08-05 from
<https://link.springer.com/journal/10846>. The page describes a peer-reviewed
journal spanning the theory and practice of intelligent systems and robotics,
including systems modelling, simulation, control, signal estimation, robotics,
and intelligent control. The present action-conditioned physical world-model
study is provisionally routed to journal Category (2),
“Systems Modelling/Simulation/Control/Computer-Aided Design.” Scope fit remains
an editorial judgment rather than a guarantee.

The same official page displayed a 2025 Journal Impact Factor of 3.1 and a
five-year value of 3.4. Impact factor is not a quartile and is not used here as
evidence that the journal is Q2.

## Author-instruction check

The official instructions at
<https://link.springer.com/journal/10846/submission-guidelines> were fetched on
2026-08-05 and recorded the following relevant requirements:

- single-blind peer review;
- an abstract of 150--250 words;
- four to six indexing keywords;
- a category number on the first page;
- an appropriate number of MSC classification codes;
- Springer Nature's LaTeX template is recommended, with editable sources,
  style files, figures, and a compiled PDF supplied at submission/revision;
- standardized declarations, including data availability and competing
  interests, where applicable.

The official December 2024 Springer Nature journal-article template was
downloaded from the publisher's author-support page and unpacked under
`paper/springer-official/sn-article-template/`. The downloaded ZIP SHA-256 was
`812e76dcaa9c28dc1bff1fb6065d51729b67d4ea140552a05088317414a3ecae`.
`code/build_springer_submission.py` flattens the audited working sources into a
single Springer-formatted TeX document after locked results exist. Author,
affiliation, funding, contribution, repository DOI, and final AI-disclosure
metadata remain explicit author-supplied fields and must not be invented.

## Mandatory Q2 gate

Current Clarivate Journal Citation Reports and Chinese Academy of Sciences
quartiles were not accessible from an authoritative institutional subscription
in this workspace. Before submission, the corresponding author must:

1. state whether “SCI 2区” means JCR Q2, CAS 2区, or both;
2. verify the journal's current category and quartile in the relevant
   institutional database on the actual submission date;
3. confirm the final scope and article type with the current author
   instructions; and
4. reroute to another verified journal if the required quartile or scope does
   not match.

The package targets Q2-level review quality. It does not claim guaranteed
acceptance or confirmed-current Q2 status.
