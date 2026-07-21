# PyGeoFetch — JOSS Submission Guide

Complete checklist and step-by-step instructions for submitting to the
Journal of Open Source Software (JOSS).

---

## Pre-submission checklist

Work through every item before opening a submission at joss.theoj.org.

### Repository requirements
- [ ] Repository is **public on GitHub** (not private, not a fork)
- [ ] Repository has been public for **more than 6 months** with active development
      spanning that period (JOSS will desk-reject recent public repos)
- [ ] Repository has an **OSI-approved open source license** (`LICENSE` file — MIT ✓)
- [ ] `README.md` explains what the software does and how to install it
- [ ] Installation instructions work from a clean environment
- [ ] `pyproject.toml` is present and correct

### Software quality requirements
- [ ] Software is **feature-complete** (no half-baked solutions)
- [ ] **Automated tests** are present and passing (`pytest tests/` → 70 tests pass)
- [ ] Documentation exists (docstrings, notebooks, readthedocs)
- [ ] Package installs via `pip install pygeofetch` from PyPI
- [ ] CLI entry point `PyGeoFetch --version` works
- [ ] `PyGeoFetch doctor` completes without error

### Paper file requirements
- [ ] `paper/paper.md` exists in the repository
- [ ] `paper/paper.bib` exists in the repository
- [ ] Paper compiles to PDF without errors (test with Docker or GitHub Action)
- [ ] Word count is between 750 and 1750 words (`wc -w paper/paper.md`)
- [ ] All required sections present:
  - [ ] Summary
  - [ ] Statement of need
  - [ ] State of the field
  - [ ] Software design
  - [ ] Research impact statement
  - [ ] AI usage disclosure
  - [ ] Acknowledgements
  - [ ] References
- [ ] All citations in `paper.md` have a matching entry in `paper.bib`
- [ ] Author ORCID iDs are correct (register free at https://orcid.org)
- [ ] Affiliations are accurate

---

## Step 1 — Compile the paper locally

Test that `paper.md` compiles to PDF before submitting.

**Option A — Docker (recommended):**
```bash
# From the root of your repository
docker run --rm \
    --volume $PWD/paper:/data \
    --user $(id -u):$(id -g) \
    --env JOURNAL=joss \
    openjournals/inara

# Output: paper/paper.pdf
```

**Option B — GitHub Actions (automatic):**
Push any change to `paper/` and the workflow at
`.github/workflows/draft-pdf.yml` will compile the PDF.
Download it from the Actions tab → latest run → Artifacts.

**Option C — Online preview:**
Go to https://whedon.theoj.org and paste your `paper.md`.

---

## Step 2 — Verify all citations resolve

```bash
# Every @key in paper.md must appear in paper.bib
python3 << 'PYEOF'
import re

paper = open("paper/paper.md").read()
bib   = open("paper/paper.bib").read()

cited  = set(re.findall(r'@(\w+)', paper))
defined = set(re.findall(r'^@\w+\{(\w+)', bib, re.MULTILINE))

missing = cited - defined
if missing:
    print(f"MISSING from paper.bib: {missing}")
else:
    print(f"All {len(cited)} citations resolved correctly")
PYEOF
```

---

## Step 3 — Update your ORCID

In `paper/paper.md`, replace the placeholder ORCID:
```yaml
authors:
  - name: Kubis Appiah
    orcid: 0000-0000-0000-0000   # ← replace with your real ORCID
    affiliation: "1"
```

Register for a free ORCID at https://orcid.org/register

---

## Step 4 — Submit at joss.theoj.org

1. Go to **https://joss.theoj.org/papers/new**
2. Sign in with your GitHub account
3. Fill in the form:

| Field | Value |
|-------|-------|
| Repository URL | `https://github.com/YOUR_USERNAME/PyGeoFetch` |
| Branch | `main` |
| Software version | `v1.3.0` |
| Programming language | `Python` |
| Software categories | `Earth Sciences, Remote Sensing, Geospatial` |
| Submission type | `Original Software Publication` |

4. Click **Start Review**

JOSS will create a public review issue at
`https://github.com/openjournals/joss-reviews/issues/NNNN`

---

## Step 5 — The review process

JOSS reviews are **open and public** on GitHub.

**Timeline:**
- Editor assignment: 1–2 weeks
- Review: 2–8 weeks (two reviewers minimum)
- Revision and acceptance: 1–4 weeks additional

**What reviewers check** (from the JOSS review checklist):
- [ ] Software has an OSI-approved license
- [ ] Paper has required sections (Summary, Statement of Need, etc.)
- [ ] Software has clear installation instructions
- [ ] Software has automated tests
- [ ] Tests pass in a clean environment
- [ ] Documentation is sufficient
- [ ] Statement of need is convincing
- [ ] Software makes a significant contribution

**Responding to reviewers:**
- All responses happen in the public GitHub issue
- Address each reviewer comment directly
- Use the `@editorialbot` commands when ready:
  ```
  @editorialbot check references
  @editorialbot generate pdf
  @editorialbot set v1.3.0 as version
  ```

---

## Step 6 — After acceptance

Once the paper is accepted:
1. JOSS mints a **CrossRef DOI** (e.g., `10.21105/joss.NNNNN`)
2. The paper appears at `https://joss.theoj.org/papers/10.21105/joss.NNNNN`
3. Add the DOI badge to your README:
   ```markdown
   [![DOI](https://joss.theoj.org/papers/10.21105/joss.NNNNN/status.svg)](https://doi.org/10.21105/joss.NNNNN)
   ```
4. Citation for users:
   ```
   Appiah, K. (2026). PyGeoFetch: A Unified Python Framework for
   Multi-Provider Satellite Data Acquisition and Geospatial Processing.
   Journal of Open Source Software, XX(YY), NNNNN.
   https://doi.org/10.21105/joss.NNNNN
   ```

---

## Common desk rejection reasons (avoid these)

| Reason | How PyGeoFetch avoids it |
|--------|--------------------------|
| Thin API wrapper | PyGeoFetch integrates 22 providers + processing pipeline + orbit files |
| No tests | 70 integration contract tests, all passing |
| No documentation | Notebooks, README, docstrings, readthedocs |
| Repo too new | Ensure 6+ months of public development history |
| Missing paper sections | All 8 required sections present |
| Word count outside 750–1750 | Current: ~1,450 words |
| Citations not in bib file | All 25 citations resolved |
| Paper focuses on results, not software | Paper describes design, not research findings |

---

## Files checklist

```
PyGeoFetch/
├── paper/
│   ├── paper.md              ✓  JOSS manuscript
│   └── paper.bib             ✓  BibTeX references (25 entries)
├── .github/
│   └── workflows/
│       └── draft-pdf.yml     ✓  Auto PDF compilation
├── pygeofetch/               ✓  Package source
├── tests/                    ✓  70 contract tests
├── notebooks/                ✓  9 example notebooks
├── README.md                 ✓  Installation + quickstart
├── LICENSE                   ✓  MIT
├── pyproject.toml            ✓  Package metadata
└── CHANGELOG.md              ✓  Release history
```
