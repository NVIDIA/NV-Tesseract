# Contributing to NV-Tesseract

Thank you for your interest in this project.

## Issues

We track bugs, feature requests, and questions in **[GitHub Issues](https://github.com/NVIDIA/NV-Tesseract/issues)**.

1. **Search first** — Check open and recently closed issues for duplicates before opening a new one.
2. **Security** — Do **not** file security vulnerabilities as public issues. Follow [`SECURITY.md`](SECURITY.md) instead.
3. **Choose a clear title** — Summarize the problem or request in a few words (for example, “Forecasting: error when `context_df` has extra columns”).
4. **Describe the context** — In the body, include at least:
   - **Area:** `forecasting`, `ad_diffusion`, or other (for example, `scripts`, `CI`, documentation).
   - **Environment:** OS, Python version, and how you installed the package (`uv`, `pip`, editable install, and which subproject: `forecasting` or `ad_diffusion`).
   - **What you expected** vs **what happened** (for bugs), or **use case and proposed behavior** (for features).
5. **Reproducible bugs** — For defects, add a **minimal** code snippet, sample data (or steps to generate it), and the full error message or traceback. If the issue is version-specific, state the package and dependency versions (for example, from `uv pip freeze` in your environment).
6. **Link from pull requests** — If a change fixes or implements an issue, reference it in the PR description with `Fixes #123` or `Closes #123` (or `Refs #123` for partial work) so the record stays connected.

Maintainers may ask for more detail, a smaller reproducer, or a quick test under a different OS or Python version.

## Branch and pull request workflow

The active development branch is **`dev`**. Open pull requests against `dev` unless a maintainer explicitly asks you to target another branch.

The `main` branch is protected and reserved for release-ready code. Changes should reach `main` through maintainer-managed promotion from `dev`, not through regular contributor pull requests.

Before opening a pull request:

1. Branch from the latest `dev`.
2. Keep the change focused on one issue or feature.
3. Include tests or documentation updates when behavior changes.
4. Reference the related issue in the pull request description when one exists.

## Licensing

**License for contributions:** This project **only** accepts contributions that you license under the **Apache License, Version 2.0**. Do not submit code under any other license terms (including proprietary, copyleft-only, or incompatible licenses). By opening a pull request or otherwise contributing, you agree that your contributions will be distributed under Apache-2.0 on the same terms as the rest of this repository (see [`LICENSE`](LICENSE) at the repository root).

That obligation is separate from the **Developer Certificate of Origin** below: every commit must still include a valid `Signed-off-by` line certifying the DCO.

Third-party notices and PyPI dependency summaries are in [`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md). **Verbatim upstream license files** for source included in this repo live under [`third_party/`](third_party/) (see [`third_party/README.md`](third_party/README.md)). Attribution required by the Apache License for redistributed third-party components appears in [`NOTICE`](NOTICE) and in [`LICENSE`](LICENSE) under **ADDITIONAL THIRD-PARTY LICENSES**.

### Source file headers

- **NVIDIA-authored** Python files use SPDX short-form tags (`SPDX-FileCopyrightText`, `SPDX-License-Identifier`) per SPDX Specification v2.3, Annex E. Use the copyright attribution **`Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES. All rights reserved.`** (see [`scripts/add_spdx_headers.py`](scripts/add_spdx_headers.py) for the shared template).
- Files that include third-party or modified third-party code must retain **upstream copyright and license notices** in the file (SPDX alone is not a substitute where upstream terms require full notice).

You can refresh SPDX headers on Python trees with:

```bash
make spdx
```

Verify headers before submitting:

```bash
make spdx-check
```

## Intellectual property review (NVIDIA contributors)

If you are contributing as part of your work at NVIDIA, follow your organization’s **IP Review Process for Open Source** (internal documentation on Confluence: *IP Review Process — Open Source*) before submitting changes intended for external release. Ensure your contributions are cleared for distribution under Apache-2.0 and do not incorporate material you are not entitled to license.

External contributors should ensure they have the rights to submit their changes under Apache-2.0 and that third-party code meets the notice requirements above.

## Developer Certificate of Origin

We use the **Developer Certificate of Origin (DCO)** for contribution approval.

**Link to DCO:** [Developer Certificate of Origin](https://developercertificate.org/)

#### Signing Your Work

- We require that all contributors “sign-off” on their commits. This certifies that the contribution is your original work, or you have rights to submit it under the same license, or a compatible license.
  - Any contribution which contains commits that are not Signed-Off will not be accepted.
- To sign off on a commit you simply use the `--signoff` (or `-s`) option when committing your changes:

  ```bash
  $ git commit -s -m "Add cool feature."
  ```

  This will append the following to your commit message:

  ```text
  Signed-off-by: Your Name <your.email@example.com>
  ```

- Full text of the DCO (<https://developercertificate.org/>):

  ```text
  Developer Certificate of Origin
  Version 1.1
  Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
  Everyone is permitted to copy and distribute verbatim copies of this
  license document, but changing it is not allowed.
  Developer's Certificate of Origin 1.1
  By making a contribution to this project, I certify that:
  (a) The contribution was created in whole or in part by me and I
      have the right to submit it under the open source license
      indicated in the file; or
  (b) The contribution is based upon previous work that, to the best
      of my knowledge, is covered under an appropriate open source
      license and I have the right under that license to submit that
      work with modifications, whether created in whole or in part
      by me, under the same open source license (unless I am
      permitted to submit under a different license), as indicated
      in the file; or
  (c) The contribution was provided directly to me by some other
      person who certified (a), (b) or (c) and I have not modified
      it.
  (d) I understand and agree that this project and the contribution
      are public and that a record of the contribution (including all
      personal information I submit with it, including my sign-off) is
      maintained indefinitely and may be redistributed consistent with
      this project or the open source license(s) involved.
  ```

Contributions consisting of commits without a valid sign-off cannot be merged.

## Development workflow

- Install dependencies with **[uv](https://docs.astral.sh/uv/)** per the top-level [`README.md`](README.md).
- Lint and format (Ruff): `make lint` / `make lint-fix` from the repository root.
- Forecasting tests (example): `cd forecasting && uv run pytest sdk/tests`.

Open a pull request against `dev` with a clear description of the change and any relevant issue references.
