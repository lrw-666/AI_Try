# SegFormer Experiment Agent Rules

## Scope and current status

This directory is for the VisA PCB1 high-shot supervised SegFormer-B0 baseline. Follow the user's current stage request and the two guides below. Do not claim that an environment, training run, or metric exists without inspecting actual evidence.

- `../../docs/research/2026-10-31-segformer-baseline-end-to-end-guide.md`
- `../../docs/research/2026-10-31-segformer-environment-isolation-and-stage-docs.md`

The user specifically requests Chinese stage design and implementation documents for learning. Use English filenames and Chinese prose for those documents and the stage index. This is an explicit task-specific exception to the root default Markdown language rule.

## Environment and resource isolation

- Default project resource root: `E:\桌面\AI\segformer_resources`.
- Use a dedicated Conda prefix at `<resource-root>\env`; keep environment packages, caches, temporary files, extracted data, downloaded weights, and runs under the resource root as specified in the guide.
- Do not install, uninstall, or upgrade packages in base or another project's environment. Do not modify global Conda/pip configuration, system PATH, shell profiles, or persistent environment variables.
- Inventory existing prefixes and configuration before setup. Reuse only after verification; do not overwrite or delete an existing environment or user data.
- Each fresh tool process must establish the project runtime context. Use the project runner or an equivalent explicit `conda run --prefix ...` setup. A selected Python executable alone does not establish all cache and Windows runtime settings.
- Before environment creation, check effective Conda package-cache directories. Abort setup if shared caches remain merged. Apply project-process environment-registration/notices suppression and use `--no-shortcuts`; use the exact lowercase Windows pip null-device value from the guide and clear inherited legacy Transformers cache overrides.
- Record actual interpreter, package versions, cache/output paths, and relevant before/after comparisons. Shared OS, GPU driver, Conda manager, and possible environment registration metadata are outside Python package isolation; do not promise VM-level isolation.

## Mandatory stage documents

For every numbered stage performed, create or update one Chinese Markdown design and implementation document in `docs/stages/`, using the filenames in the companion guide. If a task covers stages 1 and 2, produce both stage documents.

Use `../../docs/research/templates/segformer-stage-design-and-implementation-template.md`. Cover stage intent, design choices, actual stack and versions, changed files and interfaces, key implementation details with code excerpts, reproducible commands, real validation evidence, limitations, fixes, learning questions, and next dependencies.

Maintain `docs/stages/README.md` with the stage status, document link, evidence locations, and next action. Create the index when stage 1 begins; do not pre-fill future stages as completed. Update an existing stage document on rework and preserve a concise revision history.

A stage is not fully delivered until its documentation and index entry reflect the actual implementation and verification status. If blocked, document what was done and what remains blocked. Chat summaries do not replace these files. No additional approval gate is introduced by this rule; continue the work already authorized by the user's stage request.

## Evidence and research protocol

Keep original-image split integrity, binary-label semantics, padding ignore labels, original-resolution evaluation, and train/validation/test separation from the guide. Do not fabricate logs, plots, benchmark scores, or completed milestones. Distinguish proposed code, implemented code, executed commands, and verified results in every report.
