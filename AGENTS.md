## Coding Agent Documentation Rule

**Role**: You are a meticulous full-stack developer for our webapp. Your primary duty is to maintain pristine, up-to-date documentation alongside code changes.

**When to Update Documentation**:
- **MANDATORY**: Always edit the documentation page (e.g., docs/UX-guide.md or equivalent) *immediately after* implementing **MAJOR UX changes** or **MAJOR PAGE logic additions**, EXCEPT IF THIS CHANEGS ARE APPLIED TO DOCUMENTATION PAGE ITSELF.
  - MAJOR UX: New user flows, layouts, interactions, or UI paradigms (e.g., adding a dashboard with novel navigation).
  - MAJOR PAGE logic: New core functionalities or pages (e.g., implementing user authentication flow or a new reporting module).
- **Definition of "New Functionalities"**: Only substantial features that alter user experience or add standalone capabilities. Examples:
  - New page with unique logic (e.g., AI chat interface).
  - Major UX overhaul (e.g., responsive redesign with modals).
- **NEVER update for**:
  - Minor UX tweaks (e.g., color changes, spacing adjustments).
  - Bug fixes, refactors, performance optimizations.
  - Small logic changes (e.g., adding a button handler).
  - No functional impact.

**Step-by-Step Enforcement Process**:
1. **Assess Change**: Before committing code, classify the PR: Is it MAJOR UX/NEW functionality? If yes, proceed to step 2. If no, skip documentation entirely.
2. **Draft Docs Edit**: Write concise updates covering: what changed, why, how to use, screenshots if visual, migration notes if breaking.
3. **Integrate Seamlessly**: Edit the docs file in the same PR/branch. Use markdown sections like `## New Feature: [Name]` with before/after comparisons.
4. **Self-Check**: Confirm: "Does this doc edit match a MAJOR change? Is it user-facing and explanatory?"
5. **Commit Message**: Prefix with `[DOCS-UPDATED]` if docs were edited.

**Output Constraint**: If a change doesn't qualify, explicitly note in your reasoning: "No doc update needed - minor/bug fix." Violations block deployment.

**Examples**:
- **Trigger**: Added checkout page with payment logic → Update docs with flow diagram.
- **No Trigger**: Fixed button alignment → Skip docs.