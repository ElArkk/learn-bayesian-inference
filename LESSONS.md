# Project lessons

## Marimo live state, saved source, and restarts

### What happened

After the Marimo server restarted, the course looked as if most of its prose was
gone. The `COURSE_PROSE` data was still present in `inference_lab.py`. An older
version of the shared rendering cell had become active and was saved again. That
renderer did not display the expanded chapter sections.

The restart did not delete the 24 prose records. The visible result was wrong
because the prose data and the code that rendered it were from different
versions.

### Why this can happen

A live Marimo notebook has several related states:

1. The Python file on disk.
2. The code-mode notebook document used by the editor.
3. The runtime dependency graph and Python objects.
4. The browser editor state and its pending autosave work.

These states can differ for a short time. The active Marimo runtime owns the
notebook while the editor is open. A direct edit to the `.py` file does not
automatically replace the active runtime state. A later browser save can write
the older active cell back to disk.

Multiple browser tabs or old editor sessions increase this risk. A server
restart also creates a new runtime, so an old page can reconnect with stale
editor state. Restarting is not a safe way to merge live and disk changes.

### Reproduced refresh-overwrite failure

This failure later occurred again without a direct edit to `inference_lab.py`.
The paired agent changed Lab 11 through `marimo._code_mode`. The runtime graph
contained the new `one_hmc_transition` exercise, while the browser-facing
code-mode document still contained the old `hmc_momentum_step` exercise. The
agent copied the verified runtime cells into the document, confirmed that the
document and runtime matched, confirmed that the Python file contained the new
code, and passed the clean test suite.

When the learner refreshed the existing browser page, that page restored its
older editor document. Marimo then autosaved the old document into
`inference_lab.py`. This changed Lab 11 back to `hmc_momentum_step` and also
restored older versions of some hidden infrastructure cells. The prose output
could still look new because the runtime graph and cached outputs had not all
changed together.

This happened more than once. Therefore, a normal refresh of a stale Marimo
editor page can overwrite newer code-mode edits even after runtime-to-document
and document-to-disk equality were briefly confirmed. This is not only the
usual warning that direct file edits can be overwritten by a live notebook.
The browser editor document itself can be the stale writer.

### Required working rule

Use only one browser tab and one active editor session for a notebook.

While that editor is active:

- Read and change notebook cells through the active Marimo session.
- Use `marimo._code_mode` for durable paired edits.
- Do not edit `inference_lab.py` directly with a second process.
- Do not start another editor or open another notebook tab to refresh the view.
- Do not restart the server only to make a change appear.

After each material edit:

1. Run the changed cell in the active session.
2. Inspect live cell errors and the dependent output.
3. Confirm that the active cell source contains the new code.
4. Confirm that `inference_lab.py` contains the same code.
5. Run `marimo check --strict`, tests, and Ruff.
6. Before a planned restart, confirm again that runtime and disk source match.
7. After the restart, verify one chapter with a large known prose marker before
   doing more work.

### Recovery procedure

If content appears to vanish after a restart:

1. Do not open more tabs and do not restart again.
2. Check whether the content data still exists on disk.
3. Compare the active rendering cell with the saved rendering cell.
4. Restore the verified newer cell through the active Marimo session.
5. Run that cell and its dependent cells.
6. Confirm both the visible output and the saved `.py` file.
7. Run the complete clean-process checks.

If refreshing the same page restores the old document again, stop using that
browser/server session. Use this stronger recovery:

1. Save learner UI values and preserve all visible learner exercise cells.
2. Copy verified runtime code into the code-mode document for the affected
   cells and hidden infrastructure cells.
3. Confirm the repaired Python file and run the complete test suite.
4. Stop only the stale notebook server before the browser can autosave again.
5. Confirm the repaired file while that server is stopped.
6. Start one clean server on a new port, without opening another tab.
7. Navigate the existing tab to the new port. A new origin prevents the old
   browser editor state from being reused.
8. Verify the visible learner cell, runtime cell, and saved source again.

Do not rewrite large content blocks until this comparison proves that the data
is actually absent.

### Bug classification

Do not describe this incident as proven Marimo data deletion. The data was not
deleted.

The behavior is consistent with a known Marimo ownership and synchronization
limitation: the live runtime can overwrite direct file changes, and external
edits do not always move into a live notebook without explicit synchronization.
Marimo has a file watcher setting, but it controls whether changed cells are
marked stale or run automatically. It is not a general conflict-resolution
system for several editor states.

The first incident included session-management mistakes: the agent opened too
many notebook views and did not prove runtime-to-disk equality before restart.
The later Lab 11 incident is stronger evidence. It reproduced a stale browser
document overwriting newer code-mode and disk source after a refresh, even
after those sources had been checked. Treat this as a Marimo synchronization
defect or limitation until a smaller reproduction identifies the exact layer.

A focused bug report should include the Marimo version, autosave settings,
exact tab count, server port, cell ID, browser-document source, runtime-graph
source, file source before refresh, and all three sources after refresh. The
Lab 11 `hmc_momentum_step` to `one_hmc_transition` change is a useful concrete
reproduction because the two versions are easy to distinguish.

### References

- Marimo Pair warns that the active runtime is the source of truth and can
  overwrite direct file edits:
  <https://github.com/marimo-team/marimo-pair/blob/main/skills/marimo-pair/SKILL.md>
- Marimo issue 8955 describes current friction when external tools change a
  notebook file:
  <https://github.com/marimo-team/marimo/issues/8955>
- Marimo's runtime configuration defines `watcher_on_save` as stale-or-autorun
  behavior for file changes:
  <https://github.com/marimo-team/marimo/blob/main/marimo/_config/config.py>
