# Working Model Downloader

A ComfyUI custom node for getting a workflow's models onto disk — driven by the
links the workflow's author already wrote down.

You download a third-party workflow, open it, and half the loaders are red. This
extension reads the workflow's own documentation — the Note and MarkdownNote nodes
where authors paste their download links — and pairs those links with the loader
slots that are actually missing. Sort it out once in the sidebar; the result is
pinned into a single node, so headless runs, API calls, and re-opening the
workflow on a new machine all reproduce exactly the same downloads.

## Why another one

There are already several model downloaders for ComfyUI, and the good ones are
good. Every one of them, though, starts from a *filename*: it reads the loader
widgets (and `node.properties.models`), then **searches** HuggingFace or Civitai
for something with that name. Searching is the part that goes wrong — it finds a
mirror, a differently-quantised copy, or nothing at all.

Meanwhile, most shared workflows say exactly where their models come from, in a
note, right there on the canvas. Nothing else reads them.

That is measurable, so it has been measured. Across **612 real workflows** pulled
from Civitai's Workflows category (`python tools/corpus.py`):

| in the workflow | share |
|---|---|
| has a Note or MarkdownNote node | **70%** |
| **documents its models with links** — what this reads | **52%** |
| has `properties.models` — what the others read | 17% |

Documented links are present in three times as many workflows as the metadata
every other downloader depends on.

So this extension differs in two ways:

1. **Documented links come first.** URLs in Note and MarkdownNote text are
   extracted and joined to the missing loader slots by filename. That is an exact
   answer, not a search result. Searching is still available, as a clearly-labelled
   last resort.
2. **The panel's result survives into the workflow.** Other tools are panel-only,
   so nothing they figured out is available on a headless run. Here the panel
   writes a pinned manifest into one node, and that node makes it true again
   wherever the workflow is opened next.

## Install

Clone into `ComfyUI/custom_nodes/` and restart:

```bash
git clone https://github.com/jmtoball/comfyui-working-model-downloader.git
```

The only dependency is `requests`, which ComfyUI already has.

## Using it

**1. Open the *Model Downloader* sidebar tab and press *Scan workflow*.**
You get one row per model, showing where it will go and *why* — colour-coded by
how much the answer can be trusted:

| | source | meaning |
|---|---|---|
| 🟢 | documented / properties | a link the workflow itself provides |
| 🔵 | manifest / rule / manual | pinned, a rule you taught it, or your own choice |
| 🟡 | search | an exact filename match found by searching — verify it |
| 🔴 | unresolved | it will not guess; choose a folder or paste a link |

**2. Fix anything that needs fixing.** Change a folder, rename a file, pick between
search candidates, or paste a URL for something nothing could find. A URL you paste
is remembered as a rule, so the next workflow needing that file resolves itself.

**3. *Download selected*,** then **4. *Save to workflow*.**

Step 4 creates (or updates) a single **Working Model Downloader** node holding the
pinned manifest. Save the workflow and you are done: queueing that workflow
anywhere — the UI, `POST /prompt`, a headless worker — re-fetches anything absent
before the graph runs.

### API keys

Read from the environment by default (`HF_TOKEN` or a `hf` CLI login;
`CIVITAI_API_KEY`), and overridable at runtime under **API keys** in the panel.
Panel-set keys are stored in ComfyUI's user directory with mode `0600` — never in
the extension folder, and never in the workflow. Civitai keys are sent as an
`Authorization` header rather than a `?token=` query parameter, so they cannot leak
through a log line or a referrer.

## The node

One node, five widgets. It carries configuration and executes it; it does not scan
or guess — that is the panel's job.

| widget | what it does |
|---|---|
| `manifest` | the pinned JSON. Written by the panel, hand-editable, diffable |
| `enforce` | `before_execution` (default) or `on_node_execution` — see below |
| `on_failure` | `error` stops the run; `warn` logs and continues |
| `verify_hash` | check the pinned sha256 before the file is moved into place |
| `queue_timeout` | seconds to wait at queue time; `0` waits as long as it takes |

### How it guarantees the files exist first

ComfyUI gives an `OUTPUT_NODE` no ordering guarantee against a loader, so
downloading inside `execute()` is not enough — a `CheckpointLoader` may already
have run and failed. In `before_execution` mode the extension uses two hooks:

- `PromptServer.add_on_prompt_handler` fires inside `POST /prompt`, before
  validation and queueing: the downloads start there, without blocking.
- `VALIDATE_INPUTS` is awaited for every node before *any* node executes: that is
  where it waits, and where a failure becomes a validation error naming the file,
  instead of a confusing loader crash later.

The cost is that `POST /prompt` stays open until the transfer finishes — which is
what you want for headless provisioning. `on_node_execution` is the alternative:
it downloads when the node runs, with a proper progress bar, but only orders nodes
downstream of the node's `passthrough` output.

### Manifest format

```json
{
  "version": 1,
  "entries": [
    {
      "url": "https://huggingface.co/…/resolve/main/flux1-dev.safetensors",
      "filename": "flux1-dev.safetensors",
      "folder": "diffusion_models",
      "provider": "huggingface",
      "sha256": "…",
      "size": 23802932552,
      "slot": {"node": "12", "input": "unet_name"},
      "origin": "note", "tier": "documented"
    }
  ]
}
```

Applying a manifest performs no resolution, no search and no API calls beyond the
download itself. Credentials are never written into it.

## How a destination is decided

In order; the first that answers wins, and the answer is shown with its reason:

1. **The loader's own combo.** A node's model input draws its options from a
   `folder_paths` folder, and that folder is where ComfyUI will look for the file.
   Derived by evaluating each node class's `INPUT_TYPES()`, so third-party loaders
   work too.
2. **A rule you taught it** (`rules.json` in the user directory).
3. **`properties.models[].directory`** — ComfyUI's own embedded metadata.
4. **Provider metadata** — Civitai's `model.type`; a HuggingFace repo's layout
   (including the `split_files/<folder>/` convention the `Comfy-Org` mirrors use),
   tags, and failing that its name.
5. **The filename** — `*.vae.safetensors`, `control_*`, `t5xxl_*`, `learned_embeds`,
   and so on.
6. **Nothing.** A file with no signal is reported as unresolved rather than dropped
   into `checkpoints` and forgotten. A wrong folder is worse than a question.

Only folders this ComfyUI actually has are offered, and `extra_model_paths.yaml`
roots are respected — both when choosing where to write and when checking whether
a model is already present somewhere.

## Downloading

Resumable (`.part` + HTTP `Range`), atomic (`os.replace` only after verification),
checksum-verified where the source publishes a hash, and free-space checked before
it starts. Beyond that, it refuses to write things that are not models:

- An **error page saved as a `.safetensors`** is the classic failure in this
  space — an auth redirect gets written under the model's name and the download
  "succeeds". Bodies are judged on their actual bytes, not just their declared
  content type. An unauthenticated Civitai download redirecting to `/login` is
  reported as "Civitai API key required".
- A **`200` answering a `Range` request** means the server ignored the range;
  appending would corrupt the file, so it restarts.
- **`416`** on a resume means the `.part` is already complete.
- A **stale HuggingFace Xet signature** (a 403 from the CDN) is refreshed once from
  the stable Hub URL.
- **Credentials are dropped on any cross-host redirect**, because presigned object
  storage rejects a request that also carries an `Authorization` header.

## Command line

The core has no ComfyUI dependency, so all of it works standalone:

```bash
python -m wmd.cli scan workflow.json          # what the workflow says about its models
python -m wmd.cli resolve <url> [--search]    # where each file would go, and why
python -m wmd.cli download <url> [--dry-run]  # --dry-run prints a manifest
python -m wmd.cli apply manifest.json         # the headless path, exactly as the node runs it
```

Set `WMD_MODELS_DIR` to point it at a `models/` tree when ComfyUI is not importable.

## Development

```bash
pip install requests aiohttp pytest responses ruff
pytest tests
ruff check .
```

`wmd/` never imports ComfyUI — `wmd/comfy_env.py` is the single seam where the real
`folder_paths` is used when available — so the whole core is testable without it.
The tests supply a stand-in `folder_paths` and node registry instead.

### Testing against real workflows

Unit tests prove the logic does what it should; `tools/corpus.py` proves it survives
what people actually publish:

```bash
python tools/corpus.py fetch      # real workflows from Civitai -> .corpus/ (gitignored)
python tools/corpus.py scan       # what the scanner finds in them
python tools/corpus.py coverage   # how often a destination can be named
```

Set `CIVITAI_API_KEY` to reach the workflows whose authors require a login — about
two thirds of them. The corpus is other people's work, so it is fetched on demand
rather than committed; several of the filename heuristics above were derived from
it, and `coverage` is how to check a change to them actually helps.

## Prior art

The other downloaders in this space are worth knowing about, and reading them
saved this one from repeating several mistakes:
[ComfyUI-ModelResolver](https://github.com/21omen/ComfyUI-ModelResolver),
[ComfyUI-Missing-Models-Fetcher](https://github.com/Adomess/ComfyUI-Missing-Models-Fetcher),
[comfyui-ez-dl](https://github.com/fuselayer/comfyui-ez-dl),
[ComfyUI-advanced-model-manager](https://github.com/BISAM20/ComfyUI-advanced-model-manager),
[comfyui-tiny-model-manager](https://github.com/Zellione/comfyui-tiny-model-manager),
[bs-comfyui-model-manager](https://github.com/juangea/bs-comfyui-model-manager),
[CivitaiManager](https://github.com/nregret/CivitaiManager).

The technique of identifying a model slot by matching a node's combo options
against a `folder_paths` file list is ComfyUI-ModelResolver's (MIT) good idea;
it is reimplemented here rather than copied. Several of the download guards above
exist because those projects hit the failures first.

## Licence

MIT.
