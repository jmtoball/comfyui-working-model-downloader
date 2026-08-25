// Stub of ComfyUI's api module returning fixture data shaped like the real routes.
const FOLDERS = ["checkpoints", "loras", "vae", "text_encoders", "diffusion_models",
  "clip_vision", "controlnet", "upscale_models", "embeddings", "frame_interpolation"];

const REASONS = [
  "UNETLoader reads unet_name from diffusion_models",
  "the workflow declares text_encoders",
  "the repo stores it under text_encoders/",
  "VAELoader reads vae_name from vae",
  "the filename looks like a frame_interpolation entry",
  "no rule, metadata or filename signal identified this",
];

function items(count) {
  if (window.__noneMissing) return [];
  const out = [];
  for (let i = 0; i < count; i += 1) {
    const unresolved = i % 6 === 5;
    // Every third link is documented but unused by the graph, as notes really are.
    const need = unresolved ? "required" : i % 3 === 2 ? "optional" : "required";
    out.push({
      url: `https://huggingface.co/org/repo/resolve/main/model_${i}.safetensors`,
      source_url: `https://huggingface.co/org/repo/resolve/main/model_${i}.safetensors`,
      provider: "huggingface",
      filename: `some_quite_long_model_name_v${i}_fp8_e4m3fn.safetensors`,
      folder: unresolved ? null : FOLDERS[i % FOLDERS.length],
      size: 1024 ** 3 * (i + 1),
      sha256: "a".repeat(64),
      tier: unresolved ? "unresolved" : ["documented", "properties", "search", "rule"][i % 4],
      reason: REASONS[i % REASONS.length],
      error: null,
      origin: "note",
      origin_node: String(2580 + i),
      existing_path: window.__allPresent ? "/models/present" : i % 7 === 3 ? "/models/x" : null,
      dest_path: "/models/x",
      resolved: !unresolved,
      need,
      slot: { node: "12", input: "unet_name", node_type: "UNETLoader" },
      candidates: [],
    });
  }
  return out;
}

function jobs(count) {
  const out = [];
  for (let i = 0; i < count; i += 1) {
    const total = 1024 ** 3 * (i + 2);
    out.push({
      id: `wmd-${i}`,
      url: "https://huggingface.co/x",
      filename: `some_quite_long_model_name_v${i}_fp8_e4m3fn.safetensors`,
      folder: FOLDERS[i % FOLDERS.length],
      dest: "/models/x",
      provider: "huggingface",
      status: i < 2 ? "downloading" : "queued",
      downloaded: i < 2 ? total * 0.21 : 0,
      total,
      speed: i < 2 ? 48.1 * 1024 * 1024 : 0,
      error: null,
      source: "panel",
      created: 0,
      updated: 0,
    });
  }
  return out;
}

// Jobs the fake server has been asked to start, keyed by workflow.
const queued = new Map();

// Real listener plumbing: the panel's handling of pushed progress is where a
// job belonging to another workflow can leak in, so tests must be able to fire one.
const listeners = new Map();
window.__emit = (type, detail) => {
  for (const fn of listeners.get(type) || []) fn({ detail });
};

export const api = {
  addEventListener(type, fn) {
    listeners.set(type, [...(listeners.get(type) || []), fn]);
  },
  async fetchApi(path, options = {}) {
    const body = options.body ? JSON.parse(options.body) : {};
    if (path.includes("/download")) {
      const key = body.workflow_key || "";
      queued.set(key, [...(queued.get(key) || []), ...(body.items || [])]);
      return new Response(JSON.stringify({ jobs: [] }), {
        headers: { "Content-Type": "application/json" },
      });
    }
    if (path.includes("/manifest")) {
      window.__pinned = body.items || [];
      return new Response(
        JSON.stringify({
          manifest: JSON.stringify({ version: 1, entries: body.items || [] }, null, 2),
          node_type: "WMD_ModelDownloader",
        }),
        { headers: { "Content-Type": "application/json" } },
      );
    }
    return this._fixture(path);
  },
  async _fixture(path) {
    const body2 = path.includes("/folders")
      ? { folders: FOLDERS.map((key) => ({ key, paths: ["/models/" + key] })) }
      : path.includes("/config")
      ? { hf_token_set: true, hf_token_hint: "…d139", hf_token_source: "user",
          civitai_api_key_set: false, civitai_api_key_hint: "", civitai_api_key_source: "",
          prefs: {}, config_path: "/user/config.json" }
      : path.includes("/jobs")
      ? { jobs: jobs(Number(window.__jobCount ?? 0)) }
      : path.includes("/resolve")
      ? { items: items(Number(window.__itemCount ?? 14)) }
      : {};
    return new Response(JSON.stringify(body2), { headers: { "Content-Type": "application/json" } });
  },
};
