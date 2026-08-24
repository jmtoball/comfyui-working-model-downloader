// Stub of ComfyUI's app module, enough to mount the panel.
export const app = {
  graph: {
    _nodes: [],
    serialize: () => ({ nodes: [] }),
    setDirtyCanvas() {},
    add() {},
  },
  graphToPrompt: async () => ({ output: {} }),
  registerExtension(extension) {
    window.__extension = extension;
    extension.setup?.();
  },
  extensionManager: {
    registerSidebarTab(tab) {
      window.__tab = tab;
    },
    setting: { get: () => undefined, set: async () => {} },
  },
};
