import { useEffect, useState } from "react";
import type { ModelSpec, LayerSpec } from "../hooks/useModelBuilder";

type Props = {
  value?: ModelSpec; // controlled prop
  onChange?: (spec: ModelSpec) => void;
};

const defaultSpec: ModelSpec = [
  { type: "dense", units: 4, activation: "tanh", inputShape: [2] },
  { type: "dense", units: 2, activation: "softmax" },
];

export default function LayerEditor({ value, onChange }: Props) {
  const isControlled = value !== undefined;
  const [internalSpec, setInternalSpec] = useState<ModelSpec>(
    value ?? defaultSpec
  );

  const spec = isControlled ? (value as ModelSpec) : internalSpec;

  useEffect(() => {
    onChange?.(spec);
  }, [spec, onChange]);

  function updateSpec(newSpec: ModelSpec) {
    if (isControlled) {
      onChange?.(newSpec);
    } else {
      setInternalSpec(newSpec);
    }
  }

  function updateLayer(idx: number, patch: Partial<LayerSpec>) {
    updateSpec(spec.map((l, i) => (i === idx ? { ...l, ...patch } : l)));
  }

  function addDense() {
    updateSpec([...spec, { type: "dense", units: 4, activation: "relu" }]);
  }

  function removeLayer(idx: number) {
    updateSpec(spec.filter((_, i) => i !== idx));
  }

  function moveLayerUp(idx: number) {
    if (idx === 0) return;
    const copy = [...spec];
    [copy[idx - 1], copy[idx]] = [copy[idx], copy[idx - 1]];
    updateSpec(copy);
  }

  function moveLayerDown(idx: number) {
    if (idx === spec.length - 1) return;
    const copy = [...spec];
    [copy[idx + 1], copy[idx]] = [copy[idx], copy[idx + 1]];
    updateSpec(copy);
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-lg font-medium text-blue-700">Layer Editor</h3>
        <button
          className="px-3 py-1 rounded bg-blue-500 text-white"
          onClick={addDense}
        >
          + Add Dense
        </button>
      </div>

      <div className="space-y-3">
        {spec.map((layer, idx) => (
          <div
            key={idx}
            className="p-3 bg-gray-100 border border-gray-300 rounded shadow-sm"
          >
            <div className="flex items-center justify-between mb-2">
              <strong className="text-black">
                {layer.type.toUpperCase()} layer
              </strong>
              <div className="space-x-2">
                <button
                  onClick={() => moveLayerUp(idx)}
                  disabled={idx === 0}
                  className="px-2 py-1 rounded bg-gray-200 text-black"
                >
                  ↑
                </button>
                <button
                  onClick={() => moveLayerDown(idx)}
                  disabled={idx === spec.length - 1}
                  className="px-2 py-1 rounded bg-gray-200 text-black"
                >
                  ↓
                </button>
                <button
                  onClick={() => removeLayer(idx)}
                  className="px-2 py-1 rounded bg-red-500 text-white"
                >
                  ✕
                </button>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-2">
              <label className="text-sm text-black">
                Units
                <input
                  type="number"
                  value={layer.units}
                  onChange={(e) =>
                    updateLayer(idx, { units: Number(e.target.value) })
                  }
                  className="ml-2 w-full px-2 py-1 border border-gray-300 rounded bg-white text-black"
                />
              </label>

              <label className="text-sm text-black">
                Activation
                <select
                  value={layer.activation}
                  onChange={(e) =>
                    updateLayer(idx, {
                      activation: e.target.value as LayerSpec["activation"],
                    })
                  }
                  className="ml-2 w-full px-2 py-1 border border-gray-300 rounded bg-white text-black"
                >
                  <option value="relu">relu</option>
                  <option value="tanh">tanh</option>
                  <option value="sigmoid">sigmoid</option>
                  <option value="softmax">softmax</option>
                  <option value="linear">linear</option>
                </select>
              </label>

              {idx === 0 && (
                <label className="text-sm text-black col-span-2">
                  Input shape (comma-separated)
                  <input
                    type="text"
                    defaultValue={(layer.inputShape || []).join(",")}
                    onBlur={(e) => {
                      const parts = e.target.value
                        .split(",")
                        .map((s) => Number(s.trim()))
                        .filter((n) => !Number.isNaN(n));
                      updateLayer(idx, {
                        inputShape: parts.length ? parts : undefined,
                      });
                    }}
                    className="ml-2 w-full px-2 py-1 border border-gray-300 rounded bg-white text-black"
                  />
                </label>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
