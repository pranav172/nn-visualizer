// src/App.tsx
import { useMemo, useRef, useState } from "react";
import type { ChangeEvent } from "react";

import LayerEditor from "./components/LayerEditor";
import NetworkCanvas from "./components/NetworkCanvas";
import MetricsChart from "./components/MetricsChart";
import ActivationPanel from "./components/ActivationPanel";

import { buildModelFromSpec, type ModelSpec } from "./hooks/useModelBuilder";
import { useTrainer } from "./hooks/useTrainer";
import { disposeTensors } from "./data/datasets";
import { chooseDataset } from "./data/dataSelector";
import { useForwardActivations } from "./hooks/useForwardActivations";
import * as tf from "@tensorflow/tfjs";

type EvalRow = {
  index: number;
  input: number[];
  probs: number[]; // length C
  predClass: number;
  trueClass: number;
  confident: boolean;
};

export default function App() {
  // Default XOR spec
  const defaultXorSpec: ModelSpec = [
    { type: "dense", units: 4, activation: "tanh", inputShape: [2] },
    { type: "dense", units: 2, activation: "softmax" },
  ];

  // Model spec controlled by App
  const [spec, setSpec] = useState<ModelSpec | undefined>(defaultXorSpec);

  // training state and options
  const [isTraining, setIsTraining] = useState(false);
  const [epochs, setEpochs] = useState<number>(50);
  const [lr, setLr] = useState<number>(0.03);

  // dataset selector
  const [datasetName, setDatasetName] = useState<"xor" | "moons" | "spiral">(
    "xor"
  );

  const { train, stop, metrics } = useTrainer();

  // keep a reference to the built model so we can dispose it when replaced
  const modelRef = useRef<tf.LayersModel | null>(null);

  // forward activations helper
  const { run: runForward } = useForwardActivations();

  // forward-pass activations state (layers -> units)
  const [activations, setActivations] = useState<number[][] | null>(null);

  // evaluation results to show in UI
  const [evalRows, setEvalRows] = useState<EvalRow[] | null>(null);
  const [lastEvalLoss, setLastEvalLoss] = useState<number | null>(null);
  const [lastEvalAcc, setLastEvalAcc] = useState<number | null>(null);

  // Run-all results (for Run All XOR)
  const [allForwardResults, setAllForwardResults] = useState<
    { input: number[]; activations?: number[][]; probs?: number[] }[] | null
  >(null);

  // confidence threshold
  const [threshold, setThreshold] = useState<number>(0.5);

  function buildModel() {
    if (!spec) throw new Error("No model spec present");
    // dispose previous model if present
    if (modelRef.current) {
      try {
        modelRef.current.dispose();
      } catch {
        // ignore dispose errors
      }
      modelRef.current = null;
    }
    const model = buildModelFromSpec(spec);
    modelRef.current = model;
    return model;
  }

  async function handleTrain() {
    setIsTraining(true);
    try {
      const model = buildModel();
      const opt = tf.train.adam(lr);
      const data = chooseDataset(datasetName);
      await train(
        model,
        { X_train: data.X_train, y_train: data.y_train },
        {
          epochs,
          batchSize: 16,
          optimizer: opt,
          loss: "categoricalCrossentropy",
        }
      );
      // dispose training data tensors after use
      try {
        disposeTensors(data);
      } catch {
        // ignore disposal errors
      }
    } catch (err) {
      console.error("Training failed or was stopped", err);
    } finally {
      setIsTraining(false);
    }
  }

  function handleStop() {
    stop();
    setIsTraining(false);
  }

  // Evaluate model and build evalRows for the UI table
  async function handleEvaluate() {
    if (!modelRef.current) {
      alert("Build model first by training (or press Train).");
      return;
    }

    const model = modelRef.current;
    const data = chooseDataset(datasetName);

    try {
      const evalOut = (await model.evaluate(data.X_test, data.y_test, {
        batchSize: 32,
      })) as tf.Scalar | tf.Scalar[];

      if (Array.isArray(evalOut)) {
        const lossNum = (await evalOut[0].array()) as number;
        const accNum = (await evalOut[1].array()) as number;
        setLastEvalLoss(lossNum);
        setLastEvalAcc(accNum);
        evalOut.forEach((s) => s.dispose());
      } else {
        const lossNum = (await evalOut.array()) as number;
        setLastEvalLoss(lossNum);
        setLastEvalAcc(null);
        evalOut.dispose();
      }

      // predictions
      const preds = model.predict(data.X_test) as tf.Tensor; // shape [N, C]
      const probs = (await preds.array()) as number[][];
      const predClasses = (await preds.argMax(-1).array()) as number[];
      const trueClasses = (await data.y_test.argMax(-1).array()) as number[]; // [0..C-1]
      const inputs = (await data.X_test.array()) as number[][];

      const rows: EvalRow[] = inputs.map((inp, i) => {
        const pred = predClasses[i];
        const probOfPred = probs[i][pred] ?? 0;
        const confident = probOfPred >= threshold;
        return {
          index: i,
          input: inp,
          probs: probs[i],
          predClass: pred,
          trueClass: trueClasses[i],
          confident,
        };
      });

      setEvalRows(rows);

      // cleanup
      tf.dispose(preds);
      try {
        disposeTensors(data);
      } catch {
        // ignore disposal errors
      }
    } catch (err) {
      console.error("Evaluation error", err);
      try {
        disposeTensors(data);
      } catch {
        // ignore disposal errors
      }
    }
  }

  // Run forward pass for a single input and show activations
  async function handleForwardPass(input: number[]) {
    if (!modelRef.current) {
      alert("Train or load a model first.");
      return;
    }
    try {
      const acts = await runForward(modelRef.current, input);
      // log for debugging
      // console.log("Forward activations (per layer):", acts);
      setActivations(acts);
    } catch (err) {
      console.error("Forward pass error", err);
    }
  }

  // Run all XOR samples, collect activations + probs, and set table state
  async function handleRunAllXOR() {
    if (!modelRef.current) {
      alert("Train or load a model first.");
      return;
    }
    const inputs = [
      [0, 0],
      [0, 1],
      [1, 0],
      [1, 1],
    ];

    const results: {
      input: number[];
      activations?: number[][];
      probs?: number[];
    }[] = [];

    for (const inp of inputs) {
      // get layer activations via runForward
      const acts = await runForward(modelRef.current, inp);
      // get final predictions (probabilities)
      const predsTensor = modelRef.current.predict(
        tf.tensor2d([inp])
      ) as tf.Tensor;
      const pArr = (await predsTensor.array()) as number[][];
      const probs = pArr[0];
      tf.dispose(predsTensor);
      results.push({ input: inp, activations: acts ?? undefined, probs });
    }

    setAllForwardResults(results);
    // set canvas activations to last run for visibility
    setActivations(results[results.length - 1].activations ?? null);
  }

  // Save / Load helpers
  async function handleSaveDownload() {
    if (!modelRef.current) {
      alert("No model to save. Train first.");
      return;
    }
    try {
      await modelRef.current.save("downloads://nn-visualizer-model");
    } catch (err) {
      console.error("Save download failed", err);
      alert("Save failed: " + String(err));
    }
  }

  async function handleSaveIndexedDB() {
    if (!modelRef.current) {
      alert("No model to save. Train first.");
      return;
    }
    try {
      await modelRef.current.save("indexeddb://nn-visualizer-model");
      alert("Model saved to IndexedDB (nn-visualizer-model).");
    } catch (err) {
      console.error("IndexedDB save failed", err);
      alert("Save failed: " + String(err));
    }
  }

  async function handleLoadIndexedDB() {
    try {
      const loaded = (await tf.loadLayersModel(
        "indexeddb://nn-visualizer-model"
      )) as tf.LayersModel;
      if (modelRef.current) {
        try {
          modelRef.current.dispose();
        } catch {
          // ignore
        }
      }
      modelRef.current = loaded;
      alert("Model loaded from IndexedDB.");
    } catch (err) {
      console.error("Load failed", err);
      alert("Load failed: " + String(err));
    }
  }

  // memoized simplified view of layers for the canvas
  const vizSpec = useMemo(() => {
    if (!spec) return [];
    return spec.map((l) => ({ units: l.units }));
  }, [spec]);

  return (
    <div className="min-h-screen bg-white text-black">
      <header className="p-4 border-b border-gray-300 flex items-center justify-between bg-white">
        <h1 className="text-2xl font-semibold text-blue-700">
          NN Visualizer — Final
        </h1>

        <div className="flex items-center gap-3">
          <select
            value={datasetName}
            onChange={(e: ChangeEvent<HTMLSelectElement>) =>
              setDatasetName(e.target.value as "xor" | "moons" | "spiral")
            }
            className="px-2 py-1 border rounded"
          >
            <option value="xor">XOR</option>
            <option value="moons">Moons</option>
            <option value="spiral">Spiral</option>
          </select>

          <div className="flex items-center gap-2">
            <label className="text-sm text-gray-700">Epochs</label>
            <input
              type="number"
              value={epochs}
              min={1}
              onChange={(e) => setEpochs(Number(e.target.value))}
              className="w-20 px-2 py-1 border rounded"
            />
          </div>

          <div className="flex items-center gap-2">
            <label className="text-sm text-gray-700">LR</label>
            <input
              type="number"
              step="0.001"
              value={lr}
              min={0.0001}
              onChange={(e) => setLr(Number(e.target.value))}
              className="w-24 px-2 py-1 border rounded"
            />
          </div>

          <button
            onClick={handleTrain}
            disabled={isTraining}
            className="px-3 py-1 rounded bg-blue-600 text-white disabled:opacity-50"
          >
            Train
          </button>
          <button
            onClick={handleStop}
            disabled={!isTraining}
            className="px-3 py-1 rounded bg-red-500 text-white disabled:opacity-50"
          >
            Stop
          </button>
          <button
            onClick={handleEvaluate}
            className="px-3 py-1 rounded bg-sky-600 text-white"
          >
            Evaluate
          </button>
        </div>
      </header>

      <main className="p-6 grid grid-cols-4 gap-6">
        <aside className="col-span-1 bg-white border border-gray-200 p-4 rounded-md shadow-sm overflow-y-auto">
          <LayerEditor value={spec} onChange={(s) => setSpec(s)} />
          <div className="mt-4 text-sm text-gray-600">
            Tip: For XOR use inputShape [2], hidden layer 4 units (tanh), output
            2 units (softmax).
          </div>

          <div className="mt-4 space-y-3">
            <ActivationPanel
              onRun={(input) => void handleForwardPass(input)}
              onRunAll={handleRunAllXOR}
            />

            <div className="p-3 border rounded bg-white">
              <h4 className="text-sm font-medium text-blue-700 mb-2">Model</h4>
              <div className="flex flex-col gap-2">
                <button
                  onClick={handleSaveDownload}
                  className="px-3 py-1 rounded bg-green-600 text-white"
                >
                  Download Model
                </button>
                <button
                  onClick={handleSaveIndexedDB}
                  className="px-3 py-1 rounded bg-emerald-500 text-white"
                >
                  Save to IndexedDB
                </button>
                <button
                  onClick={handleLoadIndexedDB}
                  className="px-3 py-1 rounded bg-purple-600 text-white"
                >
                  Load from IndexedDB
                </button>
              </div>
            </div>

            <div className="p-3 border rounded bg-white">
              <h4 className="text-sm font-medium text-blue-700 mb-2">
                Confidence Threshold
              </h4>
              <div className="flex items-center gap-3">
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.01}
                  value={threshold}
                  onChange={(e) => setThreshold(Number(e.target.value))}
                  className="w-full"
                />
                <div className="w-12 text-right">{threshold.toFixed(2)}</div>
              </div>
            </div>
          </div>
        </aside>

        <section className="col-span-3 bg-white border border-gray-200 p-4 rounded-md shadow-sm flex flex-col gap-4">
          <div>
            <h3 className="text-lg font-medium text-blue-700 mb-3">
              Network Canvas
            </h3>
            <NetworkCanvas
              spec={vizSpec}
              activations={activations ?? undefined}
            />
            {/* numeric activations display */}
            {activations && (
              <div className="mt-2 text-sm">
                <h4 className="font-medium">Activations (layer → units)</h4>
                {activations.map((layerActs, li) => (
                  <div key={li}>
                    <strong>Layer {li}:</strong>{" "}
                    {layerActs.map((v, ui) => (
                      <span key={ui} className="inline-block mr-2">
                        {v.toFixed(3)}
                      </span>
                    ))}
                  </div>
                ))}
              </div>
            )}
          </div>

          <div className="bg-white border border-gray-200 p-3 rounded shadow-sm">
            <h3 className="text-md font-medium text-blue-700 mb-2">
              Training Metrics
            </h3>
            <div style={{ height: 260 }}>
              <MetricsChart metrics={metrics} showAccuracy />
            </div>
          </div>

          <div className="bg-white border border-gray-200 p-3 rounded shadow-sm">
            <h3 className="text-md font-medium text-blue-700 mb-2">
              Evaluation
            </h3>

            <div className="mb-3 text-sm">
              <strong>Last eval:</strong>{" "}
              {lastEvalLoss !== null ? (
                <>
                  Loss: {lastEvalLoss.toFixed(4)}{" "}
                  {lastEvalAcc !== null
                    ? ` | Acc: ${lastEvalAcc.toFixed(4)}`
                    : null}
                </>
              ) : (
                "No evaluation yet"
              )}
            </div>

            {evalRows ? (
              <table className="w-full text-sm table-fixed border-collapse">
                <thead>
                  <tr className="text-left">
                    <th className="pb-2">#</th>
                    <th className="pb-2">Input</th>
                    <th className="pb-2">Pred probs</th>
                    <th className="pb-2">Pred</th>
                    <th className="pb-2">True</th>
                    <th className="pb-2">Confident</th>
                  </tr>
                </thead>
                <tbody>
                  {evalRows.map((r) => (
                    <tr key={r.index} className="border-t">
                      <td className="py-2">{r.index}</td>
                      <td className="py-2">{`[${r.input
                        .map((v) => v.toFixed(0))
                        .join(", ")}]`}</td>
                      <td className="py-2">{`[${r.probs
                        .map((p) => p.toFixed(3))
                        .join(", ")}]`}</td>
                      <td className="py-2">{r.predClass}</td>
                      <td className="py-2">{r.trueClass}</td>
                      <td className="py-2">
                        <span
                          className={
                            r.confident ? "text-green-600" : "text-red-600"
                          }
                        >
                          {r.confident ? "OK" : "LOW"}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <div className="text-sm text-gray-600">
                No predictions yet — click Evaluate after training.
              </div>
            )}

            {/* Run-all results table */}
            {allForwardResults && (
              <div className="mt-4 p-3 border rounded bg-white">
                <h4 className="font-medium text-blue-700 mb-2">
                  Forward Results — All XOR
                </h4>
                <table className="w-full text-sm">
                  <thead>
                    <tr>
                      <th>#</th>
                      <th>Input</th>
                      <th>Hidden (layer0)</th>
                      <th>Output probs</th>
                    </tr>
                  </thead>
                  <tbody>
                    {allForwardResults.map((r, i) => (
                      <tr key={i} className="border-t">
                        <td className="py-2">{i}</td>
                        <td className="py-2">{`[${r.input.join(", ")}]`}</td>
                        <td className="py-2">
                          {r.activations?.[0]
                            ?.map((v) => v.toFixed(3))
                            .join(", ") ?? "n/a"}
                        </td>
                        <td className="py-2">
                          {r.probs?.map((p) => p.toFixed(3)).join(", ") ??
                            "n/a"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </section>
      </main>
    </div>
  );
}
