// src/hooks/useTrainer.ts
import { useRef, useState } from "react";
import * as tf from "@tensorflow/tfjs";

export type TrainMetrics = { epoch: number; loss: number; acc?: number };

/**
 * useTrainer
 * - train(model, data, options)
 * - stop()
 * - metrics: epoch-by-epoch metrics for UI plotting
 */
export function useTrainer() {
  const [metrics, setMetrics] = useState<TrainMetrics[]>([]);
  const stopRef = useRef(false);

  async function train(
    model: tf.LayersModel,
    data: { X_train: tf.Tensor; y_train: tf.Tensor },
    options?: {
      epochs?: number;
      batchSize?: number;
      optimizer?: tf.Optimizer | string;
      loss?: string;
    }
  ) {
    const {
      epochs = 20,
      batchSize = 4,
      // Default to an Adam optimizer instance with a moderate LR
      optimizer = tf.train.adam(0.03),
      // Default loss appropriate for softmax + one-hot labels
      loss = "categoricalCrossentropy",
    } = options || {};

    stopRef.current = false;

    // Build compiler args: either pass the optimizer instance or let tfjs accept the string
    const compileArgs: {
      optimizer: tf.Optimizer | string;
      loss: string;
      metrics: string[];
    } = {
      optimizer: optimizer as tf.Optimizer | string,
      loss,
      metrics: ["accuracy"],
    };

    // Compile model
    model.compile(compileArgs);

    // reset metrics for a fresh run
    setMetrics([]);

    try {
      await model.fit(data.X_train, data.y_train, {
        epochs,
        batchSize,
        callbacks: {
          onEpochEnd: async (epoch: number, logs?: tf.Logs) => {
            // logs may contain keys like loss, acc or accuracy depending on TFJS version
            const logsRecord = logs as unknown as
              | Record<string, number>
              | undefined;

            const entry: TrainMetrics = {
              epoch: epoch + 1,
              loss: logsRecord?.loss ?? NaN,
              acc: logsRecord?.acc ?? logsRecord?.accuracy,
            };

            // append metric (functional update to avoid stale closure issues)
            setMetrics((prev) => [...prev, entry]);

            // yield to the browser to keep UI responsive
            await tf.nextFrame();

            if (stopRef.current) {
              // gracefully abort training by throwing inside callback
              throw new Error("Training stopped by user");
            }
          },
        },
      });
    } catch (err) {
      // If the throw is from a user stop, swallow silently; else rethrow so caller can handle/log
      if ((err as Error).message !== "Training stopped by user") {
        console.error("Training error", err);
        throw err;
      }
    }
  }

  function stop() {
    stopRef.current = true;
  }

  return { train, stop, metrics } as const;
}
