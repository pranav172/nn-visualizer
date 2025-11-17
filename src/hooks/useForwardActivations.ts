// src/hooks/useForwardActivations.ts
import * as tf from "@tensorflow/tfjs";

/**
 * Run a forward pass and capture activations for each layer.
 * Returns activations as number[][] where each item is the activation vector for that layer.
 */
export async function runForwardActivations(
  model: tf.LayersModel,
  input: number[]
): Promise<number[][] | null> {
  if (!model) return null;

  // start with a tensor for the input; type as the general tf.Tensor so we can reassign
  let x: tf.Tensor = tf.tensor2d([input]);
  const activations: number[][] = [];

  try {
    for (const layer of model.layers) {
      // apply the layer to the current tensor
      const outRaw = layer.apply(x);

      // layer.apply can return a Tensor or Tensor[]; normalize to a single Tensor
      let out: tf.Tensor;
      if (Array.isArray(outRaw)) {
        out = outRaw[0] as tf.Tensor;
      } else {
        out = outRaw as tf.Tensor;
      }

      // read values (may be 1D or 2D; cast to number[][] and take first row)
      const arr = (await out.array()) as number[][];
      activations.push(
        Array.isArray(arr[0]) ? arr[0] : (arr as unknown as number[])
      );

      // dispose previous tensor and set x to out for next iteration
      try {
        x.dispose();
      } catch {
        // ignore
      }
      x = out;
    }

    // dispose final intermediate tensor
    try {
      x.dispose();
    } catch {
      // ignore
    }

    return activations;
  } catch (err) {
    // ensure disposal on error
    try {
      x.dispose();
    } catch {
      // ignore
    }
    console.error("Forward activations error", err);
    return null;
  }
}

/**
 * Hook-like wrapper that returns a run function. Kept as simple helper to call from components.
 */
export function useForwardActivations() {
  return {
    run: runForwardActivations,
  };
}
