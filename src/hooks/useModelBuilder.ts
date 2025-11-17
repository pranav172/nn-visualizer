import * as tf from "@tensorflow/tfjs";

export type LayerSpec = {
  type: "dense";
  units: number;
  activation?: "relu" | "tanh" | "sigmoid" | "softmax" | "linear";
  inputShape?: number[]; // only set on the first layer
};

export type ModelSpec = LayerSpec[];

/**
 * Build a tf.Sequential model from a simple spec.
 */
export function buildModelFromSpec(spec: ModelSpec): tf.LayersModel {
  const model = tf.sequential();

  spec.forEach((layer, idx) => {
    if (layer.type === "dense") {
      // Build the config inline to avoid depending on a specific tfjs exported type name.
      if (layer.inputShape && idx === 0) {
        model.add(
          tf.layers.dense({
            units: layer.units,
            activation: layer.activation,
            inputShape: layer.inputShape,
          })
        );
      } else {
        model.add(
          tf.layers.dense({
            units: layer.units,
            activation: layer.activation,
          })
        );
      }
    } else {
      // safe fallback if we later add other layer kinds
      const t = (layer as { type: unknown }).type + "";
      throw new Error(`Unsupported layer type: ${t}`);
    }
  });

  return model;
}
