import * as tf from "@tensorflow/tfjs";
import { getXORData } from "./datasets";

export function getMoons() {
  const X = tf.tensor2d([
    [0.1, 0.3],
    [0.2, 0.4],
    [0.9, 0.7],
    [0.8, 0.65],
  ]);
  const y = tf.tensor2d([
    [1, 0],
    [1, 0],
    [0, 1],
    [0, 1],
  ]);
  return { X_train: X, y_train: y, X_test: X, y_test: y };
}

export function getSpiral() {
  const X = tf.randomUniform([200, 2]);
  const y = tf.oneHot(tf.randomUniform([200], 0, 2, "int32"), 2);
  return { X_train: X, y_train: y, X_test: X, y_test: y };
}

export function chooseDataset(name: string) {
  switch (name) {
    case "moons":
      return getMoons();
    case "spiral":
      return getSpiral();
    default:
      return getXORData();
  }
}
