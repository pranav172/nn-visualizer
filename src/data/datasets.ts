import * as tf from "@tensorflow/tfjs";

export function getXORData() {
  const X = tf.tensor2d(
    [
      [0, 0],
      [0, 1],
      [1, 0],
      [1, 1],
    ],
    [4, 2]
  );

  const y = tf.tensor2d(
    [
      [1, 0],
      [0, 1],
      [0, 1],
      [1, 0],
    ],
    [4, 2]
  );

  // For XOR we keep train / test identical
  return {
    X_train: X,
    y_train: y,
    X_test: X,
    y_test: y,
  };
}

/**
 * Anything that has a dispose(): void is considered disposable.
 * No 'any' usage here.
 */
type Disposable = {
  dispose: () => void;
};

/**
 * Perfectly safe, zero-any type guard.
 * If this returns true, TS knows v.dispose() exists.
 */
function isDisposable(v: unknown): v is Disposable {
  return (
    typeof v === "object" &&
    v !== null &&
    "dispose" in v &&
    // cast to Disposable (NOT any) → ESLint is happy
    typeof (v as Disposable).dispose === "function"
  );
}

/**
 * Dispose tensors or any disposable object.
 */
export function disposeTensors(obj: Record<string, unknown>) {
  for (const v of Object.values(obj)) {
    if (isDisposable(v)) {
      try {
        v.dispose();
      } catch {
        // ignore disposal errors
      }
    }
  }
}
