// src/components/ActivationPanel.tsx
import { useState } from "react";

type Props = {
  onRun: (input: number[]) => void;
  onRunAll?: () => void;
};

export default function ActivationPanel({ onRun, onRunAll }: Props) {
  const [a, setA] = useState("0");
  const [b, setB] = useState("0");

  return (
    <div className="p-3 border rounded bg-white shadow-sm">
      <h3 className="font-medium text-blue-700 mb-2">Forward Pass</h3>

      <div className="flex gap-2 mb-3">
        <input
          type="number"
          value={a}
          onChange={(e) => setA(e.target.value)}
          className="border px-2 py-1 rounded w-20"
        />
        <input
          type="number"
          value={b}
          onChange={(e) => setB(e.target.value)}
          className="border px-2 py-1 rounded w-20"
        />
      </div>

      <div className="flex gap-2">
        <button
          onClick={() => onRun([Number(a), Number(b)])}
          className="px-3 py-1 rounded bg-blue-600 text-white"
        >
          Run Forward Pass
        </button>

        <button
          onClick={() => onRunAll?.()}
          className="px-3 py-1 rounded bg-indigo-600 text-white"
        >
          Run All XOR
        </button>
      </div>
    </div>
  );
}
