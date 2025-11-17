// src/components/NetworkCanvas.tsx
import type { FC } from "react";

type LayerViz = { units: number };
type Props = {
  spec?: LayerViz[]; // e.g. [{ units: 4 }, { units: 2 }]
  // optional activations: activations[layerIndex][unitIndex] (numbers, e.g. tanh or probs)
  activations?: number[][];
};

const SVG_WIDTH = 900;
const SVG_HEIGHT = 420;
const LAYER_X_MARGIN = 60;
const NEURON_RADIUS = 10;

const getLayerX = (index: number, layerCount: number) => {
  const usableWidth = SVG_WIDTH - LAYER_X_MARGIN * 2;
  if (layerCount === 1) return LAYER_X_MARGIN + usableWidth / 2;
  return LAYER_X_MARGIN + (usableWidth * index) / (layerCount - 1);
};

const getNeuronY = (unitIndex: number, units: number) => {
  const topMargin = 40;
  const bottomMargin = 40;
  if (units === 1)
    return topMargin + (SVG_HEIGHT - topMargin - bottomMargin) / 2;
  return (
    topMargin +
    (unitIndex * (SVG_HEIGHT - topMargin - bottomMargin)) / (units - 1)
  );
};

const clamp = (v: number, lo = 0, hi = 1) => Math.max(lo, Math.min(hi, v));

const activationToFill = (a?: number) => {
  if (a === undefined || Number.isNaN(a)) return "#93c5fd"; // light-blue default for white theme
  // map activation magnitude to alpha 0.15..1
  const alpha = 0.15 + 0.85 * clamp(Math.abs(a), 0, 1);
  return `rgba(59,130,246,${alpha})`; // blue with variable opacity
};

const formatActivation = (a?: number) =>
  a === undefined || Number.isNaN(a) ? "n/a" : a.toFixed(3);

const NetworkCanvas: FC<Props> = ({ spec = [], activations }) => {
  const layers = spec;
  const layerCount = layers.length;

  // Precompute neuron positions for edges
  const neuronPositions = layers.map((layer, li) =>
    Array.from({ length: layer.units }).map((_, ui) => {
      const x = getLayerX(li, Math.max(1, layerCount || 1));
      const y = getNeuronY(ui, layer.units);
      return { x, y };
    })
  );

  return (
    <div>
      <svg
        width="100%"
        viewBox={`0 0 ${SVG_WIDTH} ${SVG_HEIGHT}`}
        style={{
          background: "white",
          borderRadius: 8,
        }}
      >
        <rect
          x={0}
          y={0}
          width={SVG_WIDTH}
          height={SVG_HEIGHT}
          fill="transparent"
          rx={8}
        />

        {/* Draw edges (full connectivity between adjacent layers) */}
        {neuronPositions.map((layerPos, li) =>
          li < neuronPositions.length - 1
            ? layerPos.flatMap((fromPos, fi) =>
                neuronPositions[li + 1].map((toPos, ti) => {
                  const key = `e-${li}-${fi}-${li + 1}-${ti}`;
                  return (
                    <line
                      key={key}
                      x1={fromPos.x}
                      y1={fromPos.y}
                      x2={toPos.x}
                      y2={toPos.y}
                      stroke="#e5e7eb"
                      strokeWidth={1}
                      opacity={0.9}
                    />
                  );
                })
              )
            : null
        )}

        {/* Draw neurons */}
        {neuronPositions.map((layerPos, li) =>
          layerPos.map((pos, ui) => {
            const act = activations?.[li]?.[ui];
            const fill = activationToFill(act);
            const key = `n-${li}-${ui}`;
            const titleText = `Layer ${li} Unit ${ui}\nact: ${formatActivation(
              act
            )}`;
            return (
              <g key={key}>
                <circle
                  cx={pos.x}
                  cy={pos.y}
                  r={NEURON_RADIUS}
                  fill={fill}
                  stroke="#3b82f6"
                  strokeWidth={1.2}
                />
                {/* native tooltip for numeric activation */}
                <title>{titleText}</title>

                {/* neuron label */}
                <text
                  x={pos.x + NEURON_RADIUS + 6}
                  y={pos.y + 4}
                  fontSize={10}
                  fontFamily="monospace"
                  fill="#374151"
                >
                  {`${li}:${ui}`}
                </text>
              </g>
            );
          })
        )}

        {/* Small header text */}
        <text x={16} y={20} fill="#1e40af" fontFamily="monospace" fontSize={12}>
          Canvas ready — neurons and connections (simplified)
        </text>
      </svg>
    </div>
  );
};

export default NetworkCanvas;
