// src/components/MetricsChart.tsx
import { Line } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
} from "chart.js";
import type { TrainMetrics } from "../hooks/useTrainer";
import type { ChartDataset } from "chart.js";

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

type Props = {
  metrics: TrainMetrics[];
  showAccuracy?: boolean;
};

export default function MetricsChart({ metrics, showAccuracy = false }: Props) {
  const labels = metrics.map((m) => `E${m.epoch}`);

  // Chart.js dataset type for a line chart with numeric (or null) points
  const lossDataset: ChartDataset<"line", (number | null)[]> = {
    label: "Loss",
    data: metrics.map((m) => (Number.isFinite(m.loss) ? m.loss : null)),
    tension: 0.2,
    fill: false,
    borderColor: "#2563eb", // blue
    backgroundColor: "#bfdbfe",
    pointRadius: 3,
  };

  const datasets: ChartDataset<"line", (number | null)[]>[] = [lossDataset];

  if (showAccuracy) {
    const accDataset: ChartDataset<"line", (number | null)[]> = {
      label: "Accuracy",
      data: metrics.map((m) =>
        typeof m.acc === "number" && Number.isFinite(m.acc) ? m.acc : null
      ),
      tension: 0.2,
      fill: false,
      borderColor: "#0ea5a4", // teal for accuracy
      backgroundColor: "#bbf7d0",
      pointRadius: 3,
    };
    datasets.push(accDataset);
  }

  const data = { labels, datasets };

  const options = {
    responsive: true,
    plugins: {
      legend: {
        position: "top" as const,
        labels: { color: "#0f172a" }, // dark text for legend
      },
      title: { display: false, text: "Training metrics" },
      tooltip: { enabled: true },
    },
    scales: {
      x: {
        ticks: { color: "#0f172a" },
        grid: { color: "#eef2ff" }, // subtle grid
      },
      y: {
        beginAtZero: true,
        ticks: { color: "#0f172a" },
        grid: { color: "#eef2ff" },
      },
    },
  };

  return <Line data={data} options={options} />;
}
