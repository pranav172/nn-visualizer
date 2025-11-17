📌 Neural Network Visualizer (Interactive + Trainable)

An interactive Neural Network Visualizer built with React, TypeScript, TensorFlow.js, and Vite.
It allows users to design, train, visualize, and evaluate neural networks directly in the browser — no backend required.

🚀 Live Demo: https://nn-visualizer-zeta.vercel.app/

✨ Features
🧠 1. Visual Neural Network Builder

Add/remove Dense layers

Choose activation functions (ReLU, tanh, sigmoid, softmax, linear)

Customize input shapes

Real-time architecture preview

🎨 2. Network Canvas (with Activations)

Renders neurons + fullyconnected edges

Neurons dynamically change color based on activation

Hover tooltips show exact activation values

Updated live during forward pass

📊 3. Live Training Metrics

Tracks loss and accuracy per epoch

Smooth line charts (Chart.js)

Supports adjustable learning rate + epochs

🔍 4. Dataset Playground

Switch between:

XOR

Moons

Spiral
Each dataset has train/test splits and categorical labels.

⚡ 5. Forward Pass & Activations

Run forward pass on any custom input

See per-layer activations

“Run All XOR” mode displays activations + predictions for all four XOR inputs

🧪 6. Evaluation Mode

Displays predicted probabilities

Shows confidence thresholds

Computes final test-set loss and accuracy

Results table highlighting low-confidence predictions

💾 7. Save & Load Models

Save model to IndexedDB

Download model files (model.json + weights)

Load uploaded models for inference

🛠️ Tech Stack
Technology	Purpose
React + TypeScript	UI and component logic
TensorFlow.js	Neural network creation, training, inference
Vite	Fast bundling & development
TailwindCSS	Styling
Chart.js	Training metric graphs
IndexedDB	Local model storage
📁 Project Structure
src/
 ├─ components/
 │   ├─ LayerEditor.tsx
 │   ├─ NetworkCanvas.tsx
 │   ├─ MetricsChart.tsx
 │   ├─ ActivationPanel.tsx
 │
 ├─ hooks/
 │   ├─ useModelBuilder.ts
 │   ├─ useTrainer.ts
 │   ├─ useForwardActivations.ts
 │
 ├─ data/
 │   ├─ datasets.ts
 │   ├─ dataSelector.ts
 │
 ├─ App.tsx
 └─ main.tsx

🧩 How It Works
🧱 Build Model

A simple JSON-like layer spec is converted into a tf.Sequential model.

🎓 Training

Uses TensorFlow.js .fit() with epoch callbacks to update the UI live.

🔥 Activations

Forward-pass is manually implemented through layer.apply() to extract
per-layer activations even during inference.

🖼 Visualization

SVG-based network canvas dynamically renders neuron layers, links, and activation color.

🚀 Running Locally
git clone <repo-url>
cd nn-visualizer
npm install
npm run dev


Open:
👉 http://localhost:5173

🧪 Build for Production
npm run build
npm run preview

🌐 Deployment

This project is deployed on Vercel.
To deploy your own version:

npm i -g vercel
vercel


Or use the Vercel GitHub import UI.

🙌 Author

Pranav Raj
Beginner in Machine Learning | Deep Learning Learner | React & AI Projects
Passionate about building practical tools to understand ML better.

⭐ If you find this useful

Please ⭐ the repo — it motivates further development!
