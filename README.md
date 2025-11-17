

# 🚀 Neural Network Visualizer (Interactive + Trainable)

An interactive neural network visualizer built with **React**, **TypeScript**, **TensorFlow.js**, and **Vite**.
Design, train, visualize, and evaluate neural networks directly in the browser — **no backend required**.

🔗 **Live Demo:** [https://nn-visualizer-zeta.vercel.app/](https://nn-visualizer-zeta.vercel.app/)

---

## ✨ Features

### 🧠 1. Visual Neural Network Builder

* Add/remove Dense layers
* Choose activation functions *(ReLU, tanh, sigmoid, softmax, linear)*
* Define input shapes
* Real-time architecture preview

### 🎨 2. Network Canvas (with Activations)

* Renders neurons + fully connected edges
* Color-coded neurons based on activation strength
* Hover tooltips show exact activation values
* Updates live during forward pass

### 📊 3. Live Training Metrics

* Tracks **loss** and **accuracy** per epoch
* Smooth line charts (Chart.js)
* Adjustable learning rate and number of epochs

### 🔍 4. Dataset Playground

* **XOR**
* **Moons**
* **Spiral**

### ⚡ 5. Forward Pass Visualizer

* Run forward pass for any input
* View activations layer-by-layer
* “Run All XOR” mode shows full inference table

### 🧪 6. Evaluation Mode

* Compute **test accuracy** and **loss**
* View predicted probabilities
* Confidence threshold classification
* Clean evaluation table

### 💾 7. Save & Load Models

* Download model (`model.json` + weights)
* Save/load using **IndexedDB**
* Continue training after loading

---

## 🛠 Tech Stack

* React + TypeScript
* TensorFlow.js
* Vite
* TailwindCSS
* Chart.js
* IndexedDB

---

## 📂 Project Structure

```
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
```

---

## ▶️ Running Locally

```bash
git clone <your-repo-url>
cd nn-visualizer
npm install
npm run dev
```

**Local dev:** [http://localhost:5173/](http://localhost:5173/)

---

## 🏗️ Production Build

```bash
npm run build
npm run preview
```

---

## 🌐 Deployment

This project is deployed via **Vercel**.
To deploy your own fork:

```bash
npm i -g vercel
vercel
```

Or import the GitHub repo directly:
[https://vercel.com/new](https://vercel.com/new)

---

## 👤 Author

**Pranav Raj**
AI & ML Developer • Deep Learning Learner • Full-Stack ML Enthusiast

---
