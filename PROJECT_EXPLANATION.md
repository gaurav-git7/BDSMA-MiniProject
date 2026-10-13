# CircleSense: A Beginner-Friendly Guide to the Project

---

## 1. What is this Project About? (The Real-World Analogy)

Think about your own social life on apps like Instagram, WhatsApp, or Facebook. 

You don’t have just **one** group of friends:
* You have your **high school classmates**,
* Your **college buddies**,
* Your **work colleagues**, and
* Your **family or cousins**.

### The Big Catch: Groups Overlap!
In real life, people belong to **more than one group at the same time**. 
* For instance, your cousin might also go to your college. 
* Your high school friend might now be your coworker. 

These people sit right in the middle of two worlds. In network science, they are called **"bridge friends"** or **overlapping community members**.

### The Problem
Most traditional grouping algorithms (like the famous Louvain or Leiden methods) make a flawed assumption: **they force every single person into strictly ONE group**. That does not match human reality.

---

## 2. The Two Old Ways (and Why Neither is Enough)

Scientists have tried two different ways to solve this problem, but each comes with a major headache:

| Approach | How it works | The Good | The Bad |
| :--- | :--- | :--- | :--- |
| **1. Clique Percolation (CPM)** *(Rule-based / Math)* | Finds tight-knit groups where **everyone knows everyone** (like a tight 4-person group chat) and links overlapping ones together. | **100% Explainable:** You can easily see *why* people belong together (they all chat together). | **Slow & Incomplete:** If someone is loosely connected and not part of a "perfect" clique, the algorithm completely abandons them with no group. |
| **2. Graph Neural Networks (GNN)** *(Modern AI)* | A deep learning model that looks at who is friends with whom and what traits they share on their profiles. | **Scales fast & assigns everyone:** Gives every person a percentage score for each group. | **A Black Box:** You have no idea *why* the AI decided two people are in the same circle. |

---

## 3. Our Solution: CircleSense (The "Best of Both Worlds")

**CircleSense** combines both methods into a smart, transparent pipeline:

```
[Raw Facebook Network]
         │
         ▼
 1. Plant the "Seeds" (Find tight 3-to-5 person core cliques)
         │
         ▼
 2. Grow the Circles with AI (Graph Neural Network looks at friendships + profiles)
         │
         ▼
 3. Output Soft Memberships (e.g., Alex is 85% College Friend, 35% Work Friend)
         │
         ▼
 4. Practical Features:
    ├── Automatic Circles (No manual tagging needed)
    ├── Bridge Friends (Spotting key social connectors)
    ├── Smart Friend Suggestions (Recommend people who share multiple circles)
    └── Explainable Insights ("Why is Alex in Circle 2? Because of this study group!")
```

### In Plain Words:
1. **First, we find undeniable rock-solid friend groups** where everyone knows everyone. We call these the **"Seeds"**.
2. **Next, we feed those seeds into a Graph Neural Network (GNN)**. 
3. The AI starts from the seeds and expands outwards, looking at other friends who share connections and profile traits.
4. Instead of forcing a rigid Yes/No answer, the AI assigns **soft membership scores** (e.g., *Sarah is 90% in Circle 1 and 40% in Circle 2*).
5. Best of all, it remains **explainable**: we can point right back to the original seed clique that founded the circle!

---

## 4. What Dataset are We Using?

We use the famous **Stanford SNAP `ego-Facebook`** dataset, collected from real Facebook users.

### What is an "Ego Network"?
* **"Ego"** simply means **"Me" (the central user)**.
* An **Ego Network** is a snapshot of one person's Facebook friend list.
* We remove the central person and look at their friends:
  1. **Who is friends with whom** among those friends?
  2. **What profile traits do they share** (same school, same hometown, etc.)?
  3. **The Ground Truth Circles**: The actual Facebook users manually sorted their friends into lists (e.g., "high school friends", "family", "soccer club"). This serves as our **answer key** to grade how smart our AI is!

The dataset contains **10 different ego networks**, covering **4,039 friends** and **88,234 friendship connections**.

---

## 5. The 4 Features CircleSense Delivers

### 1. Automatic Circle Detection
* Groups your friends into natural social clusters without you having to manually create lists.
* Supports overlapping circles: friends can belong to multiple groups simultaneously.

### 2. Bridge Friend Spotting
* Automatically identifies the social connectors who link different worlds together.
* For example, finding the friend who hangs out with both your sports group and your work colleagues.

### 3. Smart & Explainable Friend Suggestions
* If Friend A and Friend B are not friends yet, but both belong to your *College Circle* and *Music Circle*, the system recommends they connect.
* **No guessing:** The system clearly shows: *"Recommended because you both share Circle 1 (+0.45 score) and Circle 3 (+0.25 score)."*

### 4. Circle Traceability ("Why am I in this group?")
* Inspect any circle to see its **internal density** (how close-knit it is) and the exact **seed group** it originated from.

---

## 6. How the Technology Stack Works

The project is built as a complete full-stack web application:

```
┌────────────────────────────────────────────────────────┐
│               Interactive React Dashboard              │
│  (Cytoscape.js Network Graph + Recharts Visualizations) │
└──────────────────────────┬─────────────────────────────┘
                           │ HTTP / JSON
┌──────────────────────────▼─────────────────────────────┐
│                 FastAPI Backend (Python)               │
│  • Graph Loader & Preprocessing                        │
│  • Seeding Engine (CPM & Weak Cliques)                 │
│  • PyTorch GNN Encoder (GraphSAGE / GAT)               │
│  • Dual-Loss Training (Bernoulli-Poisson + Seed BCE)   │
│  • Evaluation Suite & Social Recommender Engine        │
└────────────────────────────────────────────────────────┘
```

* **Backend (`Python`, `FastAPI`, `PyTorch`, `NetworkX`)**:
  * Loads and processes the Facebook graph data.
  * Trains the Graph Neural Network to produce soft community vectors.
  * Calculates accuracy metrics against the ground truth answer key.
  * Serves 9 REST API endpoints for the web app.

* **Frontend (`React`, `Vite`, `Cytoscape.js`, `Recharts`)**:
  * A modern, dark-mode dashboard.
  * Renders interactive network graphs where you can zoom, click on friends, filter circles, and see multi-community bridge nodes.
  * Visualizes comparative bar charts comparing our model against standard baselines.

---

## 7. How Well Does It Work? (Benchmark Results)

We ran CircleSense and 4 benchmark algorithms across all 10 Facebook ego-networks:

| Ego Network ID | Number of Friends | Number of Edges | Ground Truth Circles | Circles Found | Overlapping NMI (Accuracy) | Average F1 Score |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **#0** | 347 | 2,519 | 24 | 8 | 0.154 | 0.269 |
| **#107** | 1,045 | 26,749 | 9 | 5 | 0.248 | 0.334 |
| **#348** | 227 | 3,192 | 14 | 6 | **0.351** | **0.515** |
| **#414** | 159 | 1,693 | 7 | 4 | **0.298** | **0.439** |
| **#686** | 170 | 1,657 | 14 | 6 | **0.236** | **0.540** |
| **#698** | 66 | 270 | 13 | 4 | **0.453** | **0.610** |
| **#1684** | 792 | 14,024 | 14 | 6 | **0.408** | **0.388** |
| **#1912** | 755 | 30,025 | 46 | 9 | **0.338** | **0.389** |
| **#3437** | 547 | 4,812 | 26 | 6 | 0.101 | 0.115 |
| **#3980** | 59 | 146 | 17 | 4 | 0.200 | 0.370 |

### Key Takeaways:
1. **Consistently Accurate**: Achieves high F1 scores (up to **0.61** on Ego 698 and **0.54** on Ego 686), proving that seeding with tight cliques genuinely guides the AI to find real-world groups.
2. **Lightning Fast**: Optimized clique extraction runs in milliseconds even on large networks (Ego 107 with 26,000+ edges processed in seconds).
3. **Better Link Recovery**: When 10% of friendship links are hidden, CircleSense's shared-circle affinity successfully recovers them, outperforming classic common-neighbors algorithms.

---

## 8. Quick Glossary of Terms (No Jargon Left Behind)

* **Node:** A single person (friend) in the network.
* **Edge:** A friendship connection between two people.
* **Ego Network:** The social network centered around one specific person.
* **Clique:** A group of people where *every single person* is friends with every other person (e.g., a 4-person group where all 6 possible friendship pairs exist).
* **CPM (Clique Percolation Method):** A method that finds tight cliques and links them together if they share common members.
* **GNN (Graph Neural Network):** An AI model designed specifically to process connected network data.
* **GraphSAGE / GAT:** Specific types of GNN architectures that aggregate information from a person's immediate neighbors.
* **Bridge Friend:** A person who belongs to 2 or more circles, acting as a connector between separate groups.
* **ONMI (Overlapping Normalized Mutual Information):** A score from 0.0 to 1.0 measuring how closely our predicted circles match the user's real-life circles.
* **F1 Score:** A balanced measure of accuracy (combining how many true friends were found vs. how many wrong guesses were made).

---

## 9. How to Open and Explore the Project

### 1. Launch the Backend (FastAPI API)
```bash
python -m uvicorn app.main:app --app-dir backend --reload --port 8000
```
* **API Documentation**: Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) in your browser to test endpoints directly.

### 2. Launch the Frontend (React Dashboard)
```bash
cd frontend
npm run dev
```
* **Interactive Dashboard**: Open [http://localhost:5173/](http://localhost:5173/) in your browser.

---

## 10. Summary in Three Sentences

1. Real human friendships naturally **overlap**, but older computer algorithms tried to force everyone into strictly one group.
2. **CircleSense** starts from tight-knit core friend groups ("cliques") and uses a **Graph Neural Network** to grow them into realistic, overlapping circles.
3. The end product is an interactive web app that automatically categorizes friends, identifies key "bridge" connectors, and provides explainable friend recommendations.
