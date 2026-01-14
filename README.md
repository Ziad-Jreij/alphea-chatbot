# Customer-Service-Chatbot 🤖

A **real-time intelligent customer support assistant** with advanced NLP understanding, emotion awareness, criticality detection, and human-agent collaboration.

---

## 📌 Overview

This project implements a smart customer-service chatbot capable of understanding:

- **How the customer feels** (emotion & sentiment),
- **How critical the situation is**, independently of emotion,
- **What topics and product/service aspects are being discussed**, and
- **When a human agent should be involved**.

The system combines **Emotion Detection**, **Criticality Classification**, **Aspect-Based Sentiment Analysis (ABSA)**, and **Automatic Human Escalation** into a single, unified, and explainable real-time pipeline.

---

## 🚀 Why This Project Is Novel

While many chatbots focus on intent recognition or basic sentiment analysis, existing systems do not combine **emotion awareness, criticality assessment, escalation logic, and aspect-level analysis** in real time to guide human-agent intervention.

This project uniquely integrates:

- Fine-grained **emotion detection** (GoEmotions-based)
- **Criticality scoring independent of emotion**
- Hybrid **rule-based and model-based escalation**
- **Aspect-level sentiment analysis** for products and services
- A **real-time dashboard** for human agents
- **Two automatically generated datasets** for analytics and business insights

Together, these components form a system that does not simply “chat”, but **monitors conversation risk, protects users in high-critical situations, and enables data-driven customer experience analysis**.

---

## ✨ Key Features

### 🔹 1. Emotion & Sentiment Detection
- Built using a pre-trained model fine-tuned on the **GoEmotions dataset**
- Extracts dominant emotion, sentiment polarity, and confidence scores
- Helps interpret customer attitudes such as confusion, frustration, joy, or fear

---

### 🔹 2. Criticality Classification
- Zero-shot classification of messages into **Low**, **Medium**, or **High** criticality
- Works even when sentiment is neutral or positive by focusing on semantic meaning

**Example:**
> “I think someone stole my credit card.”  
> → Neutral emotion, **High criticality**

---

### 🔹 3. Smart Escalation to Human Agents
- Maps criticality levels to escalation strategies
- Supports:
  - Immediate escalation
  - Delayed escalation
  - Monitoring without escalation
- Escalation decisions are **explainable** (emotion, criticality, chatbot uncertainty)

---

### 🔹 4. Aspect-Based Sentiment Analysis (ABSA)
- Extracts specific product or service aspects from user messages
- Assigns sentiment per aspect
- Supports multi-aspect messages
- Provides detailed feedback beyond overall sentiment

---

### 🔹 5. Human-Agent Dashboard
- Displays conversations in real time
- Shows emotion timeline, criticality score, escalation triggers, and conversation risk
- Enables agents to prioritize high-risk cases efficiently

---

## 📊 Automatically Generated Datasets

### 🗂️ 1. Conversation-Level Dataset

Each conversation is stored as a single row with fields such as:

- `conversation_id`
- `time`
- `conversation_risk`
- `overall_emotion`
- `escalated` (yes/no)
- `conversation_duration`
- `number_of_messages`

**Example insights:**
- Peak hours with high-risk conversations
- Escalation frequency and causes
- Emotion–risk correlations
- Customer satisfaction prediction using emotion and risk patterns

---

### 🗂️ 2. ABSA-Level Dataset

Each detected aspect becomes a separate entry:

- `conversation_id`
- `aspect`
- `aspect_sentiment` (positive / neutral / negative)
- `confidence_score`
- `emotion`
- `category` (Features, Cost, Performance, Support…)

**Example insights:**
- Most complained-about product features
- Aspect-level satisfaction analysis
- Emotion trends by product category
- Product improvement prioritization

---

## 🛠️ Running the Project

### 1️⃣ Activate the Virtual Environment
```bash
source .venv/bin/activate
```
## 2️⃣ Train the Rasa Model

```bash
rasa data validate
rasa train
```
## 3️⃣ Run the System (Multiple Terminals)

### Terminal A – Action Server
```bash
rasa run actions
```
### Terminal B – Rasa Server
```bash
rasa run --enable-api --cors "*" --debug
```
### Terminal C – Web Chat Interface
```bash
python3 -m http.server 8000
```
Open in your browser:

http://127.0.0.1:8000/webchat.html

### Terminal D – Dashboard Backend
```bash
python3 dashboard_backend.py
```
Open in your browser:

http://127.0.0.1:8000/dashboard.html
