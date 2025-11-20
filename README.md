Customer-Service-Chatbot
A real-time, intelligent customer support assistant with advanced NLP understanding, emotion awareness, and human-agent collaboration.

Overview

This project develops a smart customer-service chatbot capable of understanding how the customer feels, how critical their situation is, what topics they are talking about, and when to involve a human agent.
It combines Emotion Detection, Sentiment Analysis, Criticality Classification, Automatic Escalation, and Aspect-Based Sentiment Analysis (ABSA) into one unified system.

Why this project is novel

While many chatbots can classify intent or detect basic sentiment, no existing work combines all four capabilities in real-time to guide human-agent escalation.
This project is the first to integrate:

Fine-grained emotion understanding (GoEmotions-based)

Criticality scoring that is independent of emotion

Rule-based and model-based escalation mapping

Aspect-level analysis of products/services

A live dashboard to help human agents see risk, emotions, and escalation triggers

Two automatically generated datasets for future analytics and business insights

Together, these create a system that does not just “chat”, but monitors the conversation, protects users during high-risk cases, and gives companies a powerful way to analyze customer experience.

Key Features
🔹 1. Emotion & Sentiment Detection

Built using a pre-trained model on the GoEmotions dataset

Extracts the top emotion, sentiment, and confidence scores

Helps understand customer attitude (e.g., confusion, joy, annoyance, fear)

🔹 2. Criticality Classification

Zero-shot classification to detect whether a message is Low, Medium, or High critical

Works even when the sentiment is positive or ambiguous, by focusing on the meaning

Example:

“I think someone stole my credit card.” → Neutral emotion but High criticality

🔹 3. Smart Escalation to Human Agents

Maps criticality levels to escalation rules

Can trigger:

Immediate escalation

Delayed escalation

Monitoring without escalation

Human agents can see why escalation happened (emotion, criticality, chatbot uncertainty…)

🔹 4. Aspect-Based Sentiment Analysis (ABSA)

Extracts specific product or service aspects mentioned by the user

Detects sentiment per aspect

Supports multi-aspect messages

Helps companies understand detailed customer feedback beyond “overall sentiment”

🔹 5. Human-Agent Dashboard

Shows conversations in real time

Displays:

Emotion timeline

Criticality score

Escalation triggers

Conversation risk score

Helps agents quickly prioritize high-risk customers

Automatically Generated Datasets

The system produces two datasets that companies can use for analytics, reporting, and long-term decision-making.

1. Conversation-Level Dataset

Each conversation becomes one row with columns such as:

conversation_id

time

conversation_risk

overall_emotion

escalated (yes/no)

conversation_duration

number_of_messages

Insights companies can extract:

Peak hours with most high-risk conversations

How many conversations were escalated and why

Correlation between emotion patterns and customer churn

Customer satisfaction prediction using risk + emotion patterns

2. ABSA-Level Dataset

Each aspect mentioned in the conversation becomes one entry:

conversation_id

aspect (e.g., battery life, delivery time, screen quality…)

aspect_sentiment (positive, neutral, negative…)

confidence_score

emotion

category (Features, Cost, Performance, Support…)

Insights companies can extract:

Which product features customers praise or complain about

Aspect-level satisfaction (e.g., “Battery Life is the #1 complaint”)

Emotion trends per product category

Prioritization for product improvements
