[![Live Demo](https://img.shields.io/badge/Live_Demo-TruthScan_AI-brightgreen?style=for-the-badge&logo=vercel)](https://truth-ai-nu.vercel.app/)

TruthScan AI 📰🔍
TruthScan AI is an advanced fake news detection application that utilizes a hybrid architecture, combining classical machine learning models with Large Language Model (LLM) fact-grounding to analyze news authenticity.

🚀 Overview
The system provides a transparent, explainable verdict on news content by leveraging:

Hybrid Analysis: A secondary classical machine learning reference model (e.g., Decision Tree, Logistic Regression) paired with primary analysis by LLaMA 3.3 70B.

Fact-Grounding (RAG): Integrates live web searches via duckduckgo-search to verify claims against current information rather than relying solely on static training data.

Explainable Results: Outputs include specific confidence scores (Claim Accuracy, Language Quality, Source Credibility), AI-generated summaries, and self-critique.

🏗️ Technical Architecture
The application is built using the following stack:

Backend: A app.py script powers the Flask web server, handling article extraction, model inference, and database interactions.

Containerization: The project is configured for deployment using a Dockerfile, with dependencies managed through packages.txt and requirements.txt.

Deployment Configuration: Uses gunicorn for production-ready server management, with environment-specific settings defined in runtime.txt.

Frontend: The user interface is rendered via index.html, which communicates with the backend via JSON to display dynamic analysis results and prediction history.

Storage: A local sqlite3 database (history.db) stores user analysis history.

🛠️ Key Files
app.py: Main Flask application logic, including the article scraping pipeline and LLM integration.

Dockerfile: Instructions for containerizing the application for environments like Hugging Face Spaces.

packages.txt: List of required system-level dependencies.

requirements.txt: List of Python library dependencies (e.g., flask, groq, scikit-learn).

index.html: Frontend template providing the user interface for text/URL input and dashboard visualization.

dockerignore: Specifies files to exclude from the container build.

💡 How It Works
Input: Users provide either text or a URL via index.html.

Extraction: app.py employs a pipeline utilizing trafilatura, newspaper3k, and BeautifulSoup to extract clean article text.

Analysis: The text is processed by a selected ML model for a baseline classification and then sent to Groq's LLaMA 3.3 70B for deep reasoning and fact-checking.

Reporting: Results are saved to history.db and rendered on the index.html dashboard, showing the final verdict, confidence metrics, and reasoning.

Developed by Tilal Ahmed
