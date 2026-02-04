# AI-Augmented SIEM Security: Adversarial Log Injection and RAG Robustness

## Overview

Modern Security Information and Event Management (SIEM) platforms increasingly integrate Large Language Models (LLMs) to assist analysts with log triage, threat interpretation, and response planning. Retrieval-Augmented Generation (RAG) enables these systems to contextualize alerts using historical logs, threat intelligence, and playbooks.

While these systems improve analyst productivity, they introduce a new and largely unstudied attack surface:

> **Adversarial telemetry injection through attacker-controlled logs.**

This project investigates the security, robustness, and reliability of AI-augmented SIEM pipelines. Specifically, it evaluates how malicious or adversarial log content can manipulate retrieval and downstream LLM reasoning to influence analyst guidance and automated response recommendations.

---

## Project Goals

This project aims to:

1. **Build a RAG-based SIEM copilot prototype**
   - Integrate with a real SIEM environment
   - Provide analyst-style threat summaries and action recommendations

2. **Develop an adversarial log injection framework**
   - Simulate attacker-controlled telemetry
   - Generate structured prompt injection attacks through log artifacts

3. **Measure failure modes in AI-assisted SOC pipelines**
   - Quantify injection success rates
   - Identify vulnerable log fields and retrieval behaviors
   - Evaluate impact on analyst decision recommendations

4. **Evaluate mitigation strategies**
   - Log sanitization techniques
   - Retrieval trust filtering
   - Prompt isolation and structured reasoning approaches
   - Multi-stage verification pipelines

---

## Research Motivation

Traditional prompt injection research focuses primarily on direct user inputs or web retrieval. However, in real security operations centers (SOCs), attackers frequently control telemetry data such as:

- Filenames
- Command arguments
- Process metadata
- Network payload logs
- Alert descriptions

When these artifacts are incorporated into RAG pipelines, they may:

- Override system prompts
- Redirect AI analysis
- Inflate or suppress threat severity
- Produce misleading remediation guidance

This project systematically studies these risks in operational SIEM environments.

---

## Research Questions

This project explores the following questions:

### RQ1 — Injection Vulnerability
How susceptible are RAG-based SIEM copilots to adversarial telemetry injection?

### RQ2 — Retrieval Amplification
How does retrieval strategy influence injection success and model manipulation?

### RQ3 — Attack Surface Characterization
Which telemetry fields produce the highest rates of successful prompt injection?

### RQ4 — Defensive Effectiveness
Which mitigation strategies reduce injection success while preserving analyst utility?

---

Security Telemetry (Logs / Alerts)
↓
Wazuh SIEM Pipeline
↓
Log Retrieval Engine
↓
Retrieval-Augmented Context Builder
↓
LLM Reasoning Engine
↓
Analyst Guidance / Action Planning


The project focuses on adversarial manipulation across this pipeline.

---

## Key Contributions

This project aims to provide:

- A reproducible framework for adversarial log injection testing
- A structured taxonomy of telemetry-based prompt injection attacks
- Empirical evaluation of RAG vulnerability in SIEM environments
- Defense strategies for secure AI-augmented SOC workflows
- A prototype integration demonstrating real-world applicability

---

## Attack Taxonomy (Planned)

The project evaluates multiple classes of injection attacks, including:

### Prompt Termination Attacks
Attempts to prematurely override system instructions using log artifacts.

### Instruction Override Attacks
Logs containing malicious directives targeting LLM reasoning.

### Context Poisoning
Injection of false or misleading threat context into retrieval results.

### Severity Manipulation
Logs crafted to alter threat prioritization or analyst urgency.

### Trust Impersonation
Telemetry designed to appear as trusted threat intelligence or internal policy data.

---

## Evaluation Metrics

The project measures system robustness using:

- Injection Success Rate
- Analyst Recommendation Deviation
- Threat Severity Distortion
- Retrieval Poisoning Effectiveness
- Mitigation Defense Effectiveness
- Response Consistency and Reliability

---

## Implementation Components

### SIEM Integration
- Log ingestion and alert handling
- SOC workflow simulation
- Alert enrichment pipeline

### RAG Pipeline
- Log and threat intelligence retrieval
- Context ranking and filtering
- Structured prompt construction

### Adversarial Injection Engine
- Synthetic attack generation
- Telemetry manipulation tooling
- Automated evaluation harness

### Defense Mechanisms
- Log sanitization
- Context validation
- Multi-stage prompt isolation
- Trust scoring for retrieval outputs

---

## Expected Impact

This work contributes to emerging research in:

- Security of AI-assisted cybersecurity tools
- Robustness of retrieval-augmented generation systems
- Adversarial machine learning in operational environments
- Human-AI interaction in security decision workflows

---

## Project Status

🚧 Active Research and Development

### Planned Phases

1. SIEM + RAG prototype development  
2. Adversarial telemetry injection framework  
3. Experimental evaluation  
4. Defense strategy implementation  
5. Publication preparation  

---

## Future Work

Potential extensions include:

- Multi-agent verification architectures
- Real-time streaming injection detection
- Analyst behavioral trust studies
- Cross-SIEM platform evaluation
- Open benchmark dataset release


## System Architecture

