# LAMIC
This repository contains the implementation for the paper **"LAMIC: Locating Relevant API Knowledge with Multi-Perspective Demonstration Enhancement In-Context Learning."**

LAMIC constructs <API, KU> tuples from API tutorials and Stack Overflow. It then applies a multi-perspective demonstration enhancement ICL framework with two main components. The multi-perspective demonstration selection strategy integrates lexical, semantic, and structural rankings to retrieve informative demonstrations. The clue-based demonstration enhancement strategy enriches each retrieved demonstration with explicit clues and a reasoning statement. Finally, LAMIC prompts the LLM with the enhanced demonstrations to locate relevant KUs for the target API.


## Repository Structure

- `LAMIC/`: data loading, multi-perspective demonstration retrieval, clue-enhanced demonstrations, in-context learning, evaluation, and research-question experiment orchestration.
- `requirements.txt`: Python dependencies required to run the experiments.

## Supplementary Materials

The appendix PDF, application study materials, user study materials, and manual verification results for irrelevant Stack Overflow API–Q&A pairs are available in the [LAMIC Supplementary Materials folder on Google Drive](https://drive.google.com/drive/folders/1Co7bTHHrbNyp3V5C2NSWc14c7yzYLzE0).
