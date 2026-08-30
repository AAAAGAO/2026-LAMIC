# LAMIC
This repository contains the implementation and experimental materials for the paper **"LAMIC: Locating Relevant API Knowledge with Multi-Perspective Demonstration Enhancement In-Context Learning."**

LAMIC constructs $\langle$API, KU$\rangle$ tuples from API tutorials and Stack Overflow. It then applies a multi-perspective demonstration enhancement ICL framework with two main components. The multi-perspective demonstration selection strategy integrates lexical, semantic, and structural rankings to retrieve informative demonstrations. The clue-based demonstration enhancement strategy enriches each retrieved demonstration with explicit clues and a reasoning statement. Finally, LAMIC prompts the LLM with the enhanced demonstrations to locate relevant KUs for the target API.


## Repository Structure

- `LAMIC/`: data loading, multi-perspective demonstration retrieval, clue-enhanced demonstrations, in-context learning, evaluation, and research-question experiment orchestration.
- `requirements.txt`: Python dependencies required to run the experiments.
- `application/`: materials used in the API knowledge retrieval application study.
- `user-study/`: anonymized evaluation sheets and ground-truth data used in the user study.


## Datasets

The repository follows the datasets and construction procedures reported in the source papers. Dataset access and detailed preprocessing descriptions are available from the corresponding publications:

- API tutorial datasets: [A More Accurate Model for Finding Tutorial Segments Explaining APIs](https://doi.org/10.1109/SANER.2016.59).
- Stack Overflow datasets: [Retrieving API Knowledge from Tutorials and Stack Overflow Based on Natural Language Queries](https://doi.org/10.1145/3565799).
