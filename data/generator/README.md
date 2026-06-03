---
license: mit
task_categories:
- text-generation
language:
- en
size_categories:
- 100K<n<1M
---

This is a training data file for [Self-RAG](https://selfrag.github.io/) that generates outputs to diverse user queries as well as reflection tokens to call the retrieval system adaptively and criticize its own output and retrieved passages.

Self-RAG is trained on our 150k diverse instruction-output pairs with interleaving passages and reflection tokens using the standard next-token prediction objective, enabling efficient and stable learning with fine-grained feedback.
At inference, we leverage reflection tokens covering diverse aspects of generations to sample the best output aligning users' preferences. See full descriptions in [our paper](https://arxiv.org/abs/2310.11511) and [code](https://github.com/AkariAsai/self-rag).

## Citation and contact
If you use this model, please cite our work: 
```
@article{asai2023selfrag,
  author    = {Asai, Akari and Wu, Zeqiu and Wang, Yizhong and Sil, Avirup and Hajishirzi, Hannaneh},
  title     = {{Self-RAG}: Learning to Retrieve, Generate, and Critique through Self-Reflection},
  year      = {2023},
  journal   = { arXiv preprint arXiv:2310.11511 },
  URL       = {https://arxiv.org/abs/2310.11511}
}
```