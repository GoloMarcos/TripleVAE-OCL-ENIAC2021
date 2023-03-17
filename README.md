# Triple-VAE: A Triple Variational Autoencoder to Represent Events in One-Class Event Detection 
- Marcos Gôlo (ICMC/USP) | marcosgolo@usp.br
- Rafael Rossi (UFMS) | rafael.g.rossi@ufms.br
- Ricardo Marcacini (ICMC/USP) | ricardo.marcacini@icmc.usp.br

# Citing:

If you use any part of this code in your research, please cite it using the following BibTex entry
```latex
@inproceedings{ref:Golo2021,
  title={Triple-VAE: A Triple Variational Autoencoder to Represent Events in One-Class Event Detection},
  author={Gôlo, Marcos PS and Rossi, Rafael G and Marcacini, Ricardo M},
  booktitle={Anais do XVIII Encontro Nacional de Inteligência Artificial e Computacional},
  pages={643--654},
  year={2021},
  organization={SBC}
}
```

# Abstract
Events are phenomena that occur at a specific time and place. Its detection can bring benefits to society since it is possible to extract knowledge from these events. Event detection is a multimodal task since these events have textual, geographical, and temporal components. Most multimodal research in the literature uses the concatenation of the components to represent the events. These approaches use multi-class or binary learning to detect events of interest which intensifies the user's labeling effort, where the user should label event classes even if there is no interest in detecting them. In this paper, we present the Triple-VAE approach that learn a unified representation from textual, spatial, and density modalities through a variational autoencoder, which is one of the state-of-the-art in representation learning. Our proposed Triple-VAE obtains suitable event representations for one-class classification, where users provide labels only for events of interest, thereby reducing the labeling effort. We carried out an experimental evaluation with ten real-world event datasets, four multimodal representation methods, and five evaluation metrics. Triple-VAE outperforms and presents a statistically significant difference considering the other three representation methods in all datasets. Therefore, Triple-VAE proved to be promising to represent the events in the one-class event detection scenario.

# Proposal: TripleVAE + One-Class Learning
![Proposal](/images/TVAE.png)

# Results
![Results](/images/results.png)

# Critical Diference
![Results](/images/nemenyi.png)








