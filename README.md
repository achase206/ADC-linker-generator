# Optimizing ADC Linkers via LSTM Reinforcement Learning

Alex Chase achase95@berkeley.edu <br>
Leah Sutherland leah_sutherland@berkeley.edu <br>
Sana Khan skhan2016@berkeley.edu <br>
Natalia Rivera nataliarivera@berkeley.edu <br>
Hailey Monaco haileymonaco@berkeley.edu <br>
Jonas Kazimli kazimli@berkeley.edu

## Abstract
Antibody-drug conjugates (ADCs) are targeted cancer therapies, made up of a monoclonal antibody (mAb), cytotoxic payload, and linker. To address the need for rapid and optimized linker generation, we developed a long short-term memory (LSTM) model which implements reinforcement learning to generate chemically valid and structurally optimal ADC linkers for specific antibody-conjugate pairs. The model was initially trained on a dataset of approximately 13,000 real and synthetic ADC linkers. Sequences were generated as self-referencing embedding strings (SELFIES) which guarantees chemically valid molecules, as opposed to SMILES which rely on complex syntax that is difficult for generative models to learn and apply. Linker specific motifs including heads, cleavable regions and tail groups were given their own unique tokens to ensure fidelity and to assess whether linkers were structurally complete. The generative model was then trained via reinforcement learning (RL) to optimize linker generation for desirable chemical properties and ADC specific conditions such as antibody and payload attachment and cancer target. The RL trained model produces chemically valid and unique linkers above 90% for all generated sequences and achieves a low Tanimoto similarity of less than 20%. Average chemical properties related to ease of synthesis and solubility are comparable to transformer based linker generation models. Additionally, compared to pre-RL training, the post-RL model generally produced better results for all chemical properties while maintaining high structural uniqueness. This work provides a foundational model for generating novel ADC linkers which could expedite ADC development for modern precision medicine. This model still lags behind transformer based approaches in generating structures with high drug-likeness (QED) and future work will aim to address these shortcomings. Long-term objectives for this model are to adopt a graph neural network (GNNs) architecture, utilizing transformer based predictions, and to extend desirable linker properties to include metabolite toxicity in ADC-specific tissue environments.

## Setup and Running Model
Create environment. If you have all the dependencies downloaded you can skip this. CUDA related libraries are very large...

If GPU/CUDA available:

`conda env create -f environment_gpu.yaml`

If running on CPU:

`conda env create -f environment_cpu.yaml`

Activate the environment:

`conda activate lstm_model`

Open launcher:<br>
Includes options to train new models or run existing models.

`bash launcher.sh`

## File Details

LSTM_utils.py:
Contains utility functions to condition tokens, canonicalize motifs, tag datasets, clean SELFIES/SMILES, and compute fingerprints.

LSTM_classes.py:
Defines Tokenizer, ADCDataset, LSTM Generator, Critic, and Reinforcement Learning classes

LSTM_helpers.py
Pretrains the LSTM generator and finetunes the LSTM generator via PPO.

LSTM_train.py
Trains the LSTM generator model.

LSTM_run.py
Loads trained LSTM generator model and generates ADC linker candidates.

### Project References
1.  Noriega, H. & Wang, X. AI-driven innovation in antibody-drug conjugate design. Front. Drug Discov. 5 (2025).
2. Alas, M. & Saghaeidehkordi, A. & Kaur, K. Peptide-Drug Conjugates with Different Linkers for Cancer Therapy. J Med Chem 64, 216-232 (2021).
3. Shen, L. T. & et al. ADCdb: the database of antibody-drug conjugates. Nucleic Acids Research. 52(D1): D1097-D1109 (2024).
4. Su, A. & Luo, Y. & Zhang, C. & Duan, H. Linker-GPT: design of Antibodydrug conjugates linkers with molecular generators and reinforcement learning. Nature Sci. Reports 15 (2025).
5. Segler, M. H. S., Kogej, T., Tyrchan, C. & Waller, M. P. Generating focused molecule libraries for drug discovery with recurrent neural networks. ACS Cent. Sci. 4, 120–131 (2017).
6. Shen, T. et al. Automoldesigner for antibiotic discovery: an Ai-Based Open-Source software for automated design of Small-Molecule antibiotics. J. Chem. Inf. Model. 64, 575–583 (2024).
7. Lab, D. (2025). SP3-hybridized carbons - MolModa Documentation. Pitt.edu. https://durrantlab.pitt.edu/molmoda/docs/structures/molprop/csp3/
8. Leon, M., Perezhohin, Y., Peres, F. et al. Comparing SMILES and SELFIES tokenization for enhanced chemical language modeling. Sci Rep 14, 25016 (2024). https://doi.org/10.1038/s41598-024-76440-8
9. Li, Haoran, and Wei Lu. “Mixed Cross Entropy Loss for Neural Machine Translation.” arXiv e-prints, 2021, arXiv:2106.15880. 
10. Bou, Albert, et al. "TorchRL: A data-driven decision-making library for PyTorch." ICLR 2024 Proceedings, 2024. arXiv e-prints, arXiv:2306.00577. 
11. Landrum, G. "RDKit: Open-source cheminformatics". http://www.rdkit.org
12. Sinii, Viacheslav, et al. "Steering LLM Reasoning Through Bias-Only Adaptation." 2025. arXiv e-prints, arXiv:2505.18706.
13. Machine Learning in Energy Chemistry: Introduction, Challenge and Perspective - Scientific Figure on ResearchGate. Available from: https://www.researchgate.net/figure/a-Specific-representation-coding-and-differences-between-SELFIES-and-SMILES-with_fig2_370242629 


























