# Causal4Application

This repository lists papers on causal for machine learning application. 

**Contributor:** [Anpeng Wu](https://scholar.google.com/citations?hl=zh-CN&user=VQ4m6zQAAAAJ).

### Contents (Actively Updating)

- [1. Causal Basics](#1-Causal-Basics)
  - [1.1. Books](#11-Books)
- [2. Causal Effect Estimation](#2-Causal-Effect-Estimation)
  - [2.1. Representation](#21-Representation)
  - [2.2. Instrumental Variable Methods](#22-Instrumental-Variable-Methods)
  - [2.3. IV Synthesis/Selection](#23-IV-Synthesis/Selection)
  - [2.4. Generative Methods](#24-Generative-Methods)
- [3. Computer Vision](#3-Computer-Vision)
- [4. Natural Language Processing](#4-Natural-Language-Processing)
- [5. Fairness](#5-Fairness)
- [6. Reinforcement Learning](#6-Reinforcement-Learning)
	- [6.1. People Directory](#61-People-Directory)
	- [6.2. Causal for RL](#62-Causal-for-RL)
	- [6.3. RL for Causal](#63-RL-for-Causal)
- [7. Others](#7-Others)
	- [7.1. Recommendation Systems](#71-Recommendation-Systems)
- [8. Conclusion](#Conclusion)
	- [8.1. People Directory](#81-People-Directory)

## 1. Causal Basics

### 1.1. Books

1. (Basic books 2018) **The Book of Why: The New Science of Cause and Effect.** Pearl and Mackenzie. 

## 2. Causal Effect Estimation

### 2.1. Representation

1. (ICML 2016) **Learning representations for counterfactual inference.** Johansson, Fredrik, Uri Shalit, and David Sontag. [[pdf](http://proceedings.mlr.press/v48/johansson16.pdf)] [[code](https://github.com/clinicalml/cfrnet)]
2. (ICML 2017) **Estimating individual treatment effect: generalization bounds and algorithms.** Shalit, Uri, Fredrik D. Johansson, and David Sontag. [[pdf](https://arxiv.org/pdf/1606.03976.pdf)] [[code](https://github.com/clinicalml/cfrnet)]
3. (NIPS 2018) **Representation learning for treatment effect estimation from observational data.** Yao, Liuyi, et al. [[pdf](https://par.nsf.gov/servlets/purl/10123149)] [[code](https://github.com/Osier-Yi/SITE)] [[meta](https://papers.nips.cc/paper/2018/hash/a50abba8132a77191791390c3eb19fe7-Abstract.html)]
4. [IJCAI 2019] **CounterFactual Regression with Importance Sampling Weights.** Negar Hassanpour, and Russell Greiner. [[pdf](https://www.ijcai.org/proceedings/2019/0815.pdf)] [[meta](https://www.ijcai.org/proceedings/2019/815)]
5. [ICLR 2019] **Learning Disentangled Representations for CounterFactual Regression.** Hassanpour, Negar, and Russell Greiner. [[pdf](https://openreview.net/attachment?id=HkxBJT4YvB&name=original_pdf)] [[code](https://www.dropbox.com/sh/vrux2exqwc9uh7k/AAAR4tlJLScPlkmPruvbrTJQa?dl=0)] [[meta](https://openreview.net/forum?id=HkxBJT4YvB)] 
6. [Arxiv 2020] **Learning decomposed representation for counterfactual inference.** Wu, Anpeng, et al. [[pdf](https://arxiv.org/pdf/2006.07040.pdf)]
7. [ICLR 2022 under-review] **Mutual Information Minimization Based Disentangled Learning Framework For Causal Effect Estimation.** Anonymous [[pdf](https://openreview.net/pdf?id=XLjtkZbYUT)] [[meta](https://openreview.net/forum?id=XLjtkZbYUT)]

### 2.2. Instrumental Variable Methods

1. [ICML 2017] **A Flexible Approach for Counterfactual Prediction.** Hartford, Jason, et al. [[pdf](http://proceedings.mlr.press/v70/hartford17a/hartford17a.pdf)] [[code](https://github.com/jhartford/DeepIV)]
2. [Arxiv 2018] **Adversarial Generalized Method of Moments.** Lewis, Greg, and Vasilis Syrgkanis. [[pdf](https://arxiv.org/pdf/1803.07164.pdf)] [[code](https://github.com/vsyrgkanis/adversarial_gmm)]
3. [NIPS 2019] **Kernel Instrumental Variable Regression.** [[pdf](https://papers.nips.cc/paper/2019/file/17b3c7061788dbe82de5abe9f6fe22b3-Paper.pdf)] [[code](https://github.com/r4hu1-5in9h/KIV)] [[meta](https://papers.nips.cc/paper/2019/hash/17b3c7061788dbe82de5abe9f6fe22b3-Abstract.html)]
4. [NIPS 2019] **Deep Generalized Method of Moments for Instrumental Variable Analysis.** Bennett, Andrew, Nathan Kallus, and Tobias Schnabel.  [[pdf](https://papers.nips.cc/paper/2019/file/15d185eaa7c954e77f5343d941e25fbd-Paper.pdf)] [[code](https://github.com/CausalML/DeepGMM)] [[meta](https://papers.nips.cc/paper/2019/hash/15d185eaa7c954e77f5343d941e25fbd-Abstract.html)]
5. [NIPS 2020] **Minimax estimation of conditional moment models.** Dikkala, Nishanth, et al. [[pdf](https://papers.nips.cc/paper/2020/file/8fcd9e5482a62a5fa130468f4cf641ef-Paper.pdf)] [[code](https://github.com/microsoft/AdversarialGMM)] [[meta](https://papers.nips.cc/paper/2020/hash/8fcd9e5482a62a5fa130468f4cf641ef-Abstract.html)]
6. [NIPS 2020] **Dual instrumental variable regression.** Muandet, Krikamol, et al. [[pdf](https://papers.nips.cc/paper/2020/file/1c383cd30b7c298ab50293adfecb7b18-Paper.pdf)] [[code](https://github.com/krikamol/DualIV-NeurIPS2020)] [[meta](https://papers.nips.cc/paper/2020/hash/1c383cd30b7c298ab50293adfecb7b18-Abstract.html)]
7. [ICDM 2019] **One-stage deep instrumental variable method for causal inference from observational data.** Lin, Adi, et al. [[pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8970756)] [[meta](https://ieeexplore.ieee.org/document/8970756)]
8. [ICLR 2021] **Learning Deep Features in Instrumental Variable Regression.** Xu, Liyuan, et al. [[pdf](https://openreview.net/pdf?id=sy4Kg_ZQmS7)] [[code](https://openreview.net/attachment?id=sy4Kg_ZQmS7&name=supplementary_material)] [[meta](https://openreview.net/forum?id=sy4Kg_ZQmS7)]
9. [ICLR 2022 under-review] **Treatment effect estimation with confounder balanced instrumental variable regression.** Anonymous [[pdf](https://openreview.net/pdf?id=zxm7rzEPaj)] [[code](https://openreview.net/attachment?id=zxm7rzEPaj&name=supplementary_material)] [[meta](https://openreview.net/forum?id=zxm7rzEPaj)]

### 2.3. IV Synthesis/Selection

1. [Statistics in medicine 2016] **Combining information on multiple instrumental variables in Mendelian randomization: comparison of allele score and summarized data methods.** Burgess, Stephen, Frank Dudbridge, and Simon G. Thompson. [[pdf](https://onlinelibrary.wiley.com/doi/pdf/10.1002/sim.6835)]
2. [AISTATS 2020] **Ivy: Instrumental variable synthesis for causal inference.** Kuang, Zhaobin, et al. [[pdf](http://proceedings.mlr.press/v108/kuang20a/kuang20a.pdf)]
3. [ICML 2021] **Valid Causal Inference with (Some) Invalid Instruments.** Hartford, Jason S., et al. [[pdf](http://proceedings.mlr.press/v139/hartford21a/hartford21a.pdf)]
4. [Entropy 2021] **The Role of Instrumental Variables in Causal Inference Based on Independence of Cause and Mechanism.** Sokolovska, Nataliya, and Pierre-Henri Wuillemin. [[pdf](https://www.mdpi.com/1099-4300/23/8/928/pdf)]
5. [Arxiv 2021] **Constructing valid instrumental variables in generalised linear causal models from directed acyclic graphs.** Hoveid, Øyvind. [[pdf](https://arxiv.org/pdf/2102.08056.pdf)]
6. [TKDD, 2022] **Auto IV: Counterfactual Prediction via Automatic Instrumental Variable Decomposition.** Yuan, Junkun, et al. [[pdf](https://arxiv.org/pdf/2107.05884.pdf)]

### 2.4. Generative Methods

1. [NIPS 2017] **Causal Effect Inference with Deep Latent-Variable Models.** Louizos, Christos, et al. [[pdf](https://papers.nips.cc/paper/2017/file/94b5bde6de888ddf9cde6748ad2523d1-Paper.pdf)] [[code](https://github.com/AMLab-Amsterdam/CEVAE)] [[meta](https://papers.nips.cc/paper/2017/hash/94b5bde6de888ddf9cde6748ad2523d1-Abstract.html)]
2. [ICLR 2018] **GANITE: Estimation of Individualized Treatment Effects using Generative Adversarial Nets.** Yoon, Jinsung, James Jordon, and Mihaela Van Der Schaar. [[pdf](https://openreview.net/pdf?id=ByKWUeWA-)] [[meta](https://openreview.net/forum?id=ByKWUeWA-)]
3. [Arxiv 2020] **Estimating the effects of continuous-valued interventions using generative adversarial networks.** Bica, Ioana, James Jordon, and Mihaela van der Schaar. [[pdf](https://arxiv.org/pdf/2002.12326.pdf)]
4. [AAAI 2020] **Treatment effect estimation with disentangled latent factors.** Zhang, Weijia, Lin Liu, and Jiuyong Li. [[pdf](https://www.aaai.org/AAAI21Papers/AAAI-155.ZhangW.pdf)] [[code](https://github.com/WeijiaZhang24/TEDVAE)]
5. [ICLR 2022 under-review] **β -Intact-VAE: Identifying and Estimating Causal Effects under Limited Overlap.** Wu, Pengzhou, and Kenji Fukumizu. [[pdf](https://openreview.net/pdf?id=q7n2RngwOM)] [[code](https://openreview.net/attachment?id=q7n2RngwOM&name=supplementary_material)] [[meta](https://openreview.net/forum?id=q7n2RngwOM)]

## 3. Computer Vision

1. [ICML 2020] **Weakly-supervised disentanglement without compromises.** Locatello, Francesco, and Schölkopf, et al.  [[pdf](http://proceedings.mlr.press/v119/locatello20a/locatello20a.pdf)] [[code](https://github.com/google-research/disentanglement_lib)]
2. [ICLR 2021] **Counterfactual Generative Networks.** [[pdf](https://openreview.net/pdf?id=BXewfAYMmJw)] [[code](https://github.com/autonomousvision/)] [[meta](https://openreview.net/forum?id=BXewfAYMmJw)]
3. [CVPR 2021] **Disentangled Representation Learning via Neural Structural Causal Models.** [[pdf](https://openaccess.thecvf.com/content/CVPR2021/papers/Yang_CausalVAE_Disentangled_Representation_Learning_via_Neural_Structural_Causal_Models_CVPR_2021_paper.pdf)] [[code](https://github.com/huawei-noah/trustworthyAI)] [[meta](https://paperswithcode.com/paper/causalvae-structured-causal-disentanglement)]

## 4. Natural Language Processing

1. (ACL 2016) **Discovery of Treatments from Text Corpora.** Christian Fong, Justin Grimmer. [[pdf](https://www.aclweb.org/anthology/P16-1151.pdf)]
2. (JMLR 2016) **Learning representations for counterfactual inference.**
	Fredrik Johansson, Uri Shalit, David Sontag. [[pdf](http://proceedings.mlr.press/v48/johansson16.pdf)]
3. (EMNLP 2017) **Detecting and Explaining Causes From Text For a Time Series Event.** Dongyeop Kang, Varun Gangal, Ang Lu, Zheng Chen, Eduard Hovy. [[pdf](https://arxiv.org/pdf/1707.08852.pdf)]
4. (NAACL 2018) **Deconfounded lexicon induction for interpretable social science.** _Reid Pryzant, Kelly Shen, Dan Jurafsky, Stefan Wagner_. [[pdf](https://www.aclweb.org/anthology/N18-1146.pdf)]
5. (EMNLP 2018) **Challenges of Using Text Classifiers for Causal Inference.** Zach Wood-Doughty, Ilya Shpitser, Mark Dredze. [[pdf](https://arxiv.org/pdf/1810.00956.pdf)]
6. (Political Analysis 2018) **Matching with text data: An experimental evaluation of methods for matching documents and of measuring match quality.** Reagan Mozer, Luke Miratrix, Aaron Russell Kaufman, L Jason Anastasopoulos. [[pdf](https://arxiv.org/pdf/1801.00644.pdf)]
7. (EMNLP 2019) **Topics to Avoid: Demoting Latent Confounds in Text Classification.** Sachin Kumar, Shuly Wintner, Noah A. Smith, Yulia Tsvetkov. [[pdf](https://arxiv.org/pdf/1909.00453.pdf)]
8. (ACL 2019) **Counterfactual Data Augmentation for Mitigating Gender Stereotypes in Languages with Rich Morphology.** Ran Zmigrod, Sabrina J. Mielke, Hanna Wallach, Ryan Cotterell. [[pdf](https://www.aclweb.org/anthology/P19-1161.pdf)]
9. (EMNLP 2019) **Weakly Supervised Multilingual Causality Extraction from Wikipedia.** Chikara Hashimoto. [[pdf](https://www.aclweb.org/anthology/D19-1296.pdf)]
10. (EMNLP 2019) **Counterfactual Story Reasoning and Generation.**
	Lianhui Qin, Antoine Bosselut, Ari Holtzman, Chandra Bhagavatula, Elizabeth Clark, Yejin Choi. [[pdf](https://arxiv.org/pdf/1909.04076.pdf)]
11. (IJCAI 2020) **Guided Generation of Cause and Effect.** Zhongyang Li, Xiao Ding, Ting Liu, J. Edward Hu, Benjamin Van Durme. [[pdf](https://www.ijcai.org/Proceedings/2020/0502.pdf)] [[video](https://www.ijcai.org/proceedings/2020/video/24610)]
12. (EMNLP 2020) **GLUCOSE: GeneraLized and COntextualized Story Explanations.** Nasrin Mostafazadeh, Aditya Kalyanpur, Lori Moon, David Buchanan, Lauren Berkowitz, Or Biran, Jennifer Chu-Carroll. [[pdf](https://arxiv.org/pdf/2009.07758.pdf)]
13. (EMNLP 2020) **Unsupervised Discovery of Implicit Gender Bias.** Anjalie Field, Yulia Tsvetkov. [[pdf](https://www.aclweb.org/anthology/2020.emnlp-main.44.pdf)]
14. (CL 2020) **CausaLM: Causal Model Explanation Through Counterfactual Language Models.** Amir Feder, Nadav Oved, Uri Shalit, Roi Reichart. [[pdf](https://arxiv.org/pdf/2005.13407.pdf)]
15. (TACL 2020) **Amnesic Probing: Behavioral Explanation with Amnesic Counterfactuals.** Yanai Elazar, Shauli Ravfogel, Alon Jacovi, Yoav Goldberg. [[pdf](https://arxiv.org/pdf/2006.00995.pdf)]
16. (CL 2020) **CausaLM: Causal Model Explanation Through Counterfactual Language Models.** Amir Feder, Nadav Oved, Uri Shalit and Roi Reichart. [[pdf](https://arxiv.org/pdf/2005.13407.pdf)]
17. (ICLR 2020) **Learning the Difference that Makes a Difference with Counterfactually-Augmented Data.** Divyansh Kaushik, Eduard Hovy, Zachary C. Lipton. [[pdf](https://arxiv.org/pdf/1909.12434.pdf)]
18. (ACL 2020) **Text and Causal Inference: A Review of Using Text to Remove Confounding from Causal Estimates.** Katherine A. Keith, David Jensen, and Brendan O'Connor. [[pdf](https://www.aclweb.org/anthology/2020.acl-main.474.pdf)]
19. (UAI 2020) **Adapting Text Embeddings for Causal Inference.** Victor Veitch, Dhanya Sridhar, David M. Blei. [[pdf](https://arxiv.org/pdf/1905.12741.pdf)]
20. (AJPS 2020) **Adjusting for confounding with text matching.**
	Margaret E Roberts, Brandon M Stewart, and Richard A Nielsen. [[pdf](http://www.mit.edu/~rnielsen/textmatching.pdf)]
21. (CSCW 2020) **Quantifying the Causal Effects of Conversational Tendencies.** Justine Zhang, Sendhil Mullainathan, Cristian Danescu-Niculescu-Mizil. [[pdf](https://arxiv.org/pdf/2009.03897.pdf)]
22. (EMNLP Oral 2021) **Causal Direction of Data Collection Matters: Implications of Causal and Anticausal Learning for NLP.** Zhijing Jin*, Julius von Kügelgen*, Jingwei Ni, Tejas Vaidhya, Ayush Kaushal, Mrinmaya Sachan, Bernhard Schölkopf. [[pdf](https://arxiv.org/pdf/2110.03618)] [[talk](https://drive.google.com/file/d/19dFDslovDFzBgkXN5lKrgV9xP8kWrB5e/)]
23. (NAACL 2021) **Counterfactual Data Augmentation for Neural Machine Translation.** Qi Liu, Matt Kusner, Phil Blunsom. [[pdf](https://www.aclweb.org/anthology/2021.naacl-main.18.pdf)]
24. (ACL 2021) **Causal Analysis of Syntactic Agreement Mechanisms in Neural Language Models.** Matthew Finlayson, Aaron Mueller, Sebastian Gehrmann, Stuart Shieber, Tal Linzen, Yonatan Belinkov. [[pdf](https://arxiv.org/pdf/2106.06087.pdf)]

## 5. Fairness

1. [ICML 2019] **Flexibly Fair Representation Learning by Disentanglement.** Creager, Elliot, et al. [[pdf](http://proceedings.mlr.press/v97/creager19a/creager19a.pdf)] [[code](https://github.com/burklight/VariationalPrivacyFairness)]

## 6. Reinforcement Learning

### 6.1. People Directory

**Elias Bareinboim** (Columbia), US. [[home page](https://causalai.net/)]

### 6.2. Causal for RL

1. [ICDL 2009] **Can reinforcement learning explain the development of causal inference in multisensory integration?** Weisswange, Thomas H., et al. [[pdf](https://www.honda-ri.de/pubs/pdf/1317.pdf)]
2. [Arxiv 2018] **Deconfounding Reinforcement Learning in Observational Settings.** Lu, Chaochao, Bernhard Schölkopf, and José Miguel Hernández-Lobato. [[pdf](https://arxiv.org/pdf/1812.10576.pdf)]
3. [Arxiv 2021] **On Instrumental Variable Regression for Deep Offline Policy Evaluation.** Chen, Yutian (Deepmind),  et al. [[pdf](https://arxiv.org/pdf/2105.10148.pdf)] [[code](https://github.com/liyuan9988/IVOPEwithACME)]
4. [Arxiv 2021] **Instrumental Variable Value Iteration for Causal Offline Reinforcement Learning.** Liao, Luofeng, et al. [[pdf](https://arxiv.org/pdf/2102.09907.pdf)]

### 6.3. RL for Causal

1. [ICLR 2020] **Causal Discovery with Reinforcement Learning.** [[pdf](https://openreview.net/pdf?id=S1g2skStPB)] [[code](https://github.com/MichelDeudon/neural-combinatorial-optimization-rl-tensorflow)] [[meta](https://openreview.net/forum?id=S1g2skStPB)]

## 7. Others

### 7.1. Recommendation Systems

1. [RecSys 2020] **Unbiased learning for the causal effect of recommendation.** Sato, Masahiro, et al. [[pdf](https://arxiv.org/pdf/2008.04563.pdf)]

## 8. Conclusion

### 8.1. People Directory

- **Judea Pearl** (UCLA), US.
- **Bernhard Schölkopf** (Max Planck Institute, MPI Tübingen), Tübingen, Germany. [[group intro](http://webdav.tuebingen.mpg.de/causality/)]
- **Dominik Janzing** (Amazon Tübingen; former: MPI Tübingen), Tübingen, Germany.
- **Joris Mooij** (University of Amsterdam; former: MPI Tübingen), Netherlands. [[home page](https://staff.fnwi.uva.nl/j.m.mooij/)]
- **Jonas Peters** (Copenhagen University; former: MPI Tübingen), Denmark. [[home page](http://web.math.ku.dk/~peters/)]
- **Peter Bühlmann** (ETH), Switzerland.
- **Marloes Maathuis** (ETH), Switzerland. [[video](https://www.youtube.com/watch?v=Q_2cgCeAjpo)]
- **Nicolai Meinshausen** (ETH), Switzerland. E.g., Anchor regression.
- **Negar Kiyavash** (EPFL), Switzerland. E.g., causal structure learning.
- **Kun Zhang** (CMU; former: MPI Tübingen), US. 
- **Peter Spirtes** (CMU Philosophy), US.
- **Cosma Shalizi** (CMU), US.
- **David Sontag** (MIT), US.
- **Caroline Uhler** (MIT), US.
- **Victor Chernozhukov** (MIT), US.
- **Elias Bareinboim** (Columbia), US. [[home page](https://causalai.net/)]
- **Andrew Gelman** (Columbia; former: UCLA with Judea Pearl), US. [[video](https://www.youtube.com/watch?time_continue=25&v=cuE9eHSbjNI&app=desktop)]
- **Ilya Shpitser** (JHU; former: UCLA with Judea Pearl), US. [[home page](https://www.cs.jhu.edu/faculty/ilya-shpitser-3/)]
- **Kosuke Imai** (Harvard; former: Princeton), US. 
- **James Robins** (Harvard), US.
- **Ferederick Eberhardt** (Caltech; former: CMU), US.
- **David Heckerman** (Amazon), US.
- **Leon Bottou** (Facebook AI), US.
- **Thomas Richardson** (UW), US.
- **Stephan Hartmann** (LMU), Munich, Germany.
- **Cheng Soon Ong** (Data61; former: MPI Tübingen), Canberra, Australia.

## Contributions

All types of contributions to this paper list is welcome. Feel free to open a Pull Request. 

Contact: [Anpeng Wu](https://scholar.google.com/citations?hl=zh-CN&user=VQ4m6zQAAAAJ), PhD of Fei Wu and Kun Kuang at Zhejiang University, working on Causal & Representation & RL. 

## How to Cite This Repo
```bibtex
@misc{causal2022anpwu,
  author = {Anpeng Wu},
  title = {Causal for Machine Learning},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/anpwu/Causal4Application}}
}
```

