## Meta-learning co-training
Co-training is a semi-supervised learning setting where two learners (agents), trained on different perspectives of the data, iteratively label additional samples
The rationale of this approach is that the different learner perspectives will producein a more diverse labeled set, resulting in more effective classifiers. 
While co-training proved effective in multiple cases, the labeling mechanisms used by existing approaches are heuristic and error-prone. 

We present here CoMet, a meta learning-based co-training algorithm. 
CoMet utilizes meta-models trainedon previously-analyzed datasets to select the samples to be labeledfor the current dataset. 


![Meta-features extraction](https://github.com/Zaksg/MetaCoTrainingV2/blob/master/comet_framework.png)

