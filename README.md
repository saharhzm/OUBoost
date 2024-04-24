OUBoost: boosting based over and under sampling technique for handling imbalanced data 2023

SH Mostafaei, J Tanha

International Journal of Machine Learning and Cybernetics 14 (10), 3393-3411

https://link.springer.com/article/10.1007/s13042-023-01839-0

![architecture](https://github.com/saharhzm/OUBoost/assets/74831239/b3000139-6fef-43d1-aa2b-2a3b8d99da86)
Most real-world datasets usually contain imbalanced data. Learning from datasets where the number of samples in one class (minority) is much smaller than in another class (majority) creates biased classifiers to the majority class. The overall prediction accuracy in imbalanced datasets is higher than 90%, while this accuracy is relatively lower for minority classes. In this paper, we first propose a new technique for under-sampling based on the Peak clustering method from the majority class on imbalanced datasets. We then propose a novel boosting-based algorithm for learning from imbalanced datasets, based on a combination of the proposed Peak under-sampling algorithm and over-sampling technique (SMOTE) in the boosting procedure, named OUBoost. In the proposed OUBoost algorithm, misclassified examples are not given equal weights. OUBoost selects useful examples from the majority class and creates synthetic examples for the minority class. In fact, it indirectly updates the weights of samples. We designed experiments using several evaluation metrics, such as Recall, MCC, Gmean, and F-score on 30 real-world imbalanced datasets. The results show improved prediction performance in the minority class in most used datasets using OUBoost. We further report time comparisons and statistical tests to analyze our proposed algorithm in more details.
