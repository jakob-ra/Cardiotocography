# Cardiotocography
Cardiotocography (CTG) is a technique to record fetal heartbeat and uterine contractions during pregnancy. It is typically used in the third trimester, in which the fetus is prone to sudden drops in oxygen supply (hypoxia). This can lead to death and long-term disability. Based on CTG data, doctors decide whether to perform an intervention, such as a Caesarian section. Doctors typically look at CTG results that have already gone through a signal processing algorithm, and make their decision based on this summary and their medical expertise. 

In this project, we aim to predict fetal health status from raw CTG data. The proposed predictive model can help doctors' decision process by providing a simple binary classification of the fetus' status: normal or suspect/pathological. The goal study is therefore not to automate diagnosis but to provide an additional piece of information that might prevent doctors from prescribing unnecessary C-sections or, even worse, fail to intervene when the fetus is in distress.

A project summary can be found at:

https://www.kaggle.com/jakrau/predicting-fetal-health-with-cardiotocography-data

along with a brief theoretical overview over different classifiers, including support vector machines (SVM), random forests, and boosted decision trees. It also discusses the 'proper' way to do cross-validation and touches upon how to deal with imbalanced data.
