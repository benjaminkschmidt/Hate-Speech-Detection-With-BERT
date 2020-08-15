# Hate-Speech-Detection-With-BERT
In automatic detection of hate speech on social media, one of the largest challenges is to only detect hate speech and offensive language but not miscategorize anti-racist speech. As anti-racist speech generally uses similar language as hate speech it is seeking to dismantle, the probability of false positives and miscategorization is a major challenge. Twitter offers the most apparent location where such conversation and miscategorization could occur. 
In the wake of the #BlackLivesMatter protests following the death of George Floyd in May 2020, the conversation and national discussion surrounding racism and anti-racism.  
Using the twitter dataset gathered by Davidson, Thomas et al. ("Automated Hate Speech Detection and the Problem of Offensive Language.") and an additional set of categorized anti-racist tweets, modeling could be used to create a model that uses three labels (hate speech, offensive non-hate speech, and non-offensive non-hate speech) and can be tested with anti-racist speech that was previously labeled as non-offenisive non-hate speech. 

By using Bag of Words modeling and BERT modeling, two major approaches to transfer learning were attempted and compared. 
For details of results, review written documentation on this repository. 
To recreate results, clone the modeling repository, using the appropriate relative pathways on local system. A Pytorch optimized GPU instance is recommended for running BERT models. 

This github repository is created in partial completion of the Masters of Science in Data Science at The George Washington University. 
