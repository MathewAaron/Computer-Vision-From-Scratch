# Adaboost-From-Scratch
"""
@author: Aaron
"""

## Adaboost using HAAR features

Developing an EM based Adaboost algorithm using HAAR features.
AdaBoost: an abbreviation for Adaptive Boosting uses a subset of features and constructs an array of “weak classifiers”. 
The weak classifiers are computed by using decision trees / decision “stumps”. 
Where the error is computed at each iteration and the weights are updated based on those errors to extract the maximum error index.

The algorithm can be derived as per the Viola-Jones paper:

![01](https://user-images.githubusercontent.com/71589098/175185087-62b12a02-c27c-424c-8153-bb08b7d31037.png)
Reference taken from paper [1].


### HAAR features
![02](https://user-images.githubusercontent.com/71589098/175185098-bcdb06eb-ad71-49cf-aeac-29807b8eccd2.png)


## Results

### Posterior Probability and Accuracy
![04](https://user-images.githubusercontent.com/71589098/175185133-f638f96b-c49b-4228-b557-690d98822709.png)


### ROC curve
![03](https://user-images.githubusercontent.com/71589098/175185145-2dfd6875-72eb-464c-b86c-27352bb8bfa9.png)


## Reference

[1] Viola, P., Jones M. Rapid object detection using a boosted cascade of simple features. In: 2001 IEEE Computer Society Conference on Computer Vision and Pattern Recognition, 2011, IEEE, pp. 511-518.
