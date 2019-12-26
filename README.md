# Focal Loss

This is the keras implementation of focal loss proposed by Lin et. al. in their Focal Loss for Dense Object Detection paper.

![focal loss](image/focal_loss.png)

## Usage

Compile your model with focal loss as sample:

Binary

`model.compile(loss=[binary_focal_loss(gamma=2)], metrics=["accuracy"], optimizer="adam")`

Categorical

`model.compile(loss=[categorical_focal_loss(gamma=2)], metrics=["accuracy"], optimizer="adam")`

alpha setting:

`model.fit(class_weight={0:alpha0, 1:alpha1, ...}, ...)` the class_weight is the alpha of focal loss.

## References

The implementation is based [@umbertogriffo](https://github.com/umbertogriffo/focal-loss-keras)
