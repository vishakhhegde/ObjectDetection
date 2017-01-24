# A Novel Object Detection Pipeline (work in progress)
The goal of this project is to explore object detection pipelines with the hope that we find something better that the current state of the art (the RCNN family).

An important observation is that most object detection pipelines consider the background to be just another class. However, since background is present in most natural images, it is intuitively clear that they have to be treated differently. We are guided by this intuition and introduce new loss functions to take this intuition into account.

## Current Object Detection Pipelines
All of the recent object detection algorithms (RCNN family) have the following pipeline:
1. Region Proposal (either through a dedicated neural network, or the same neural network which performs classification)
2. Classification
3. Aggregation of results, in which only a few bounding boxes which have a high degree of confidence are output.

The classification part of the pipeline is exactly the same as canonical image classification; they treat background regions as another class while training and during inference.

However, since background forms a significant part of most natural images (in many cases, forms a majority part of the image), it is important to treat them different from regular objects.

## Our classification pipeline

Our classification pipeline is novel, in the sense that we introduce a spherical hinge loss which forces the embedding (or features) of non-objects to have a norm zero. On the other hand, it forces the features of objects to have a unit norm. Another way to look at this is that our neural network only activates when it sees and object and does not activate for background regions.

This was part of a seminar course on representation learning. We have a report for it. Can be accessed at: https://www.dropbox.com/s/21ny6hoeb1zbdqw/project-report-331b.pdf?dl=0

If you find any bugs in the code, or are not clear about the highlevel idea of this project, please get in touch with me!

I will be working on making the code more readable. More updates coming soon! Stay tuned!

