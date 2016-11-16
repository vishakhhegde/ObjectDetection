#Scripts to Build the Model PB files

1. InceptionResNetV2
- Download the latest checkpoint file and put it in the model_pb_generation directory
- Run
	python build_inception_resnet.py <checkpoint-file-path> <output-pb-file-path>
	Example: python build_inception_resnet.py /home/ubuntu/matroid/scripts/retrain_files/model_pb_generation_scripts/inception_resnet_v2_2016_08_30.ckpt /home/ubuntu/matroid/scripts/retrain_files/model_pb_generation_scripts/inception_resnet_v2.pb
- It generates the pre-trained model with weights from file specified into the output-pb-file-path