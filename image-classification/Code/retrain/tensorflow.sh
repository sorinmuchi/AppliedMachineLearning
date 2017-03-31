docker run -it -v $HOME/tf_files:/tf_files  gcr.io/tensorflow/tensorflow:latest-devel
cd /tensorflow
git pull
bazel build --config=opt tensorflow/examples/image_retraining:retrain
python tensorflow/examples/image_retraining/retrain.py \
--bottleneck_dir=/tf_files/bottlenecks \
--how_many_training_steps 32000 \
--model_dir=/tf_files/inception \
--output_graph=/tf_files/retrained_graph.pb \
--output_labels=/tf_files/retrained_labels.txt \
--image_dir /tf_files/tiny_imagenet \
--print_misclassified_test_images \
pip install pandas
# create predictions csv
cd /tf_files/test_tiny_imagenet
python /tf_files/bulk_predictions.py tf_files/test_tiny_imagenet
# upload predictions csv to server
curl --upload-file /tf_files/submit.csv https://transfer.sh/submit.csv