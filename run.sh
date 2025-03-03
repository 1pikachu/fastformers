pip install onnxruntime==1.8.0
pip uninstall transformers -y
python setup.py install


ln -s /home2/pytorch-broad-models/fastformers/student-4L-312 student-4L-312

pruned_student_model=student-4L-312
out_dir="./output_dir"
data_dir="SuperGLUE/BoolQ"
python3 examples/fastformers/run_superglue.py \
        --model_type bert \
        --model_name_or_path ${pruned_student_model} \
        --task_name BoolQ --output_dir ${out_dir} --do_eval \
        --data_dir ${data_dir} --per_instance_eval_batch_size 1 \
        --do_lower_case --max_seq_length 512 \
        --threads_per_instance 1 --no_cuda
