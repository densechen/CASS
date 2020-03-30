# echo "EVAL CASS ..."
# python ./tools/eval.py --resume_model cass_best.pth --dataset_dir ../nocs --cuda --save_dir ../predicted_result --eval --mode cass

echo "EVAL CASS ..."
python ./tools/eval.py --save_dir ../predicted_result --mode cass


echo "EVAL NOCS ..."
python ./tools/eval.py --save_dir ../predicted_result --mode nocs
