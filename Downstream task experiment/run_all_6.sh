export EXACT_PROPORTION=0.2
export CHEBYSHEV_DEGREE=6

python run_chebyshev_ARC_easy.py \
  --model_name "/work/vb21/haochen/code/LLaDA-8B-Instruct-BnS"


python run_chebyshev_ARC_challenge.py \
  --model_name "/work/vb21/haochen/code/LLaDA-8B-Instruct-BnS"


python run_chebyshev_GPQA.py \
  --model_name "/work/vb21/haochen/code/LLaDA-8B-Instruct-BnS"


python run_chebyshev_hellaswag.py \
  --model_name "/work/vb21/haochen/code/LLaDA-8B-Instruct-BnS"


python run_chebyshev_llmu.py \
  --model_name "/work/vb21/haochen/code/LLaDA-8B-Instruct-BnS"





