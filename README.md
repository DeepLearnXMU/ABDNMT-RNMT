#### Introduction

Zhang, Xiangwen, Jinsong Su, Yue Qin, Yang Liu, Rongrong Ji, and Hongji Wang. "Asynchronous bidirectional decoding for neural machine translation." In Thirty-Second AAAI Conference on Artificial Intelligence. 2018.

#### Requirements

Python 3.7+ is required.

Install requirements and optionally install thseq.
```shell
pip install -r requirements.txt
```

#### Train

CUDA_VISIBLE_DEVICES=0 python -u abdtrain.py --train /path/to/train.en /path/to/train.de.pseudo.r2l /path/to/train.de --dev /path/to/dev.en /path/to/dev.de.pseudo.r2l /path/to/dev.de --vocab /path/to/vocab.en /path/to/vocab.de --vocab-size 99999 --model trained/run --max-step 200000 --eval-steps 5000 --shuffle 1 --batch-size 3000 3000 --batch-by-sentence 0 --arch abdrnn --criterion smoothed-ce --lr-scheduler noam --model-size 512 --warmup-steps 8000 --lr 1 --weight-tying 1 --accumulate 2 --ver 1

#### Translate

python -u translate.py --models trained/run -b 1 \
--input /path/to/test.en /path/to/test.de.pseudo.r2l > /path/to/test.de.mt

#### Citation

@inproceedings{zhang2018asynchronous,
  title={Asynchronous bidirectional decoding for neural machine translation},
  author={Zhang, Xiangwen and Su, Jinsong and Qin, Yue and Liu, Yang and Ji, Rongrong and Wang, Hongji},
  booktitle={Thirty-Second AAAI Conference on Artificial Intelligence},
  year={2018}
}

