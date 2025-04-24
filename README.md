# LLM4Ranking
NTHU CS 大四專題
### dependency
- [pyserini](https://github.com/castorini/pyserini)
```c++=
conda create --name <name> python=3.10
conda install -c conda-forge openjdk=21 maven -y
pip install torch faiss-cpu
pip install pyserini
```
pyserini depends on Java. More installation guides are [here](https://github.com/castorini/pyserini/blob/master/docs/installation.md)
- Rankgpt
```bash=
git clone https://github.com/sunnweiwei/RankGPT
```
- llmRankers
```bash=
git clone https://github.com/ielab/llm-rankers
```

### First stage retrieval (BM25)
```bash=
python -m pyserini.search.lucene \
    --threads 16 --batch-size 128 \
    --index msmarco-v1-passage \
    --topics dl19-passage \
    --output run.msmarco-v1-passage.bm25-default.dl19.txt \
    --bm25 --k1 0.9 --b 0.4
```
### pointwise approach with gpt
```bash=
python3 run.py \
   run --model_name_or_path gpt-4o-mini \
       --run_path run.msmarco-v1-passage.bm25-default.dl19.txt \
       --save_path ./run.pointwise.gpt-4o-mini.dl19.txt \
       --ir_dataset_name msmarco-passage/trec-dl-2019 \
       --openai_key <openai_key> \
  pointwise
```
> modified pointwise re-ranking method is at  
> ```llm-rankers/llmrankers/pointwise.py``` class : OpenAiPointwiseLlmRanker
### listwise approach with gpt
```bash=
python3 run.py \
  run --model_name_or_path gpt-4o-mini-2024-07-18 \
      --run_path run.msmarco-v1-passage.bm25-default.dl19.txt \
      --save_path ./output/run.liswise.gpt-4o-mini.txt \
      --ir_dataset_name msmarco-passage/trec-dl-2019 \
      --openai_key <openai_key> \
  listwise --window_size 4 \
           --step_size 2 \
           --num_repeat 5
```
### setwise heapsort approach with gpt
```bash=
python3 run.py \
  run --model_name_or_path gpt-4o-mini-2024-07-18 \
      --run_path run.msmarco-v1-passage.bm25-default.dl19.txt \
      --save_path ./run.setwise.heapsort.gpt-4o-mini.dl19.txt \
      --ir_dataset_name msmarco-passage/trec-dl-2019 \
      --openai_key <openai_key> \
      --hits 100 \
  setwise --num_child 2 \
          --method heapsort \
          --k 10
```

### evaluation
```
# usage
python -m pyserini.eval.trec_eval -c -l 2 -m <metric> <dataset> <output_file>
# example
python -m pyserini.eval.trec_eval -c -l 2 -m ndcg_cut.10 dl19-passage \
    llm-rankers/output/run.pointwise.gpt-4o-mini.dl19.txt

Results:
ndcg_cut_10             all     0.6798
```
> results are in located at ```llm-rankers/output``` and ```RankGPT/output```


## Reference
RankGPT [github](https://github.com/sunnweiwei/RankGPT)  
llm-rankers from [ielab](https://github.com/ielab/llm-rankers)  
alpha nDCG from [Novelty and diversity in information retrieval evaluation](http://plg.uwaterloo.ca/~gvcormac/novelty.pdf) 