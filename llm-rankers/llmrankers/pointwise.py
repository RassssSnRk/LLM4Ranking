from typing import List
from .rankers import LlmRanker, SearchResult
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoConfig
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from .pairwise import Text2TextGenerationDataset
import torch
from tqdm import tqdm
import tiktoken
import openai
import re
import time

class PointwiseLlmRanker(LlmRanker):

    def __init__(self, model_name_or_path, tokenizer_name_or_path, device, method="qlm", batch_size=1, cache_dir=None):
        self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_name_or_path
                                                     if tokenizer_name_or_path is not None else
                                                     model_name_or_path,
                                                     cache_dir=cache_dir)
        self.config = AutoConfig.from_pretrained(model_name_or_path, cache_dir=cache_dir)
        if self.config.model_type == 't5':
            self.llm = T5ForConditionalGeneration.from_pretrained(model_name_or_path,
                                                                  device_map='auto',
                                                                  torch_dtype=torch.float16 if device == 'cuda'
                                                                  else torch.float32,
                                                                  cache_dir=cache_dir)
        else:
            raise NotImplementedError(f"Model type {self.config.model_type} is not supported yet for pointwise :(")

        self.device = device
        self.method = method
        self.batch_size = batch_size

        self.total_compare = 0
        self.total_completion_tokens = 0
        self.total_prompt_tokens = 0

    def rerank(self, query: str, ranking: List[SearchResult]) -> List[SearchResult]:
        self.total_compare = 0
        self.total_completion_tokens = 0
        self.total_prompt_tokens = 0

        if self.method == "qlm":
            prompt = "Passage: {text}\nPlease write a question based on this passage."
            data = [prompt.format(text=doc.text) for doc in ranking]
            dataset = Text2TextGenerationDataset(data, self.tokenizer)
            loader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                collate_fn=DataCollatorWithPadding(
                    self.tokenizer,
                    max_length=512,
                    padding='longest',
                ),
                shuffle=False,
                drop_last=False,
                num_workers=4
            )

            labels = self.tokenizer.encode(f"<pad> {query}",
                                           return_tensors="pt",
                                           add_special_tokens=False).to(self.llm.device).repeat(self.batch_size, 1)
            current_id = 0
            with torch.no_grad():
                for batch_inputs in tqdm(loader):
                    self.total_compare += 1
                    self.total_prompt_tokens += batch_inputs['input_ids'].shape[0] * batch_inputs['input_ids'].shape[1]

                    batch_labels = labels if labels.shape[0] == len(batch_inputs['input_ids']) \
                        else labels[:len(batch_inputs['input_ids']), :]  # last batch might be smaller
                    self.total_prompt_tokens += batch_labels.shape[0] * batch_labels.shape[
                        1]  # we count decoder inputs as part of prompt.

                    batch_inputs = batch_inputs.to(self.llm.device)
                    logits = self.llm(input_ids=batch_inputs['input_ids'],
                                      attention_mask=batch_inputs['attention_mask'],
                                      labels=batch_labels).logits

                    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
                    scores = loss_fct(logits.view(-1, logits.size(-1)), batch_labels.view(-1))
                    scores = -1 * scores.view(-1, batch_labels.size(-1)).sum(dim=1)  # neg log prob
                    for score in scores:
                        ranking[current_id].score = score.item()
                        current_id += 1

        elif self.method == "yes_no":
            prompt = "Passage: {text}\nQuery: {query}\nDoes the passage answer the query? Answer 'Yes' or 'No'"
            yes_id = self.tokenizer.encode("Yes", add_special_tokens=False)[0]
            no_id = self.tokenizer.encode("No", add_special_tokens=False)[0]
            data = [prompt.format(text=doc.text, query=query) for doc in ranking]
            dataset = Text2TextGenerationDataset(data, self.tokenizer)
            loader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                collate_fn=DataCollatorWithPadding(
                    self.tokenizer,
                    max_length=512,
                    padding='longest',
                ),
                shuffle=False,
                drop_last=False,
                num_workers=4
            )
            decoder_input_ids = torch.Tensor([self.tokenizer.pad_token_id]).to(self.llm.device, dtype=torch.long).repeat(self.batch_size, 1)
            current_id = 0
            with torch.no_grad():
                for batch_inputs in tqdm(loader):
                    self.total_compare += 1
                    self.total_prompt_tokens += batch_inputs['input_ids'].shape[0] * batch_inputs['input_ids'].shape[1]

                    batch_inputs = batch_inputs.to(self.llm.device)

                    batch_decoder_input_ids = decoder_input_ids if decoder_input_ids.shape[0] == len(batch_inputs['input_ids']) \
                        else decoder_input_ids[:len(batch_inputs['input_ids']), :]  # last batch might be smaller

                    self.total_prompt_tokens += batch_decoder_input_ids.shape[0] * batch_decoder_input_ids.shape[
                        1]

                    logits = self.llm(input_ids=batch_inputs['input_ids'],
                                      attention_mask=batch_inputs['attention_mask'],
                                      decoder_input_ids=batch_decoder_input_ids).logits
                    yes_scores = logits[:, :, yes_id]
                    no_scores = logits[:, :, no_id]
                    batch_scores = torch.cat((yes_scores, no_scores), dim=1)
                    batch_scores = torch.nn.functional.softmax(batch_scores, dim=1)
                    scores = batch_scores[:, 0]
                    for score in scores:
                        ranking[current_id].score = score.item()
                        current_id += 1

        ranking = sorted(ranking, key=lambda x: x.score, reverse=True)
        return ranking

    def truncate(self, text, length):
        return self.tokenizer.convert_tokens_to_string(self.tokenizer.tokenize(text)[:length])


class MonoT5LlmRanker(PointwiseLlmRanker):
    def rerank(self, query: str, ranking: List[SearchResult]) -> List[SearchResult]:
        self.total_compare = 0
        self.total_completion_tokens = 0
        self.total_prompt_tokens = 0
        prompt = "Query: {query} Document: {document} Relevant:"
        data = [prompt.format(query=query, document=doc.text) for doc in ranking]
        dataset = Text2TextGenerationDataset(data, self.tokenizer)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            collate_fn=DataCollatorWithPadding(
                self.tokenizer,
                max_length=512,
                padding='longest',
            ),
            shuffle=False,
            drop_last=False,
            num_workers=4
        )
        decoder_input_ids = torch.Tensor([self.llm.config.decoder_start_token_id]).to(self.llm.device, dtype=torch.long).repeat(
            self.batch_size, 1)
        current_id = 0
        with torch.no_grad():
            for batch_inputs in tqdm(loader):
                self.total_compare += 1
                self.total_prompt_tokens += batch_inputs['input_ids'].shape[0] * batch_inputs['input_ids'].shape[1]

                batch_inputs = batch_inputs.to(self.llm.device)

                batch_decoder_input_ids = decoder_input_ids if decoder_input_ids.shape[0] == len(
                    batch_inputs['input_ids']) \
                    else decoder_input_ids[:len(batch_inputs['input_ids']), :]  # last batch might be smaller

                self.total_prompt_tokens += batch_decoder_input_ids.shape[0] * batch_decoder_input_ids.shape[
                    1]

                logits = self.llm(input_ids=batch_inputs['input_ids'],
                                  attention_mask=batch_inputs['attention_mask'],
                                  decoder_input_ids=batch_decoder_input_ids).logits

                # 6136 and 1176 are the indexes of the tokens false and true in T5.
                batch_scores = logits[:, 0, [6136, 1176]]
                batch_scores = torch.nn.functional.softmax(batch_scores, dim=1)
                scores = batch_scores[:, 1]
                for score in scores:
                    ranking[current_id].score = score.item()
                    current_id += 1

        ranking = sorted(ranking, key=lambda x: x.score, reverse=True)
        return ranking

class OpenAiPointwiseLlmRanker(LlmRanker):
    def __init__(self, model_name_or_path, api_key):
        self.llm = model_name_or_path
        self.tokenizer = tiktoken.encoding_for_model(model_name_or_path)
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.system_prompt = "You are RankGPT, an intelligent assistant specialized in selecting the most relevant passage from a pair of passages based on their relevance to the query."
        self.prompt = """Given a query {query}, how relevant is the passage to the query on a scale of 0 to 20?
        
Passage: "{doc}"

Output the score :"""
        openai.api_key = api_key
        
        self.total_compare = 0
        self.total_completion_tokens = 0
        self.total_prompt_tokens = 0

    def _get_response(self, input_text):
        while True:
            try:
                # response = openai.ChatCompletion.create(
                response = openai.chat.completions.create(
                    model=self.llm,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": input_text},
                    ],
                    temperature=0.0,
                    # request_timeout=15
                    timeout=15
                )
                # self.total_completion_tokens += int(response['usage']['completion_tokens'])
                # self.total_prompt_tokens += int(response['usage']['prompt_tokens'])
                self.total_completion_tokens += int(response.usage.completion_tokens)
                self.total_prompt_tokens += int(response.usage.prompt_tokens)

                # output = response['choices'][0]['message']['content']
                output = response.choices[0].message.content

                # matches = re.findall(r"(Passage [A-B])", output, re.MULTILINE)
                # if matches:
                #     output = matches[0][8]
                # elif output.strip() in self.CHARACTERS:
                #     pass
                # else:
                #     print(f"Unexpected output: {output}")
                #     output = "A"
                return output

            except Exception as e:
                print(str(e))
                time.sleep(0.1)
    def rerank(self, query: str, ranking: List[SearchResult]) -> List[SearchResult]:
        print(f"pointwise openai reranker, query = {query}")
        results = []
        # tmp_break = 10
        # cnt = 0
        for sr in tqdm(ranking):
            doc = sr.text
            input = self.prompt.format(query=query, doc=doc)
            # print(f'input = {input}')
            res = self._get_response(input)
            # print(f'response = {res}')
            try : 
                score = int(re.search(r'\d+', res).group())
            except : 
                score = 0
            # print(f'got score = {score}')
            results.append((int(score), SearchResult(docid=sr.docid, score=score, text=None)))
            # print(cnt, end=" ")
            # cnt+=1
            # if len(results) == tmp_break:
            #     break
        # print(results)
        print("\nsorting...")
        results = sorted(results, key=lambda x: -x[0])
        results = [sr for score, sr in results]
        print(results)
        return results


    def compare(self, query: str, docs: List):
        self.total_compare += 1
        doc1, doc2 = docs[0], docs[1]
        input_texts = [self.prompt.format(query=query, doc1=doc1, doc2=doc2),
                       self.prompt.format(query=query, doc1=doc2, doc2=doc1)]

        return [f'Passage {self._get_response(input_texts[0])}', f'Passage {self._get_response(input_texts[1])}']

    def truncate(self, text, length):
        return self.tokenizer.decode(self.tokenizer.encode(text)[:length])


