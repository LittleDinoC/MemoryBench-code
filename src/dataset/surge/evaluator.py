import json
import os
import argparse
import time
import re
from . import markdownParser,rougeBleuFuncs,structureFuncs,informationFuncs

from sentence_transformers import CrossEncoder
from tqdm import tqdm
from FlagEmbedding import FlagModel

def normalize_string(s):
    letters = re.findall(r'[a-zA-Z]', s)
    return ''.join(letters).lower()
import numpy as np

def to_python_type(obj):
    if isinstance(obj, dict):
        return {k: to_python_type(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_python_type(v) for v in obj]
    elif isinstance(obj, (np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int16, np.int32, np.int64)):
        return int(obj)
    else:
        return obj


class SurGEvaluator:
    def __init__(self, corpus_path:str = None, flag_model_path:str = None, nli_model_path:str = None, openai_model = None):
        self.openai_model = openai_model

        
        self.corpus_dir = corpus_path

        
        
        corpus = []
        self.corpus_map = {}
        self.title2docid = {}
        with open(corpus_path,'r',encoding='utf-8') as f:
            corpus = json.load(f)
        for c in corpus:
            self.corpus_map[int(c['doc_id'])] = c.copy()
            self.title2docid[normalize_string(c['Title'])] = int(c['doc_id'])
            
        if flag_model_path == None :
            self.flag_model_path = 'BAAI/bge-large-en-v1.5'
        else:
            self.flag_model_path = flag_model_path
            
        # if judge_model_path == None :
        #     self.judge_model_path = None
        # else:
        #     self.judge_model_path = judge_model_path
            
        # self.judge_model = None
        self.flag_model = None
        # if self.judge_model_path != None:
        #     self.judge_model_tokenizer = AutoTokenizer.from_pretrained(self.judge_model_path)
        # else:
        #     self.judge_model_tokenizer = None    
            
        if nli_model_path == None:
            self.nli_model_path = 'cross-encoder/nli-deberta-v3-base'
        else:
            self.nli_model_path = nli_model_path
            
        self.nli_model = None
            
    def single_eval(self, survey_info, output, eval_list = ["ALL"]):
        psg_node = markdownParser.parse_markdown(lines=output.split('\n'))
        refs  = markdownParser.parse_refs(lines=output.split('\n'))
        refid2docid = {}
        for refid,ref_title in refs.items():

            if normalize_string(ref_title) in self.title2docid:
                ref_docid = self.title2docid[normalize_string(ref_title)]
                refid2docid[refid] = ref_docid
            else:
                refid2docid[refid] = ref_title
        # print(refid2docid)
        eval_result = {
            "Information_Collection": {
                "Comprehensiveness": {
                    "Coverage": None,
                },
                "Relevance": {
                    "Paper_Level": None,
                    "Section_Level": None,
                    "Sentence_Level": None,
                }
            },
            "Survey_Structure": {
                "Structure_Quality(LLM_as_judge)": None,
                "SH-Recall": None
            },
            "Survey_Content": {
                "Relevance": {
                        "ROUGE-1": None,
                        "ROUGE-2": None,
                        "ROUGE-L": None,
                        "BLEU": None,
                    },
                "Logic": None
            }
        }
        
        if "ROUGE-BLEU" in eval_list or "ALL" in eval_list:
            # print("Evaluating ROUGE and BLEU scores...")
            try:
                r1,r2,rl,bleu = rougeBleuFuncs.eval_rougeBleu(survey_info,psg_node)
            except:
                r1,r2,rl,bleu = 0,0,0,0
            eval_result["Survey_Content"]["Relevance"]["ROUGE-1"] = r1
            eval_result["Survey_Content"]["Relevance"]["ROUGE-2"] = r2
            eval_result["Survey_Content"]["Relevance"]["ROUGE-L"] = rl
            eval_result["Survey_Content"]["Relevance"]["BLEU"] = bleu
            # print(f"ROUGE-1: {r1}, ROUGE-2: {r2}, ROUGE-L: {rl}, BLEU: {bleu}")
        
        if "SH-Recall" in eval_list or "ALL" in eval_list:
            if self.flag_model == None:
                self.flag_model = FlagModel(self.flag_model_path, 
                    query_instruction_for_retrieval="Generate a representation for this title to calculate the similarity between titles:",
                        use_fp16=True)  
            try:
                sh_recall = structureFuncs.eval_SHRecall(survey_info,psg_node,self.flag_model)
            except Exception as e:
                print("Error in SH-Recall evaluation:", e)
                sh_recall = 0
            eval_result["Survey_Structure"]["SH-Recall"] = sh_recall
            # print(f"SH-Recall: {sh_recall}")
        
        if "Structure_Quality" in eval_list or "ALL" in eval_list:
            try:
                struct_quality = structureFuncs.eval_structure_quality_client(survey_info,psg_node,self.openai_model)
            except Exception as e:
                print("Error in Structure Quality evaluation:", e)
                struct_quality = 0
                # struct_quality = structureFuncs.eval_structure_quality(survey_info,psg_node,self.judge_model,self.judge_model_tokenizer)
            eval_result["Survey_Structure"]["Structure_Quality(LLM_as_judge)"] = struct_quality 
        
        if "Coverage" in eval_list or "ALL" in eval_list:
            try:
                coverage = informationFuncs.eval_coverage(survey_info['all_cites'],refid2docid)
            except Exception as e:
                print("Error in Coverage evaluation:", e)
            eval_result["Information_Collection"]["Comprehensiveness"]["Coverage"] = coverage
            
        if "Relevance-Paper" in eval_list or "ALL" in eval_list:
            if self.nli_model == None:
                self.nli_model = CrossEncoder(self.nli_model_path)
            refcontent = {}
            for k,v in refid2docid.items():
                sen_1 = None
                sen_paper = None 
                if isinstance(v,int):
                    tmp_1 = self.corpus_map[v]['Title']
                    tmp_2 = self.corpus_map[v]['Abstract']
                    tmp_title = survey_info['survey_title']
                    sen_1 = f"There is a paper. Title: '{tmp_1}'. Abstract: '{tmp_2}'"
                    sen_paper = f"The paper titled '{tmp_1}' with the given abstract could be cited in the paper: '{tmp_title}'."
                    refcontent[k] = (sen_1,sen_paper)
                else:
                    tmp_title = survey_info['survey_title']
                    # sen_1 = refcontent[k] = f"There is a paper. Title: '{v}'. The title '{v}' describes the content of the paper."
                    # sen_paper = f"The paper titled '{v}' could be cited in the paper: '{tmp_title}'."
                    sen_1 = "[NOTEXIST]"
                    sen_paper = "[NOTEXIST]"
                    refcontent[k] = (sen_1,sen_paper)
            paper_relevance = None
            if len(refid2docid) > 0:
                try:
                    paper_relevance = informationFuncs.eval_relevance_paper(survey_info,refid2docid,refcontent,self.nli_model)
                except Exception as e:
                    print("Error in Paper-Level Relevance evaluation:", e)
                    paper_relevance = 0
            else:
                paper_relevance = 0
            eval_result["Information_Collection"]["Relevance"]["Paper_Level"] = paper_relevance
        
        if ("Relevance-Section" in eval_list and "Relevance-Sentence" in eval_list) or "ALL" in eval_list:
            try:
                if self.nli_model == None:
                    self.nli_model = CrossEncoder(self.nli_model_path)
                extracted_cites = informationFuncs.extract_cites_with_subtitle_and_sentence(psg_node)
                nli_pairs_subtitle = []
                nli_pairs_sentence = []
                for ref_num,subtitle,sentence in extracted_cites:
                    if ref_num not in refid2docid:
                        docid = "This is an irrelevant paper."
                    else:
                        docid = refid2docid[ref_num]
                    sen_1 = None
                    sen_sentence = None
                    sen_section = None
                    if isinstance(docid,int):
                        tmp_1 = self.corpus_map[docid]['Title']
                        tmp_2 = self.corpus_map[docid]['Abstract']
                        sen_1 = f"There is a paper. Title: '{tmp_1}'. Abstract: {tmp_2}"
                        sen_section = f"The paper titled '{tmp_1}' with the given abstract is relevant to the section: '{subtitle}'."
                        sen_sentence = f"The paper titled '{tmp_1}' with the given abstract could be cited in the sentence: '{sentence}'."
                    else:
                        # The title
                        
                        # sen_1 = f"There is a paper. Title: '{docid}'. The title '{docid}' describes the content of the paper."
                        # sen_section = f"The paper titled '{docid}' is relevant to the section: '{subtitle}'."
                        # sen_sentence = f"The paper titled '{docid}' could be cited in the sentence: '{sentence}'."
                        sen_1 = "[NOTEXIST]"
                        sen_section = "[NOTEXIST]"
                        sen_sentence = "[NOTEXIST]"
                    nli_pairs_sentence.append((sen_1,sen_sentence))
                    nli_pairs_subtitle.append((sen_1,sen_section))
                section_relevance = None
                sentence_relevance = None
                if len(extracted_cites) > 0:    
                    section_relevance = informationFuncs.eval_relevance_section(nli_pairs_subtitle,self.nli_model)
                    sentence_relevance = informationFuncs.eval_relevance_sentence(nli_pairs_sentence,self.nli_model)
                else:
                    sentence_relevance = 0
                    section_relevance = 0
                eval_result["Information_Collection"]["Relevance"]["Section_Level"] = section_relevance
                eval_result["Information_Collection"]["Relevance"]["Sentence_Level"] = sentence_relevance
            except Exception as e:
                print("Error in Section/Sentence-Level Relevance evaluation:", e)
                eval_result["Information_Collection"]["Relevance"]["Section_Level"] = 0
                eval_result["Information_Collection"]["Relevance"]["Sentence_Level"] = 0
        elif "Relevance-Section" in eval_list:
            if self.nli_model == None:
                self.nli_model = CrossEncoder(self.nli_model_path)
            extracted_cites = informationFuncs.extract_cites_with_subtitle_and_sentence(psg_node)
            nli_pairs_subtitle = []
            for ref_num,subtitle,sentence in extracted_cites:
                if ref_num not in refid2docid:
                    docid = "This is an irrelevant paper."
                else:
                    docid = refid2docid[ref_num]
                sen_1 = None
                if isinstance(docid,int):
                    tmp_1 = self.corpus_map[docid]['Title']
                    tmp_2 = self.corpus_map[docid]['Abstract']
                    sen_1 = f"There is a paper. Title: '{tmp_1}'. Abstract: {tmp_2}"
                    sen_section = f"The paper titled '{tmp_1}' with the given abstract is relevant to the section: '{subtitle}'."
                else:
                    # The title
                    
                    # sen_1 = f"There is a paper. Title: '{docid}'. The title '{docid}' describes the content of the paper."
                    # sen_section = f"The paper titled '{docid}' is relevant to the section: '{subtitle}'." 
                    sen_1 = "[NOTEXIST]"
                    sen_section = "[NOTEXIST]"

                nli_pairs_subtitle.append((sen_1,sen_section))
            section_relevance = None
            if len(extracted_cites) > 0:    
                section_relevance = informationFuncs.eval_relevance_section(nli_pairs_subtitle,self.nli_model)
            else:
                section_relevance = 0
            
            eval_result["Information_Collection"]["Relevance"]["Section_Level"] = section_relevance
        elif "Relevance-Sentence" in eval_list:
            if self.nli_model == None:
                self.nli_model = CrossEncoder(self.nli_model_path)
            extracted_cites = informationFuncs.extract_cites_with_subtitle_and_sentence(psg_node)
            nli_pairs_sentence = []
            for ref_num,subtitle,sentence in extracted_cites:
                if ref_num not in refid2docid:
                    docid = "This is an irrelevant paper."
                else:
                    docid = refid2docid[ref_num]
                sen_1 = None
                if isinstance(docid,int):
                    tmp_1 = self.corpus_map[docid]['Title']
                    tmp_2 = self.corpus_map[docid]['Abstract']
                    sen_1 = f"There is a paper. Title: '{tmp_1}'. Abstract: {tmp_2}"
                    sen_sentence = f"The paper titled '{tmp_1}' with the given abstract could be cited in the sentence: '{sentence}'."
                else:
                    # The title
                    
                    # sen_1 = f"There is a paper. Title: '{docid}'. The title '{docid}' describes the content of the paper."
                    # sen_sentence = f"The paper titled '{docid}' could be cited in the sentence: '{sentence}'."
                    sen_1 = "[NOTEXIST]"
                    sen_sentence = "[NOTEXIST]"
                nli_pairs_sentence.append((sen_1,sen_sentence))
            
            sentence_relevance = None
            if len(extracted_cites) > 0:    
                sentence_relevance = informationFuncs.eval_relevance_sentence(nli_pairs_sentence,self.nli_model)
            else:
                sentence_relevance = 0
            sentence_relevance = informationFuncs.eval_relevance_sentence(nli_pairs_sentence,self.nli_model)
            eval_result["Information_Collection"]["Relevance"]["Sentence_Level"] = sentence_relevance
            
        if "Logic" in eval_list or "ALL" in eval_list:
            logic = informationFuncs.eval_logic_client(psg_node,self.openai_model)
            eval_result["Survey_Content"]["Logic"] = logic
            
        # flatten the result
        flatten_result = {}
        for k1,v1 in eval_result.items():
            for k2,v2 in v1.items():
                if isinstance(v2,dict):
                    for k3,v3 in v2.items():    
                        flatten_result[f"{k1}-{k2}-{k3}"] = v3
                else:
                    flatten_result[f"{k1}-{k2}"] = v2
                    
                    
        flatten_result = to_python_type(flatten_result)
        return flatten_result