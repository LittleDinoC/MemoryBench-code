'''
该数据集已弃用，相关代码仅供参考
'''


# from src.dataset.base import BaseDataset
# from src.dataset.surge.evaluator import SurGEvaluator
# from typing import List, Dict, Any, Type, Tuple
# from pydantic import Field, BaseModel
# import re
# import jsonlines
# import json
# import os

# from src.llms import LlmFactory
# from pydantic import BaseModel, Field


# class BaseAgentConfig(BaseModel):
#     llm_provider: str = Field(
#         default="openai", 
#         description="The LLM provider to use for the agent."
#     )
#     llm_config: dict = Field(
#         default_factory=dict, 
#         description="Configuration parameters for the LLM."
#     )
    
# prompt_template = """**Task:**
# You are an expert academic writer tasked with composing a comprehensive survey paper on [TOPIC]. Based on the provided research papers and their abstracts, write a unified and cohesive survey that provides an in-depth overview of the current state of research. **Note that some papers may not be related to the topic; please ignore the unrelated ones.**

# **Requirements:**

# * **Length:** Approximately 5,000 words.
# * **Tone and Style:** Formal academic writing suitable for publication in a scholarly journal.
# * **Content:**
#   * Provide a holistic analysis that synthesizes insights from all papers.
#   * Identify overarching themes, patterns, and trends.
#   * Compare and contrast different methodologies and approaches.
#   * Highlight significant advancements, debates, and unique contributions.
#   * Discuss challenges, open problems, and future research directions.
# * **Structure:**
#   * **Abstract:** Concise summary of the survey (200–250 words).
#   * **Introduction:** Introduce the topic, its significance, and survey objectives.
#   * **Main Sections:** Organize thematically or methodologically, integrating multiple papers in each section.
#   * **Conclusion:** Summarize key findings and implications for future research.
#   * **References:** Provide a complete list of cited works. Use one of the following formats:
#     (1) A list of references with the format:
#         [num] Author(s). Title. (The title may be embraced by '*')
#     (2) A list of references with the format:
#         [num] Title
# * **Citations:** Use in-text citations (e.g., \[Author, Year]) when referring to specific papers.
# * **Coherence:** Ensure smooth transitions, consistent terminology, and logical flow.
# * **Originality:** Paraphrase and synthesize ideas; do not copy abstracts verbatim.

# **Example Start:**
# "### Abstract:
# This survey provides a comprehensive overview of ..., synthesizing findings from a wide range of influential papers published in recent years. It highlights key advancements, methodologies, and debates while offering insights into emerging challenges and future research directions.

# ### Introduction:
# The rapid evolution of ... has significantly shaped .... This survey consolidates knowledge across diverse studies to provide researchers with a coherent understanding of the field’s current landscape..."

# Now, begin writing the survey paper about [TOPIC] based on the provided papers and abstracts, using Markdown format for headings and structure."""
    
# class SurGE_Dataset(BaseDataset):

#     def __init__(self, data_path: str, dataset_name: str = "SurGE", test_metrics: List[str] = [], max_output_len: int = 8192, flag_model_path:str = None, nli_model_path:str = None):
#         self.dataset_name = dataset_name
#         # self.feedback_type = feedback_type
        
        
#         config = BaseAgentConfig(
#             llm_config = {
#                 "model": "gpt-4o-2024-05-13",
#                 "temperature": 0.8,
#                 "max_tokens": 100,
#             }
#         )
#         self.openai_model = LlmFactory.create(
#             provider_name=config.llm_provider,
#             config=config.llm_config,
#         )
#         self.evaluator = SurGEvaluator(os.path.join(data_path, 'corpus.json'), openai_model=self.openai_model, flag_model_path=flag_model_path, nli_model_path=nli_model_path)
#         super().__init__(data_path=data_path, test_metrics=test_metrics, max_output_len=max_output_len)
        
        
        

#     def _load_data(self) -> Dict[str, List[Dict[str, Any]]]:
#         raw_data = []
#         with open(os.path.join(self.data_path, 'surveys.json'), 'r', encoding='utf-8') as f:
#             surveys = json.load(f)
            
        
            
#         for s in surveys:
#             all_cites = ""
            
#             for cite in s["all_cites"]:
#                 doc = self.evaluator.corpus_map[cite]
#                 all_cites += f"""**Title:** {doc['Title']}
# **Authors:** {', '.join(doc['Authors'])}
# **Abstract:** {doc['Abstract']}

# """
#             raw_data.append({
#                 "test_idx": len(raw_data),
#                 "corpus": all_cites,
#                 "input_prompt": prompt_template.replace("[TOPIC]", s["survey_title"]),
#                 "dataset_name": self.dataset_name,
#                 # "feedback_type": self.feedback_type,
#                 "lang": "en",
#                 "info": {
#                     k: v for k, v in s.items()
#                 }
#             })
#         return raw_data

#     def evaluate_single(self, user_prompt: str, info: Dict[str, Any], llm_response: str) -> Dict[str, float]:
#         return self.evaluator.single_eval(info, llm_response)
    
    
# if __name__ == "__main__":
#     # Example usage
#     dataset = SurGE_Dataset(data_path="./raw/SurGE")
    
#     item = dataset.dataset[0]
    
#     print(">>>>> JuDGE Dataset Length:", len(dataset))
    
#     print(">>>>> Item:")
    
#     print(json.dumps(item, ensure_ascii=False, indent=2))
    
#     print(">>>>> LLM Response:")
    
#     response = dataset.openai_model.generate_response([
#         {"role": "system", "content": "You are a helpful assistant"},
#         {"role": "user", "content": item["corpus"] + "\n\n" + item["input_prompt"]},
#     ])
    
#     print(response)
    
#     print(">>>>> Evaluation Score:")
    
#     score = dataset.evaluate_single(
#         user_prompt=item["input_prompt"],
#         info=item["info"],
#         llm_response=response
#     )
    
#     print(json.dumps(score, ensure_ascii=False, indent=2))
    
#     print(list(score.keys()))
    