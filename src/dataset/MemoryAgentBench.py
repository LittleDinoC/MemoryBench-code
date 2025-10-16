'''
该数据集已弃用，相关代码仅供参考
'''

# import json
# import pandas as pd
# import os    
# from typing import List, Dict, Any
# from src.dataset.base import BaseDataset
# import re
# from editdistance import eval as edit_distance
# import numpy as np

# def clean_text_elements(text, remove_parentheses=True, normalize_ws=True, remove_nums=True):
#     """Clean text by removing various elements."""
#     if remove_parentheses:
#         text = re.sub(r"\([^()]*\)", "", text)
#     if remove_nums:
#         text = re.sub(r"^(?:\d+[\.\)、]?\s*[\-\—\–]?\s*)?", "", text)
#     if normalize_ws:
#         text = re.sub(r"\s+", " ", text).strip()
#     return text


# def clean_parentheses(text):
#     """Remove content within parentheses from text."""
#     return re.sub(r"\([^()]*\)", "", text)


# def normalize_whitespace(text):
#     """Normalize whitespace in text."""
#     return re.sub(r"\s+", " ", text).strip()


# def remove_numbering(text):
#     """Remove numbering from the beginning of text."""
#     return re.sub(r"^(?:\d+[\.\)、]?\s*[\-\—\–]?\s*)?", "", text)


# def extract_movie_name(text):
#     """
#     Extract and clean movie name from file path or text.
    
#     Args:
#         text: Raw text containing movie name
        
#     Returns:
#         Cleaned movie name
#     """
#     # Extract filename if it's a path
#     filename = text.split('/')[-1]
#     # Replace common separators with spaces
#     cleaned_name = filename.replace('_', ' ').replace('-', ' ').replace('>', ' ')
#     # Apply cleaning functions
#     return normalize_whitespace(clean_parentheses(cleaned_name))


# def find_nearest_movie(target_name, candidate_movies):
#     """
#     Find the nearest movie name using edit distance.
    
#     Args:
#         target_name: The movie name to match
#         candidate_movies: List of candidate movie names
        
#     Returns:
#         Dictionary with matching information
#     """
#     # Remove duplicates from candidates
#     unique_candidates = list(set(candidate_movies))
    
#     # Calculate edit distances
#     distances = [edit_distance(target_name.lower(), candidate.lower()) 
#                 for candidate in unique_candidates]
    
#     # Find nearest match
#     nearest_index = np.argmin(distances)
#     nearest_movie = unique_candidates[nearest_index]
    
#     return {
#         'movie_name': target_name, 
#         'min_edit_distance': distances[nearest_index], 
#         'nearest_movie': nearest_movie
#     }


# def extract_recommendation_list(text, movie_candidates=None):
#     """
#     Extract recommendation list from text output.
    
#     Args:
#         text: Text containing recommendations
#         movie_candidates: Optional list of valid movie names for matching
        
#     Returns:
#         Tuple of (recommendation_list, preference_text)
#     """
#     try:
#         # Try to split on first numbered item
#         preference_text, recommendation_text = text.split('1.', maxsplit=1)
#     except Exception as e:
#         print(e)
#         preference_text = ""
#         # Fallback: replace commas with newlines for parsing
#         recommendation_text = text.replace(',', '\n')
#     # Extract and clean recommendation items using the consolidated function
#     raw_recommendations = [
#         clean_text_elements(item.strip()) for item in recommendation_text.split('\n')
#     ]
    
#     # Match against candidates if provided
#     recommendation_list = ([find_nearest_movie(item, movie_candidates) for item in raw_recommendations] 
#                          if movie_candidates is not None else raw_recommendations)
    
#     return recommendation_list, preference_text


# def process_recsys_dataset(prediction, answer, id_to_name):
#     """Process recommendation system dataset outputs."""
#     # Load movie entity mapping


#     # Get movie candidates and parse prediction
#     movie_candidates = list(id_to_name.values())
    
#     predicted_list, _ = extract_recommendation_list(prediction, movie_candidates)
#     predicted_movies = [item['nearest_movie'] for item in predicted_list]

#     # Convert ground truth IDs to movie names / answer is a string with movie ids divided by comma
#     ground_truth_ids = [int(movie_id.strip()) for movie_id in answer]
#     ground_truth_movies = [id_to_name[movie_id] for movie_id in ground_truth_ids]

#     # Calculate recall at different cutoffs
#     recall_at_1 = sum([movie in predicted_movies[:1] for movie in ground_truth_movies]) / len(ground_truth_movies)
#     recall_at_5 = sum([movie in predicted_movies[:5] for movie in ground_truth_movies]) / len(ground_truth_movies)
#     recall_at_10 = sum([movie in predicted_movies[:10] for movie in ground_truth_movies]) / len(ground_truth_movies)
    
#     result = {
#         "recsys_recall@1": recall_at_1,
#         "recsys_recall@5": recall_at_5,
#         "recsys_recall@10": recall_at_10,
#         "parsed_output": predicted_movies,
#         "gt_movies": ground_truth_movies
#     }
    
#     return result

# class MemoryAgentBench_Dataset(BaseDataset):

#     def __init__(self, data_path: str, dataset_name: str = "MemoryAgentBench-Movie", test_metrics: List[str] = ["recsys_recall@5"], max_output_len: int = 8192):
#         self.dataset_name = dataset_name
#         # self.feedback_type = feedback_type
#         super().__init__(data_path=data_path, test_metrics=test_metrics, max_output_len=max_output_len)
#         name_to_id = json.load(open(os.path.join(self.data_path, "entity2id.json")))
#         self.id_to_name = {entity_id: extract_movie_name(name) for name, entity_id in name_to_id.items()}
     
     
#     def _load_data(self) -> Dict[str, List[Dict[str, Any]]]:
#         raw_data = []
        
        
#         df = pd.read_parquet(os.path.join(self.data_path, "Test_Time_Learning-00000-of-00001.parquet"))
#         # for index, obj in df.iterrows():
#         # 只读第一行
#         first_row = df.iloc[0]
#         questions = first_row['questions']
#         answers = first_row['answers']
#         assert len(questions) == len(answers), "Questions and answers must have the same length"
#         for i in range(len(questions)):
#             question = questions[i]
#             answer = answers[i].tolist()
#             raw_data.append({
#                 "test_idx": len(raw_data),
#                 "input_prompt": f"Pretend you are a movie recommender system. You need to recommend movies based on the example dialogues you have retrieved. Now I will give you a new conversation between a user and you (a recommender system). Based on the conversation, you reply me with 20 recommendations without extra sentences. \n For Example:\n [Conversation]\n The recommendations are: \n 1.movie1 \n 2.movie2 \n ...\n Here is the conversation: {question} \n The recommendations are:",
#                 "dataset_name": self.dataset_name,
#                 "corpus": first_row['context'],
#                 # "feedback_type": self.feedback_type,
#                 "lang": "en",
#                 "info": {
#                     'golden_answer': answer
#                 }
#             })
#         return raw_data

#     def evaluate_single(self, user_prompt: str, info: Dict[str, Any], llm_response: str) -> Dict[str, float]:
#         golden_answer = info['golden_answer']
#         return process_recsys_dataset(llm_response, golden_answer, self.id_to_name)

    
# if __name__ == "__main__":
#     # Example usage 
#     dataset = MemoryAgentBench_Dataset("./raw/MemoryAgentBench")
#     item = dataset.dataset[9]
    
#     print(">>>>>> MemoryAgentBench Dataset Length:")
#     print(len(dataset))
    
#     print("=" * 50)
    
#     print(">>>>>> MemoryAgentBench Dataset Item:")
#     print(json.dumps(item, ensure_ascii=False, indent=2))
    
#     print("=" * 50)
    
#     score = dataset.evaluate([{
#             "test_idx": 9,
#             "response": """1. The Shawshank Redemption
# 2. The Godfather
# 3. The Dark Knight
# 4. Pulp Fiction
# 5. Forrest Gump
# 6. Inception"""
#     }])
#     print(">>>>> Evaluation Score:")
#     print(json.dumps(score, ensure_ascii=False, indent=2))
