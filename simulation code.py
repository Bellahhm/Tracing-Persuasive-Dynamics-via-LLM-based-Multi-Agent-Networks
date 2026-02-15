import re
import json
import random
import requests
import matplotlib.pyplot as plt
import networkx as nx
from datetime import datetime
from typing import List, Dict, Any
from collections import defaultdict
from matplotlib.colors import LinearSegmentedColormap
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import concurrent.futures

import sys

class Logger(object):
    def __init__(self, filename, stream):
        self.terminal = stream
        self.log = open(filename, "w", encoding="utf-8")
        self.buffer = ""

    def write(self, message):
        self.terminal.write(message)
        self.buffer += message
        if "\n" in message:
            self.log.write(self.buffer)
            self.log.flush() 
            self.buffer = ""

    def flush(self):
        self.terminal.flush()
        if self.buffer:
            self.log.write(self.buffer)
            self.log.flush()
            self.buffer = ""


sys.stdout = Logger("program_prints.txt", sys.stdout)



# API key
API_KEYS = [

]

class Agent:
    def __init__(self, agent_id: int, attributes: Dict, initial_connections: List[int]):
        self.id = agent_id
        if isinstance(attributes.get('personality'), dict):
            self.attributes = attributes
        else:
            attributes['personality'] = {
                "openness": round(random.uniform(0, 1), 2),
                "conscientiousness": round(random.uniform(0, 1), 2),
                "extraversion": round(random.uniform(0, 1), 2),
                "agreeableness": round(random.uniform(0, 1), 2),
                "neuroticism": round(random.uniform(0, 1), 2)
            }
            self.attributes = attributes

        self.connections = initial_connections

        self.memory = {
            "short_term": defaultdict(list),
            "deep_reflections": {}  
        }
        self.memory.setdefault("stance_value_history", [])
        self.memory.setdefault("emotion_history", [])
        self.memory.setdefault("immediate_reflections", {})
        self.memory.setdefault("connections_history", {})

        self.emotion = 0.0  
        self.persuasion_stats = {"persuasion_count": 0, "dialogue_count": 0, "records": []}
        self.topic_engagement = defaultdict(float)
        self._sim = None  

        self.save_profile()
        print(f"[Agent {self.id} Profile]: {json.dumps(self.attributes, ensure_ascii=False, indent=2)}")

    def get_profile_str(self) -> str:
        nationality = self.attributes.get('nationality', 'Unknown')
        age = self.attributes.get('age', 'Unknown')
        gender = self.attributes.get('gender', 'Unknown')
        education = self.attributes.get('education', 'Unknown')
        occupation = self.attributes.get('occupation', 'Unknown')
        personality = self.attributes.get('personality', {})
        personality_str = ", ".join([f"{k}: {v}" for k, v in personality.items()]) if personality else "None"
        return (f"[Profile] Nationality: {nationality}, Age: {age}, Gender: {gender}, "
                f"Education: {education}, Occupation: {occupation}, Personality: {{{personality_str}}}")

    def save_profile(self):
        profile_data = {
            "attributes": self.attributes,
            "event_cognition": self.memory.get("event_cognition", {})
        }
        with open(f"agent_{self.id}_profile.json", "w", encoding="utf-8") as f:
            json.dump(profile_data, f, ensure_ascii=False, indent=2)

    def _save_to_file(self, file_type: str, record: Dict):
        try:
            filename = f"agent_{self.id}_{file_type}.jsonl"
            with open(filename, "a", encoding="utf-8") as file:
                json_record = {
                    "timestamp": datetime.now().isoformat(),
                    "data": record
                }
                file.write(json.dumps(json_record, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"Failed to save {file_type} record: {str(e)}")

    def save_memory(self, round_num: int):
        try:
            short_term_data = {
                "round": round_num,
                "memory": self.memory['short_term'].get(round_num, [])
            }
            self._save_to_file("short_term", short_term_data)
            deep_ref_data = {
                "round": round_num,
                "deep_reflections": self.memory["deep_reflections"],
                "stance_value_history": self.memory.get("stance_value_history", []),
                "emotion_history": self.memory.get("emotion_history", [])
            }
            self._save_to_file("deep_reflection", deep_ref_data)
            network_data = {
                "round": round_num,
                "connections": self.connections,
                "history": self.memory.get("connections_history", {})
            }
            self._save_to_file("network", network_data)
            persuasion_data = {
                "round": round_num,
                "records": self.persuasion_stats.get("records", []),
                "stats": self.persuasion_stats,
                "rate": self.get_persuasion_rate()
            }
            self._save_to_file("persuasion", persuasion_data)
        except Exception as e:
            print(f"Agent {self.id} failed to save memory: {str(e)}")

    def save_immediate_reflection(self, discussion_id: str, trigger: Dict, reflection: Dict):
        record = {
            "discussion_id": discussion_id,
            "trigger": trigger,
            "reflection": reflection
        }
        self._save_to_file("immediate_reflection", record)

    def update_network_connections(self, discussion_id: str, change_data: Dict):
        self.connections = list(set(self.connections + change_data.get("add", [])))
        self.connections = [c for c in self.connections if c not in change_data.get("remove", [])]
        self.memory.setdefault("connections_history", {})[discussion_id] = {
            "operation": change_data,
            "current_connections": self.connections
        }
        self._save_to_file("network", self.memory["connections_history"][discussion_id])

    def update_persuasion_stats(self, discussion_id: str, persuaded: bool):
        if persuaded:
            self.persuasion_stats["persuasion_count"] += 1
        self.persuasion_stats["dialogue_count"] += 1

        self.persuasion_stats.setdefault("records", []).append({
            "discussion_id": discussion_id,
            "persuasion_count": self.persuasion_stats["persuasion_count"],
            "dialogue_count": self.persuasion_stats["dialogue_count"],
            "rate": self.get_persuasion_rate()
        })
        persuasion_data = {
            "discussion_id": discussion_id,
            "stats": self.persuasion_stats,
            "rate": self.get_persuasion_rate()
        }
        self._save_to_file("persuasion", persuasion_data)

    def save_conversation(self, round_num: int, records: List[Dict]):
        processed = []
        for r in records:
            processed.append({
                "agent": self.id,
                "age": self.attributes.get('age', 'Unknown'),
                "gender": self.attributes.get('gender', 'Unknown'),
                "timestamp": datetime.now().isoformat(),
                "type": r.get("type", "utterance"),
                "content": str(r.get("content", "")),
                "related_agents": list(set(r.get("related_agents", []))),
                "round": round_num
            })
        self.memory['short_term'][round_num] = processed

    def get_persuasion_rate(self) -> float:
        if self.persuasion_stats["dialogue_count"] == 0:
            return 0.0
        return self.persuasion_stats["persuasion_count"] / self.persuasion_stats["dialogue_count"]


    def generate_response(self, context: List[Dict], is_initiator: bool = False) -> str:

        current_deep = self.memory["deep_reflections"].get(str(self._sim.round), {})
        keywords_summary = ", ".join(current_deep.get("keywords", [])) if current_deep.get("keywords") else "None"
        latest_emotion = current_deep.get("emotion_analysis", {}).get("current_emotion", self.emotion)
        recent_dialogues = context

        immediate_refs = self.memory.get("immediate_reflections", {})
        event_cognition = self.memory.get("event_cognition", {})
        prompt = f"""
[Deep Reflection Memory Summary]
Keywords: {keywords_summary}
Latest Emotion: {latest_emotion}

[Event Cognition]
{json.dumps(event_cognition, ensure_ascii=False, indent=2)}

[Short-Term Memory: Entire Discussion]
{json.dumps(recent_dialogues, ensure_ascii=False, indent=2)}

[Immediate Reflections]
{json.dumps(immediate_refs, ensure_ascii=False, indent=2)}

[Current State]
Stance Value: {self.attributes.get('stance_value', 0.0):.2f}
Emotion: {self.emotion:.2f}

Please generate a natural conversational response that reflects your personality and your current understanding of the event described in the event cognition. Do not include extra actions or settings.
"""
        response = self._call_llm(prompt)
        self._analyze_response(response)
        return response if response else "[No response]"


    def immediate_reflection(self, speaker_id: int, statement: str, discussion_id: str) -> Dict:
        try:
            other = self._sim.agents[speaker_id]
            other_nationality = other.attributes.get('nationality', 'Unknown')
            other_age = other.attributes.get('age', 'Unknown')
            other_gender = other.attributes.get('gender', 'Unknown')
            speaker_info = f"{other_gender} {other_age} years old, from {other_nationality}"
        except KeyError:
            speaker_info = "Unknown"
        event_data = self.memory.get("event_cognition", {})
        event_basic = event_data.get("basic_cognition", "No event cognition available")
        keywords_list = event_data.get("keywords", [])
        event_keywords = ", ".join(keywords_list) if isinstance(keywords_list, list) else "None"
        stance_val = round(self.attributes.get('stance_value', 0.0), 2)
        prompt = f"""
[Current Stance] {stance_val}
[Event Background] {event_basic}; Keywords: {event_keywords}

Please read the following utterance:
Utterance by Agent {speaker_id} ({speaker_info})
Content: "{statement}"

Note: In your answer, do not consider yourself as the source of persuasion; if you are persuaded, only list the IDs of other agents.

Please answer in order:
1. How logical is this utterance? (High/Medium/Low)
2. What is the emotional tendency and value (0~1)?
3. How aligned is it with your stance? (High/Medium/Low)
4. Is a rebuttal necessary? (Yes/No)
5. Are you willing to continue the discussion? (Yes/No)
6. Have you been persuaded by the other agent’s viewpoint? (Yes/No)
7. If persuaded, list the IDs of all agents that persuaded you (separated by commas); if not, explain why and propose further questions.
8. Provide an overall explanation of your reflection and state your emotion value (0~1).

Please reply in the following format:
Logical Consistency: ...
Emotional Tendency: ...
Stance Alignment: ...
Need to Refute: ...
Continue Discussion: ...
Persuaded: ...
Persuasion Sources: ...
Further Questions: ...
Overall Reflection: ...
Emotion Value: ...
"""
        response = self._call_llm(prompt)
        print(f"[Agent {self.id} immediate_reflection] {response}")
        reflection = self._parse_reflection(response)
        self.memory.setdefault("immediate_reflections", {}).setdefault(discussion_id, {})[self.id] = reflection
        trigger_info = {
            "speaker_id": speaker_id,
            "utterance": statement
        }
        self.save_immediate_reflection(discussion_id, trigger_info, reflection)
        return reflection

    def deep_reflection(self, round_num: int, discussion_id: str = None) -> Dict:
        round_dialogues = self.memory['short_term'].get(round_num, [])
        stance_val = round(self.attributes.get('stance_value', 0.0), 2)
        emotion_val = round(self.emotion, 2)
        prompt = f"""
[Current State] Stance: {stance_val}; Emotion: {emotion_val}
[Dialogue for this Discussion] {self._format_dialogues(round_dialogues)}

Note: In your response, do not consider yourself as the source of persuasion; if persuaded, only list the IDs of other agents.

Please conduct a deep reflection on this discussion by answering the following:
1. List at least three main viewpoints you agree with and their sources.
2. List at least two viewpoints you disagree with and your reasons.
3. Should you adjust your stance? If yes, provide the new stance value.
4. Extract up to 10 keywords from this discussion and evaluate the emotion value (0~1) for each, and indicate which agents influenced them.
5. Summarize your overall reflection on this discussion and your emotional state.
6. Have you been persuaded by other viewpoints? If yes, list the persuaded viewpoints and the IDs of the agents who persuaded you (separated by commas); if not, explain why and propose further questions.
7. Should you adjust your social network (only for agents involved in or mentioned in the dialogue) due to the opinions/stances/speaking styles/interests etc(you could be more sensitive)? Explain the specific adjustment plan and reasons.
8. Provide the adjusted emotion value (0~1).

Please strictly return in JSON format, for example:
{{
    "consensus": ["Viewpoint1", "Viewpoint2", ...],
    "accepted": {{"AgentID": "Viewpoint content", ...}},
    "rejected": {{"AgentID": "Rejection reason", ...}},
    "stance_change": "Yes/No",
    "new_stance_value": 0.xx,
    "keywords": ["Keyword1", "Keyword2", ...],
    "keywords_opinion": {{"Keyword1": {{"sentiment": 0.xx, "influenced_by": [AgentID1]}}, ...}},
    "network_changes": {{
        "add": [ID1, ID2],
        "remove": [ID3, ID4],
        "reason": "Reason for adjustment"
    }},
    "supporting_agents": [AgentID1, AgentID2],
    "opposing_agents": [AgentID3],
    "emotion_analysis": {{
        "current_emotion": 0.xx,
        "emotion_adjustment": "Suggested adjustment"
    }},
    "persuasion": {{
         "persuaded": "Yes/No/Partially",
         "persuaded_viewpoints": ["Viewpoint1", "Viewpoint2", ...],
         "persuaded_agent_ids": [ID1, ID2, ...],
         "further_questions": "Further inquiry"
    }}
}}
"""
        response = self._call_llm(prompt)
        print(f"[Agent {self.id} deep_reflection] {response}")
        reflection_data = self._parse_reflection_json(response)
        key = discussion_id if discussion_id is not None else str(round_num)
        structured_data = {
            "consensus": reflection_data.get("consensus", []),
            "accepted": reflection_data.get("accepted", {}),
            "rejected": reflection_data.get("rejected", {}),
            "stance_change": reflection_data.get("stance_change", False),
            "new_stance_value": reflection_data.get("new_stance_value", self.attributes.get("stance_value")),
            "keywords": reflection_data.get("keywords", []),
            "keywords_opinion": reflection_data.get("keywords_opinion", {}),
            "network_changes": reflection_data.get("network_changes", {"add": [], "remove": [], "reason": ""}),
            "supporting_agents": reflection_data.get("supporting_agents", []),
            "opposing_agents": reflection_data.get("opposing_agents", []),
            "emotion_analysis": reflection_data.get("emotion_analysis", {"current_emotion": self.emotion, "emotion_adjustment": ""}),
            "persuasion": reflection_data.get("persuasion", {"persuaded": False, "persuaded_viewpoints": [], "persuaded_agent_ids": [], "further_questions": ""})
        }
        self.memory["deep_reflections"][key] = structured_data
        self.memory.setdefault("stance_value_history", []).append({
            "round": round_num,
            "stance_value": self.attributes.get("stance_value")
        })
        self.memory.setdefault("emotion_history", []).append({
            "round": round_num,
            "emotion": self.emotion
        })
        self._update_deep_reflection(structured_data, key)
        return structured_data


    def _update_deep_reflection(self, reflection_data: Dict, key):
        if reflection_data.get("stance_change", False):
            try:
                new_val = float(reflection_data.get("new_stance_value", self.attributes.get("stance_value")))
            except Exception:
                new_val = self.attributes.get("stance_value")
            self.attributes["stance_value"] = max(0.0, min(1.0, new_val))
        emo_data = reflection_data.get("emotion_analysis", {})
        if emo_data.get("current_emotion") is not None:
            try:
                new_emo = float(emo_data.get("current_emotion"))
                self.emotion = max(0.0, min(1.0, new_emo))
            except:
                pass

    def query_initial_willingness(self) -> float:
        prompt = f"""
[Current State] Emotion: {self.emotion}; Stance: {self.attributes.get('stance_value', 0.0)}
Please evaluate your willingness to participate in this discussion (0~1, where 1 means very willing), and briefly explain why.
Return only a single number."""
        response = self._call_llm(prompt)
        print(f"[Agent {self.id} query_initial_willingness] {response}")
        try:
            willingness = float(re.search(r"[\d\.]+", response).group())
        except Exception as e:
            print(f"Failed to parse initial willingness: {str(e)}")
            willingness = 0.50

        continue_prompt = "Are you interested in continuing the discussion on this topic? Please answer 'Yes' or 'No' and briefly explain why."
        continue_response = self._call_llm(continue_prompt)
        try:
            continue_interest = json.loads(continue_response)
        except Exception:
            continue_interest = continue_response.strip()
        self.memory["deep_reflections"].setdefault("continue_interest", continue_interest)
        return willingness

    def initialize_event_cognition(self, event: str) -> None:
        prompt = f"""
Please read the following event description and describe your initial cognition (output only your view without including your persona numbers), extract keywords, and evaluate your emotional tendency and stance on the event (both in the range 0~1).
Return in JSON format, for example:
{{
  "basic_cognition": "Your understanding",
  "keywords": ["Keyword1", "Keyword2", "Keyword3"],
  "emotional_tendency": 0.xx,
  "stance_value": 0.xx
}}
Event description: {event}
"""
        response = self._call_llm(prompt)
        print(f"[Agent {self.id} initialize_event_cognition] {response}")
        try:
            json_text = re.search(r'\{.*\}', response, re.DOTALL)
            if json_text:
                data = json.loads(json_text.group())
            else:
                raise ValueError("Failed to extract JSON block")
        except Exception as e:
            print(f"Event cognition parsing failed: {str(e)}")
            data = {
                "basic_cognition": "",
                "keywords": [],
                "emotional_tendency": 0.0,
                "stance_value": self.attributes.get("stance_value")
            }
        self.memory["event_cognition"] = data
        try:
            self.emotion = max(0.0, min(1.0, float(data.get("emotional_tendency", 0.0))))
        except:
            self.emotion = 0.0
        try:
            self.attributes['stance_value'] = max(0.0, min(1.0, float(data.get("stance_value"))))
        except:
            self.attributes['stance_value'] = 0.0
        
        self.memory["deep_reflections"]["0"] = {
            "basic_cognition": data.get("basic_cognition", ""),
            "keywords": data.get("keywords", []),
            "emotion_analysis": {"current_emotion": self.emotion},
            "stance_value": self.attributes.get("stance_value")
        }
        
        interest_prompt = "Are you interested in the above event? If yes, list the related topics you are interested in as a JSON array; if not, return []."
        interest_response = self._call_llm(interest_prompt)
        try:
            topic_interest = json.loads(interest_response)
        except Exception:
            topic_interest = []
        self.memory["deep_reflections"].setdefault("topic_interest", topic_interest)
        self.save_profile()

    def _call_llm(self, prompt: str) -> str:
        full_prompt = f"You are Agent {self.id}.\n" + self.get_profile_str() + "\n" + prompt
        attempts = 0
        max_attempts = len(API_KEYS)
        while attempts < max_attempts:
            api_key = API_KEYS[attempts % len(API_KEYS)]
            try:
                client = OpenAI(
                    api_key=api_key,
                    base_url="https://api.deepseek.com"
                )
                response = client.chat.completions.create(
                    model="deepseek-reasoner",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant"},
                        {"role": "user", "content": full_prompt}
                    ],
                    temperature=1.3,
                    stream=False
                )
                result = response.choices[0].message.content.strip()
                if not result:
                    print("Warning: DeepSeek API returned an empty response")
                    return ""
                return result
            except Exception as e:
                print(f"API key {api_key} request failed on attempt {attempts+1}: {str(e)}")
                attempts += 1
        print("All API keys failed. Returning empty response.")
        return ""

    def _parse_reflection(self, response: str) -> Dict:
        def safe_extract(pattern, default):
            clean_response = re.sub(r'\[Agent \d+ immediate_reflection\].*', '', response, flags=re.DOTALL)
            match = re.search(pattern, clean_response, flags=re.IGNORECASE | re.MULTILINE)
            return match.group(1).strip() if match else default

        logic = safe_extract(r"Logical Consistency\s*[:：]\s*(.+)", "High")
        sentiment = safe_extract(r"Emotional Tendency\s*[:：]\s*(.+)", "Neutral")
        alignment = safe_extract(r"Stance Alignment\s*[:：]\s*(.+)", "High")
        rebut_text = safe_extract(r"Need to Refute\s*[:：]\s*([^。\n]+)", "No").lower()
        rebut_flag = False if any(kw in rebut_text for kw in ["no need", "not necessary", "no", "false"]) else True
        continue_text = safe_extract(r"(?:Continue Discussion)\s*[:：]\s*(.+?)\s*(?:\n|$)", "No").strip().lower()
        continue_flag = any(kw in continue_text for kw in ["yes", "continue", "necessary", "true"])
        comp_reason = safe_extract(r"Overall Reflection\s*[:：]\s*(.*?)(?=\n|$)", "No explanation")[:500]
        emotion_str = safe_extract(r"Emotion Value\s*[:：]\s*(\d+\.?\d*)", "")
        try:
            emotion_val = float(emotion_str)
            self.emotion = max(0.0, min(1.0, emotion_val))
        except:
            emotion_val = self.emotion

        persuaded_text = safe_extract(r"Persuaded\s*[:：]\s*(.+)", "No")
        persuaded_flag = any(kw in persuaded_text for kw in ["yes", "true", "persuaded"])
        persuaded_ids_text = safe_extract(r"Persuasion Sources\s*[:：]\s*(.+)", "")
        persuaded_agent_ids = [x.strip() for x in re.split(r"[,，\s]+", persuaded_ids_text) if x.strip().isdigit()] if persuaded_ids_text else []
        further_question = safe_extract(r"Further Questions\s*[:：]\s*(.*?)(?=\n|$)", "")

        return {
            "logic": logic,
            "sentiment": sentiment,
            "alignment": alignment,
            "rebut": rebut_flag,
            "continue": continue_flag,
            "reason": comp_reason,
            "emotion_val": round(self.emotion, 2),
            "persuaded": persuaded_flag,
            "persuaded_agent_ids": persuaded_agent_ids,
            "further_question": further_question
        }

    def _parse_reflection_json(self, response: str) -> Dict:
        try:
            cleaned = re.sub(r'[\x00-\x1F]', '', response)
            cleaned = re.sub(r'\\\'', "'", cleaned)
            data = json.loads(cleaned)
            new_val_raw = data.get("new_stance_value", self.attributes.get("stance_value"))
            try:
                new_stance_value = float(new_val_raw) if new_val_raw is not None else self.attributes.get("stance_value")
            except Exception:
                new_stance_value = self.attributes.get("stance_value")
            persuasion = data.get("persuasion", {"persuaded": False, "persuaded_viewpoints": [], "persuaded_agent_ids": [], "further_questions": ""})
            reflection_data = {
                "consensus": data.get("consensus", []),
                "accepted": data.get("accepted", {}),
                "rejected": data.get("rejected", {}),
                "stance_change": data.get("stance_change", "No") == "Yes",
                "new_stance_value": new_stance_value,
                "keywords": data.get("keywords", []),
                "keywords_opinion": data.get("keywords_opinion", {}),
                "network_changes": data.get("network_changes", {"add": [], "remove": [], "reason": ""}),
                "supporting_agents": data.get("supporting_agents", []),
                "opposing_agents": data.get("opposing_agents", []),
                "emotion_analysis": data.get("emotion_analysis", {"current_emotion": 0.0, "emotion_adjustment": ""}),
                "persuasion": persuasion
            }
            return reflection_data
        except json.JSONDecodeError:
            return self._get_default_reflection()

    def _get_default_reflection(self) -> Dict:
        return {
            "consensus": [],
            "accepted": {},
            "rejected": {},
            "stance_change": False,
            "new_stance_value": self.attributes.get("stance_value"),
            "keywords": [],
            "keywords_opinion": {},
            "network_changes": {"add": [], "remove": [], "reason": ""},
            "supporting_agents": [],
            "opposing_agents": [],
            "emotion_analysis": {"current_emotion": 0.0, "emotion_adjustment": ""},
            "persuasion": {"persuaded": False, "persuaded_viewpoints": [], "persuaded_agent_ids": [], "further_questions": ""}
        }

    def _format_dialogues(self, dialogues: List[Dict]) -> str:
        if not dialogues:
            return "(No dialogue records available)"
        return "\n".join([f"Agent {m.get('agent', m.get('speaker', 'Unknown'))}: {m.get('content')}" for m in dialogues])

    def _analyze_response(self, response: str):
        positive = len(re.findall(r"\b(support|agree|beneficial)\b", response, flags=re.IGNORECASE))
        negative = len(re.findall(r"\b(oppose|doubt|disadvantage)\b", response, flags=re.IGNORECASE))
        delta = (positive - negative) * 0.1
        self.emotion = max(0.0, min(1.0, self.emotion + delta))


    def __del__(self):
        pass


def get_education_and_occupation(age: int) -> (str, str):
    if 18 <= age <= 22:
        educations = ["High School Graduate", "College (in progress)", "University (in progress)"]
        occupations = ["Student"]
    elif 23 <= age <= 26:
        educations = ["University Graduate"]
        occupations = ["Junior Staff", "Assistant", "Sales Representative"]
    elif 27 <= age <= 35:
        educations = ["University Graduate", "Master's (in progress)", "Master's Graduate"]
        occupations = ["Engineer", "Teacher", "Doctor", "Staff"]
    elif 36 <= age <= 50:
        educations = ["University Graduate", "Master's Graduate", "PhD (in progress)", "PhD Graduate"]
        occupations = ["Middle Management", "Senior Engineer", "Professor", "Doctor"]
    elif 51 <= age <= 65:
        educations = ["University Graduate", "Master's Graduate", "PhD Graduate"]
        occupations = ["Senior Management", "Expert", "Professor", "Doctor"]
    else:
        educations = ["Unknown"]
        occupations = ["Unknown"]
    return random.choice(educations), random.choice(occupations)


class Simulation:
    def __init__(self, num_agents: int):
        self.topic = (
            "In the 2024 Paris Olympics, Algerian boxer Iman Halif won the gold medal in the women's 66 kg competition. "
            "Previously, the International Boxing Association (IBA) disqualified her from the 2023 World Championships due to a chromosomal test showing XY. "
            "Medical experts suggest that Halif may be a DSD (Differences of Sex Development) athlete, which could give her a competitive advantage in strength and muscle development; "
            "however, the International Olympic Committee (IOC) allowed her to compete, emphasizing that she has been recognized as female since birth and her passport shows female."
        )
        self.agents = self._initialize_agents(num_agents)
        self.round = 0
        self.discussion_counter = 0
        self.records = {
            "stance_evolution": defaultdict(list),
            "network_evolution": [],
            "persuasion_history": defaultdict(list),
            "node_degree_history": defaultdict(list),
            "betweenness_history": defaultdict(list),
            "clustering_history": defaultdict(list),
            "kshe_history": defaultdict(list),
            "contrality_history": defaultdict(list),
            "keyword_frequency": {},
            "persuasion_events": [],
            "connections_history": defaultdict(dict),
            "accepted_arguments_history": defaultdict(dict),
            "rejected_arguments_history": defaultdict(dict),
            "topic_history": defaultdict(dict),
            "deep_reflection_history": defaultdict(dict)
        }
        self._initialize_network(num_agents)

    def update_agents_willingness(self):
        print("\nUpdating agents' willingness to speak:")
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(agent.query_initial_willingness): aid for aid, agent in self.agents.items()}
            for future in concurrent.futures.as_completed(futures):
                aid = futures[future]
                try:
                    willingness = future.result()
                    print(f"Agent {aid} willingness: {willingness}")
                    self.agents[aid].memory.setdefault("willingness_history", {})[self.round] = willingness
                except Exception as e:
                    print(f"Error in querying willingness for Agent {aid}: {e}")

    def generate_reports(self):
        G = nx.Graph()
        for aid, agent in self.agents.items():
            deep = agent.memory["deep_reflections"].get(str(self.round), {})
            topics = deep.get("keywords", []) if deep.get("keywords") else []
            unique_keywords = topics[:3] if topics else []
            original_stance = (deep.get("stance_value", agent.attributes.get('stance_value', 0.0)))
            original_emotion = deep.get("emotion_analysis", {}).get("current_emotion", agent.emotion)
            G.add_node(aid,
                       emotion=agent.emotion,
                       original_emotion=original_emotion,
                       stance_value=agent.attributes.get('stance_value', 0.0),
                       original_stance=original_stance,
                       age=agent.attributes.get('age', 'Unknown'),
                       gender=agent.attributes.get('gender', 'Unknown'),
                       keywords=unique_keywords)
        for aid, agent in self.agents.items():
            for conn in agent.connections:
                if aid < conn:
                    G.add_edge(aid, conn)
        self.records['network_evolution'].append((self.round, G))
        
        for aid, agent in self.agents.items():
            agent.save_memory(self.round)
            self.records['stance_evolution'][aid].append((self.round, round(agent.attributes.get('stance_value', 0.0), 2)))
            self.records["persuasion_history"][aid].append((self.round, agent.get_persuasion_rate()))
            self.records["connections_history"][aid][self.round] = agent.memory.get("connections_history", {}).get(self.round, [])
            self.records["accepted_arguments_history"][aid][self.round] = [] 
            self.records["rejected_arguments_history"][aid][self.round] = [] 
            self.records["topic_history"][aid][self.round] = {}
            self.records["deep_reflection_history"][aid][self.round] = agent.memory.get("deep_reflections", {}).get(str(self.round), {})
        
        betweenness = nx.betweenness_centrality(G)
        clustering = nx.clustering(G)
        kshe = nx.core_number(G)
        try:
            centrality = nx.eigenvector_centrality(G)
        except Exception:
            centrality = nx.eigenvector_centrality_numpy(G)
        for node in G.nodes():
            self.records["node_degree_history"][node].append((self.round, G.degree(node)))
            self.records.setdefault("betweenness_history", defaultdict(list))[node].append((self.round, betweenness.get(node, 0)))
            self.records.setdefault("clustering_history", defaultdict(list))[node].append((self.round, clustering.get(node, 0)))
            self.records.setdefault("kshe_history", defaultdict(list))[node].append((self.round, kshe.get(node, 0)))
            self.records.setdefault("contrality_history", defaultdict(list))[node].append((self.round, centrality.get(node, 0)))
        
        round_keyword_freq = defaultdict(int)
        for aid, agent in self.agents.items():
            deep = agent.memory["deep_reflections"].get(str(self.round), {})
            for keyword in deep.get("keywords", []):
                round_keyword_freq[keyword] += 1
        self.records["keyword_frequency"][self.round] = dict(round_keyword_freq)

        self.plot_persuasion_rate_line_chart()
        self.plot_node_degree_line_chart()

    def plot_emotion_line_chart(self):
        plt.figure(figsize=(10, 6))
        for aid, agent in self.agents.items():
            history = []
            for key, record in agent.memory["deep_reflections"].items():
                try:
                    r = int(key)
                except:
                    r = 0
                emotion = record.get("emotion_analysis", {}).get("current_emotion", agent.emotion)
                history.append((r, emotion))
            if not history:
                continue
            history.sort(key=lambda x: x[0])
            rounds = [h[0] for h in history]
            emotions = [h[1] for h in history]
            plt.plot(rounds, emotions, marker='o', label=f"Agent {aid}")
        plt.xlabel("Round")
        plt.ylabel("Emotion Value")
        plt.title("Emotion Evolution Line Chart for Each Agent")
        plt.legend()
        plt.tight_layout()
        plt.savefig("emotion_line_chart.png")
        plt.close()

    def plot_stance_line_chart(self):
        plt.figure(figsize=(10, 6))
        for aid, agent in self.agents.items():
            history = []
            for key, record in agent.memory["deep_reflections"].items():
                try:
                    r = int(key)
                except:
                    r = 0
                stance = record.get("stance_value", agent.attributes.get("stance_value", 0.0))
                history.append((r, stance))
            if not history:
                continue
            history.sort(key=lambda x: x[0])
            rounds = [h[0] for h in history]
            stances = [h[1] for h in history]
            plt.plot(rounds, stances, marker='o', label=f"Agent {aid}")
        plt.xlabel("Round")
        plt.ylabel("Stance Value")
        plt.title("Stance Evolution Line Chart for Each Agent")
        plt.legend()
        plt.tight_layout()
        plt.savefig("stance_line_chart.png")
        plt.close()

    def plot_persuasion_rate_line_chart(self):
        plt.figure(figsize=(10, 6))
        for aid, data in self.records["persuasion_history"].items():
            rounds = [item[0] for item in data]
            rates = [item[1] for item in data]
            plt.plot(rounds, rates, marker='o', label=f"Agent {aid}")
        plt.xlabel("Round")
        plt.ylabel("Persuasion Rate")
        plt.title("Persuasion Rate Evolution Line Chart for Each Agent")
        plt.legend()
        plt.tight_layout()
        plt.savefig("persuasion_rate_line_chart.png")
        plt.close()

    def plot_node_degree_line_chart(self):
        plt.figure(figsize=(10, 6))
        for node, data in self.records["node_degree_history"].items():
            rounds = [item[0] for item in data]
            degrees = [item[1] for item in data]
            plt.plot(rounds, degrees, marker='o', label=f"Agent {node}")
        plt.xlabel("Round")
        plt.ylabel("Node Degree")
        plt.title("Node Degree Evolution Line Chart")
        plt.legend()
        plt.tight_layout()
        plt.savefig("node_degree_line_chart.png")
        plt.close()

    def plot_betweenness_line_chart(self):
        plt.figure(figsize=(10, 6))
        for node, data in self.records.get("betweenness_history", {}).items():
            rounds = [item[0] for item in data]
            values = [item[1] for item in data]
            plt.plot(rounds, values, marker='o', label=f"Agent {node}")
        plt.xlabel("Round")
        plt.ylabel("Betweenness Centrality")
        plt.title("Betweenness Centrality Evolution Line Chart for Each Agent")
        plt.legend()
        plt.tight_layout()
        plt.savefig("betweenness_line_chart.png")
        plt.close()

    def plot_clustering_line_chart(self):
        plt.figure(figsize=(10, 6))
        for node, data in self.records.get("clustering_history", {}).items():
            rounds = [item[0] for item in data]
            values = [item[1] for item in data]
            plt.plot(rounds, values, marker='o', label=f"Agent {node}")
        plt.xlabel("Round")
        plt.ylabel("Clustering Coefficient")
        plt.title("Clustering Coefficient Evolution Line Chart for Each Agent")
        plt.legend()
        plt.tight_layout()
        plt.savefig("clustering_line_chart.png")
        plt.close()

    def plot_keyword_frequency_bar_chart(self, round_num: int):
        freq = self.records["keyword_frequency"].get(round_num, {})
        if not freq:
            print(f"No keywords recorded in round {round_num}.")
            return
        keywords = list(freq.keys())
        counts = [freq[k] for k in keywords]
        plt.figure(figsize=(10, 6))
        plt.bar(keywords, counts, color='skyblue')
        plt.xlabel("Keywords")
        plt.ylabel("Frequency")
        plt.title(f"Keyword Frequency for Round {round_num}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"keyword_frequency_round{round_num}.png")
        plt.close()

    def plot_kshe_line_chart(self):
        ks_values = defaultdict(list)
        for rnd, G in self.records["network_evolution"]:
            core_numbers = nx.core_number(G)
            for node, core in core_numbers.items():
                ks_values[node].append((rnd, core))
        plt.figure(figsize=(10, 6))
        for node, values in ks_values.items():
            values.sort(key=lambda x: x[0])
            rounds = [v[0] for v in values]
            cores = [v[1] for v in values]
            plt.plot(rounds, cores, marker='o', label=f"Agent {node}")
        plt.xlabel("Round")
        plt.ylabel("K-shell Value")
        plt.title("K-shell Evolution Line Chart for Each Node")
        plt.legend()
        plt.tight_layout()
        plt.savefig("kshe_line_chart.png")
        plt.close()

    def plot_contrality_line_chart(self):
        contrality_values = defaultdict(list)
        for rnd, G in self.records["network_evolution"]:
            try:
                centrality = nx.eigenvector_centrality(G)
            except Exception:
                centrality = nx.eigenvector_centrality_numpy(G)
            for node, cent in centrality.items():
                contrality_values[node].append((rnd, cent))
        plt.figure(figsize=(10, 6))
        for node, values in contrality_values.items():
            values.sort(key=lambda x: x[0])
            rounds = [v[0] for v in values]
            cents = [v[1] for v in values]
            plt.plot(rounds, cents, marker='o', label=f"Agent {node}")
        plt.xlabel("Round")
        plt.ylabel("Eigenvector Centrality")
        plt.title("Eigenvector Centrality Evolution Line Chart for Each Agent")
        plt.legend()
        plt.tight_layout()
        plt.savefig("contrality_line_chart.png")
        plt.close()


    def _initialize_network(self, num_agents: int):
        k = 4 if num_agents >= 4 else max(1, num_agents - 1)
        p = 0.0  
        G = nx.watts_strogatz_graph(num_agents, k=k, p=p)
        self.initial_network = {} 
        for i in range(num_agents):
            if i not in G:
                G.add_node(i)
            valid_connections = [n for n in G.neighbors(i) if n < num_agents]
            self.agents[i].connections = valid_connections
            self.agents[i]._sim = self
            self.initial_network[i] = valid_connections 
    
        print("\n[Initial Network Connections]:")
        for aid, conns in self.initial_network.items():
            print(f"Agent {aid}: {conns}")

    def save_round_dialogue_record(self, round_num: int):
        round_dialogues = []
        for agent in self.agents.values():
            dialogues = agent.memory['short_term'].get(round_num, [])
            if dialogues:
                round_dialogues.extend(dialogues)
        filename = f"dialogues_round_{round_num}.json"
        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(round_dialogues, f, ensure_ascii=False, indent=2)
            print(f"Saved dialogue records for round {round_num} in {filename}")
        except Exception as e:
            print(f"Failed to save dialogue record for round {round_num}: {str(e)}")

    def save_round_network_connections(self, round_num: int):
        connections_record = {}
        for aid, agent in self.agents.items():
            connections_record[aid] = agent.connections
        filename = f"network_connections_round_{round_num}.json"
        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(connections_record, f, ensure_ascii=False, indent=2)
            print(f"Saved network connections for round {round_num} in {filename}")
        except Exception as e:
            print(f"Failed to save network connections for round {round_num}: {str(e)}")

    def run(self, total_rounds=10):
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            for _ in range(total_rounds):
                self.round += 1
                print(f"\n{'#'*40}\nRound {self.round} simulation begins\n{'#'*40}")
                self.update_agents_willingness()
                initiators = self._select_initiators()
                if not initiators:
                    print("No initiator in this round.")
                    continue
                for initiator in initiators:
                    participants = self._select_participants(initiator)
                    self._run_discussion(initiator, participants)
                self.generate_reports()
                self.save_round_dialogue_record(self.round)
                self.save_round_network_connections(self.round)
        print("\n[Simulation Ends] Global opinion leader analysis:")
        self.analyze_opinion_leaders()
        self.report_persuasion_events()
        with open("simulation_records.json", "w", encoding="utf-8") as f:
            json.dump(self.records, f, ensure_ascii=False, indent=2)
        print("\nAll simulation records have been saved to 'simulation_records.json'.")

    def _select_initiators(self) -> List[int]:
        initiators = []
        for aid, agent in self.agents.items():
            willingness = agent.memory.get("willingness_history", {}).get(self.round, 0)
            if willingness > 0.6:
                print(f"Agent {aid} selected as initiator (willingness={willingness:.2f})")
                initiators.append(aid)
        if not initiators:
            print(f"[Warning] Round {self.round} has no initiators")
        else:
            print(f"Selected {len(initiators)} initiators: {initiators}")
        return initiators

    def _run_discussion(self, initiator: int, participants: List[int]):
        self.discussion_counter += 1
        discussion_id = f"R{self.round}D{self.discussion_counter}"
        print(f"\n[Discussion Begins] {discussion_id} | Initiator: Agent {initiator}")
        context = []
        current_speaker = initiator
        active = True
        turn = 0
        discussion_state = {pid: {"active": True, "engagement": 1.0} for pid in participants}
        for pid in participants:
            self.agents[pid]._current_discussion = discussion_state[pid]
        while active and turn < 30:
            if not discussion_state[current_speaker]["active"]:
                print(f"Agent {current_speaker} has left the discussion")
                current_speaker = self._select_next_speaker(current_speaker, participants, discussion_state)
                if current_speaker is None:
                    break
                continue
            response = self.agents[current_speaker].generate_response(context, is_initiator=(turn == 0 and current_speaker == initiator))
            self.agents[current_speaker].persuasion_stats["dialogue_count"] += 1
            if response == "[Exit Discussion]":
                print(f"\n[Turn {turn+1}] Agent {current_speaker} has chosen to exit the discussion")
                discussion_state[current_speaker]["active"] = False
                active = self._check_discussion_active(discussion_state)
                current_speaker = self._select_next_speaker(current_speaker, participants, discussion_state)
                if current_speaker is None:
                    break
                continue
            print(f"\n[Turn {turn+1}] Agent {current_speaker}: {response}")
            context_entry = {"discussion_id": discussion_id, "speaker": current_speaker, "content": response, "round": self.round}
            context.append(context_entry)
            for pid in participants:
                self.agents[pid].memory['short_term'][self.round].append(context_entry)
            active = self._process_immediate_reflections(current_speaker, response, participants, discussion_state, discussion_id)
            current_speaker = self._select_next_speaker(current_speaker, participants, discussion_state)
            if current_speaker is None:
                break
            turn += 1
        for pid in participants:
            self.agents[pid]._current_discussion = None
        print(f"[Discussion Ends] Total of {turn} turns.")
        self._process_deep_reflections(participants, discussion_id)

    def _process_immediate_reflections(self, speaker: int, statement: str, participants: List[int], discussion_state: Dict, discussion_id: str) -> bool:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(self.agents[pid].immediate_reflection, speaker, statement, discussion_id): pid for pid in participants if pid != speaker}
            for future in concurrent.futures.as_completed(futures):
                pid = futures[future]
                try:
                    reflection = future.result()
                    discussion_state[pid]["active"] = reflection.get("continue", True)
                except Exception as e:
                    print(f"Error in immediate reflection for Agent {pid}: {e}")
        active_count = sum(1 for pid in participants if discussion_state[pid]["active"])
        return active_count >= 1

    def _check_discussion_active(self, discussion_state: Dict) -> bool:
        return sum(1 for state in discussion_state.values() if state["active"]) >= 1

    def _select_participants(self, initiator: int) -> List[int]:
        agent = self.agents[initiator]
        available = [cid for cid in agent.connections if cid in self.agents and cid != initiator]
        print(f"\n[Decision] Agent {initiator} selecting participants, available connections: {available}")
        prompt = f"""Please decide (only choose one format):
1. Group discussion: Select more than 2 participants from {available}. 
   [Recommendation: Group discussion is encouraged to facilitate broader exchanges and to help you get to know more agents, including those you are not familiar with.]
2. One-on-one discussion: Select 1-2 participants from {available}.

Please respond in the following format:
Discussion Format: [One-on-one/Group]
Select Participants: [List of IDs]
Reason: ..."""
        response = agent._call_llm(prompt)
        print(f" Decision response: {response}")
        formats = re.findall(r"Discussion Format\s*[:：]\s*(One-on-one|Group)", response)

        selected_format = formats[0] if formats else "Group"
        if "Group" in formats:
            selected_format = "Group"
        participants_lines = re.findall(r"Select Participants\s*[:：]\s*(.*)", response)
        participant_str = participants_lines[0] if participants_lines else ""
        try:
            if participant_str.strip().startswith("["):
                candidates = json.loads(participant_str)
            else:
                candidates = list(map(int, re.findall(r"\d+", participant_str)))
        except Exception as e:
            print(f"Error parsing participant IDs: {e}")
            candidates = []
        if selected_format == "One-on-one":
            valid_participants = [initiator]
            if candidates:
                valid_participants.append(candidates[0])
            elif available:
                valid_participants.append(random.choice(available))
        else:
            if candidates:
                valid_participants = list(set([initiator] + candidates))
            else:
                valid_participants = list(set([initiator] + available))
        print(f" Final participants: {valid_participants}")
        return valid_participants



    def _select_next_speaker(self, current: int, participants: List[int], discussion_state: Dict) -> int:
        candidates = []
        for pid in participants:
            if pid == current or not discussion_state[pid]["active"]:
                continue
            agent = self.agents[pid]
            willingness = agent.memory.get("willingness_history", {}).get(self.round, 0.5)
            imm = agent.memory.get("immediate_reflections", {}).get(self.round, {}).get(pid)
            if imm is not None and imm.get("rebut") is True:
                weight = willingness * 10
            else:
                weight = willingness
            candidates.append((pid, weight))
        if not candidates:
            return None
        total = sum(w for _, w in candidates)
        rand = random.uniform(0, total)
        cumulative = 0
        for pid, w in candidates:
            cumulative += w
            if rand <= cumulative:
                return pid
        return candidates[-1][0]

    def _process_deep_reflections(self, participants: List[int], discussion_id: str):
        print(f"\n[Deep Reflection Phase] (Discussion ID: {discussion_id}, Participating Agents: {participants})")
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(self.agents[pid].deep_reflection, self.round, discussion_id): pid for pid in participants}
            for future in concurrent.futures.as_completed(futures):
                pid = futures[future]
                try:
                    agent = self.agents[pid]
                    reflection = future.result()
                    agent.save_memory(self.round)
                    persuasion_data = reflection.get("persuasion", {})
                    persuaded_value = persuasion_data.get("persuaded", "No")
                    if persuaded_value == "Yes":
                        event_weight = 1
                    elif persuaded_value == "Partially":
                        event_weight = 0.5
                    else:
                        event_weight = 0
                    if event_weight > 0:
                        persuader_ids = persuasion_data.get("persuaded_agent_ids", [])
                        for pers_id in persuader_ids:
                            if pers_id in self.agents:
                                self.agents[pers_id].persuasion_stats["persuasion_count"] += event_weight
                        event = {
                            "round": self.round,
                            "discussion_id": discussion_id,
                            "target_agent": pid,
                            "persuader_ids": persuader_ids,
                            "persuader_weight": event_weight,
                            "statement": "[Deep Reflection Record]",
                            "speaker": None
                        }
                        self.records["persuasion_events"].append(event)
                except Exception as e:
                    print(f"Error processing deep reflection for Agent {pid}: {e}")

    def analyze_opinion_leaders(self):
        print("\n[Global Opinion Leader Analysis]")
        print("Global opinion leader analysis function is not enabled.")

    def report_persuasion_events(self):
        print("\n[Persuasion Events Record]")
        if not self.records["persuasion_events"]:
            print("No persuasion events were recorded in this simulation.")
        else:
            for event in self.records["persuasion_events"]:
                print(f"In round {event['round']}, Discussion {event['discussion_id']}, Agent {event['target_agent']} was recorded as being persuaded, "
                      f"persuasion sources: {', '.join(event['persuader_ids'])}, counted as {event['persuader_weight']} times.")

    def _initialize_agents(self, n: int) -> Dict[int, 'Agent']:
        profiles = {
            "nationality": [
                "USA", "China", "India", "Brazil", "Russia", "Japan", "Germany", "France", "UK", "Australia",
                "Canada", "South Korea", "Mexico", "South Africa", "Egypt", "Argentina", "Singapore", "Saudi Arabia", "UAE", "Sweden",
                "Netherlands", "Italy", "Spain", "Switzerland", "Norway", "Philippines", "Nigeria", "Kenya", "Ukraine", "Vietnam",
                "Malaysia", "Poland", "Turkey", "Colombia", "New Zealand", "Chile", "Indonesia", "Tanzania", "Qatar"
            ],
            "age": list(range(18, 66)),
            "gender": ["Male", "Female", "LGBT"]
        }
        agents = {}
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {}
            for i in range(n):
                nationality = random.choice(profiles["nationality"])
                age = random.choice(profiles["age"])
                gender = random.choice(profiles["gender"])
                education, occupation = get_education_and_occupation(age)
                attributes = {
                    "nationality": nationality,
                    "age": age,
                    "gender": gender,
                    "education": education,
                    "occupation": occupation
                }
                agents[i] = Agent(i, attributes, [])
                futures[executor.submit(agents[i].initialize_event_cognition, self.topic)] = i
            for future in concurrent.futures.as_completed(futures):
                i = futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"Error initializing event cognition for Agent {i}: {e}")
        return agents

    def final_analysis(self):
        self.plot_emotion_line_chart()
        self.plot_stance_line_chart()
        self.plot_persuasion_rate_line_chart()
        self.plot_node_degree_line_chart()
        self.plot_betweenness_line_chart()
        self.plot_clustering_line_chart()
        self.plot_keyword_frequency_bar_chart(self.round)
        self.plot_kshe_line_chart()
        self.plot_contrality_line_chart()
        print("Line charts have been generated:")
        print("emotion_line_chart.png, stance_line_chart.png, persuasion_rate_line_chart.png,")
        print("node_degree_line_chart.png, betweenness_line_chart.png, clustering_line_chart.png,")
        print(f"keyword_frequency_round{self.round}.png, kshe_line_chart.png, contrality_line_chart.png")
        print("\nDetailed per-agent records:")
        for aid in self.agents:
            print(f"Agent {aid}:")
            print(f"  Connections History: {self.records['connections_history'].get(aid, {})}")
            print(f"  Accepted Arguments History: {self.records.get('accepted_arguments_history', {}).get(aid, {})}")
            print(f"  Rejected Arguments History: {self.records.get('rejected_arguments_history', {}).get(aid, {})}")
            print(f"  Topic History: {self.records.get('topic_history', {}).get(aid, {})}")
            print(f"  Deep Reflection History: {self.records.get('deep_reflection_history', {}).get(aid, {})}")

        self.report_persuasion_events()
        with open("simulation_records.json", "w", encoding="utf-8") as f:
            json.dump(self.records, f, ensure_ascii=False, indent=2)
        print("\nAll simulation records have been saved to 'simulation_records.json'.")


if __name__ == "__main__":
    sim = Simulation(num_agents=20)
    sim.run(total_rounds=25)
    sim.final_analysis()

