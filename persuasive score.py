import json
import re
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt


def parse_combined_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    marker = "[Discussion Begins]"
    _, _, log_part = content.partition(marker)
    return marker + log_part


def parse_all_rounds(log_content):
    split_pat = re.compile(r'\[Discussion Begins\]\s*(R\d+D\d+)\s*\|')
    parts = split_pat.split(log_content)
    if len(parts) < 2:
        raise RuntimeError("no found")

    rounds_data = {}
    rounds_order = []
    for i in range(1, len(parts), 2):
        rid = parts[i].strip()
        sec = parts[i + 1]
        rounds_order.append(rid)

        blk_dict = {}
        dpat = re.compile(
            r'\[Agent\s*(\d+)\s*deep_reflection\]\s*```json\s*({.*?})\s*```',
            re.DOTALL
        )
        for m in dpat.finditer(sec):
            aid = int(m.group(1))
            blk = json.loads(m.group(2))
            blk_dict[aid] = blk
        rounds_data[rid] = blk_dict

    return rounds_data, rounds_order


def compute_round_contributions(deep_dict):

    contributions = {}
    agents = list(deep_dict.keys())
    for i in agents:
        round_num = 0.0  
        round_den = 0.0 
        for j in agents:
            if j == i:
                continue
            blk = deep_dict[j]
            A = blk.get("accepted") or []
            R = blk.get("rejected") or []
            if isinstance(A, dict):
                A = [clean_agent_id(k) for k in A.keys()]
            if isinstance(R, dict):
                R = [clean_agent_id(k) for k in R.keys()]
            A = [clean_agent_id(a) for a in A]
            R = [clean_agent_id(r) for r in R]
            denom = len(A)
            if denom > 0:
                cntA = A.count(i)
                cntR = R.count(i)
                cji = max(cntA - cntR, 0)
                rji = cji / denom
            else:
                rji = 0.0
            round_num += rji
            round_den += 1
        contributions[i] = (round_num, round_den)
    return contributions


def clean_agent_id(x):
    if isinstance(x, int):
        return x
    if isinstance(x, str):
        x = x.strip()
        if x.lower().startswith("agent"):
            x = x.lower().replace("agent", "").strip()
        try:
            return int(x)
        except ValueError:
            return -1  
    return -1


def main():
    filepath = r"/dataset in chaos.txt"
    log_txt = parse_combined_file(filepath)
    rounds_data, order = parse_all_rounds(log_txt)
    # order, rounds_data = rename_rounds(order, rounds_data)

    all_agents = set()
    for r in order:
        all_agents.update(rounds_data[r].keys())
    all_agents = sorted(all_agents)


    cum_num = defaultdict(float)
    cum_den = defaultdict(float)
    cumulative_data = []

    cumulative_agents = set()  
    for r in order:
        round_contrib = compute_round_contributions(rounds_data[r])
        # 更新累计贡献
        for agent, (num_val, den_val) in round_contrib.items():
            cum_num[agent] += num_val
            cum_den[agent] += den_val
        cumulative_agents.update(round_contrib.keys())
        # 当前轮次的所有 agent（包括本轮和之前出现过的 agent）
        union_agents = set(all_agents) | cumulative_agents
        row = {'round': r}
        for agent in union_agents:
            if cum_den[agent] > 0:
                row[agent] = cum_num[agent] / cum_den[agent]
            else:
                row[agent] = None
        cumulative_data.append(row)


    df = pd.DataFrame(cumulative_data)
    df.set_index('round', inplace=True)
    df.fillna(method='ffill', inplace=True)  


    df.to_excel("cumulative_persuasion_rate.xlsx")


    plt.figure(figsize=(12, 9))
    cmap = plt.get_cmap('tab20')  


    for idx, agent in enumerate(sorted(all_agents)):
        color = cmap(idx % 20)  
        plt.plot(
            df.index, 
            df[agent], 
            label=f'Agent {agent}',
            marker='o',          
            markersize=3,        
            linewidth=1,         
            color=color       
        )


    plt.xlabel('Round', fontsize=12)
    plt.ylabel('Cumulative Persuasion Rate', fontsize=12)
    plt.title('Cumulative Persuasion Rate per Agent Over Rounds', fontsize=14)
    plt.xticks(rotation=45, ha='right')


    plt.legend(
        bbox_to_anchor=(1.05, 1), 
        loc='upper left',
        frameon=True,
        fontsize=9,
        title='Agents',
        title_fontsize=10
    )


    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()


    plt.show()

if __name__ == "__main__":
    main()
