import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import make_interp_spline
from scipy.stats import pearsonr, spearmanr
import re

sns.set(style="whitegrid")
plt.rcParams.update({'figure.autolayout': True, 'font.size': 12})

def parse_round_id(r):
    m = re.match(r"R(\d+)D(\d+)", r)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    else:
        return (float('inf'), float('inf'))

def main():
    persu_excel = "cumulative_persuasion_rate.xlsx"
    net_excel   = "network_structure.xlsx"


    net_df = pd.read_excel(net_excel)
    net_df = net_df[~net_df['round'].str.match(r"R1[456]D\d+")]
    uniq_rl = sorted(net_df['round'].unique(), key=parse_round_id)
    ord_map = {r:i for i,r in enumerate(uniq_rl)}
    net_df['round_order'] = net_df['round'].map(ord_map)


    df_cum = pd.read_excel(persu_excel, sheet_name="Sheet1")


    df_cum = df_cum[~df_cum['round'].str.match(r"R1[456]D\d+")]
    persuade_long = (
        df_cum
        .melt(id_vars=['round'], var_name='agent', value_name='cum_rate')
        .assign(agent=lambda d:d['agent'].astype(int))
    )
    mapping = net_df[['round','round_order']].drop_duplicates()
    persuade_long = persuade_long.merge(mapping, on='round', how='left')


    merged = pd.merge(
        net_df[['round','round_order','agent','closeness']],
        persuade_long[['round','round_order','agent','cum_rate']],
        on=['round','round_order','agent'],
        how='left'
    )

    last_unique_rounds = sorted(merged['round'].dropna().unique(), key=parse_round_id)[-5:]
    last_data = merged[merged['round'].isin(last_unique_rounds)].sort_values(['round_order', 'agent'])
        
    print(last_data[['round', 'agent', 'closeness', 'cum_rate']])


    # 4. Intercepted to R22D235
    if 'R22D235' in mapping['round'].values:
        max_ord = mapping.loc[mapping['round']=='R22D235', 'round_order'].iloc[0]
        merged = merged[merged['round_order'] <= max_ord]
    else:
        print("no found")


    results = []
    for agent, grp in merged.groupby('agent'):
        x = grp['closeness'].values
        y = grp['cum_rate'].values
        mask = ~np.isnan(x)&~np.isnan(y)
        if mask.sum()>=2:
            pr, pp = pearsonr(x[mask], y[mask])
            sr, sp = spearmanr(x[mask], y[mask])
        else:
            pr=pp=sr=sp=np.nan
        results.append({
            'agent':agent,'pearson_r':pr,'pearson_p':pp,
            'spearman_r':sr,'spearman_p':sp,'n_pairs':int(mask.sum())
        })
    corr_df = pd.DataFrame(results).sort_values('agent')
    with pd.ExcelWriter("correlation_results.xlsx") as writer:
        corr_df.to_excel(writer, sheet_name="Correlation", index=False)
        merged.to_excel(writer, sheet_name="Merged_Data", index=False)
    print("save at correlation_results.xlsx")


    rounds_sorted = sorted(merged['round'].unique(), key=parse_round_id)
    order_map = {r: i for i, r in enumerate(rounds_sorted)}

    seen_R, major_pos, major_lbl = set(), [], []
    for r in rounds_sorted:
        Rnum, _ = parse_round_id(r)

        if Rnum in (14, 15, 16):
            continue

        if Rnum >= 17:
            Rlab = f"R{Rnum - 3}"
        else:
            Rlab = f"R{Rnum}"
        if Rlab not in seen_R:
            seen_R.add(Rlab)
            major_pos.append(order_map[r])
            major_lbl.append(Rlab)


    all_agents = sorted(merged['agent'].unique())
    selected = [14]
    palette = {14:("#2ca02c", "#98df8a")}


# all_agents = sorted(merged['agent'].unique())
# # selected = [8, 14, 18]
# selected = [14]
# palette = {
#     # 8:  ("#1f77b4", "#aec7e8"),
#     14:  ("#d62728", "#ff9896"),
#     # 18:  ("#2ca02c", "#98df8a"),
# }

    fig, ax1 = plt.subplots(figsize=(14,8))
    ax2 = ax1.twinx()


    for a in all_agents:
        grp = merged[merged['agent'] == a].sort_values('round_order')
        x = grp['round_order']
        ax1.plot(x, grp['closeness'], color='#DDDDDD', linestyle='--', linewidth=1.5)
        ax2.plot(x, grp['cum_rate'],   color='#DDDDDD', linewidth=1.5)


    for a in selected:
        grp = merged[merged['agent']==a].sort_values('round_order')
        x = grp['round_order'].values
        y_cl, y_pr = grp['closeness'].values, grp['cum_rate'].values
        deep_color, light_color = palette[a]

    # Closeness
        mask_cl = ~np.isnan(y_cl)
        if mask_cl.sum()>=4:
            xs, ys = x[mask_cl], y_cl[mask_cl]
            xs_s = np.linspace(xs.min(), xs.max(), 300)
            spl = make_interp_spline(xs, ys, k=3)
            ax1.plot(xs_s, spl(xs_s), color=light_color, linewidth=3, label=f'Arjun Closeness')
        else:
            ax1.plot(x, y_cl, color=light_color, linewidth=3, label=f'Arjun Closeness')

    # Persuasion
        mask_pr = ~np.isnan(y_pr)
        if mask_pr.sum()>=4:
            xs, ys = x[mask_pr], y_pr[mask_pr]
            xs_s = np.linspace(xs.min(), xs.max(), 300)
            spl = make_interp_spline(xs, ys, k=3)
            ax2.plot(xs_s, spl(xs_s), color=deep_color, linewidth=3, linestyle='--', label=f'Arjun Persuade')
        else:
            ax2.plot(x, y_pr, color=deep_color, linewidth=3, linestyle='--', label=f'Arjun Persuade')

    ax1.xaxis.grid(False)
    ax1.yaxis.grid(False)

    ax1.set_xticks(major_pos)
    ax1.set_xticklabels(major_lbl, rotation=45, ha='right')

    ax1.set_xlabel('Round', fontsize=14)
    ax1.set_ylabel('Closeness', fontsize=14)
    ax2.set_ylabel('Cumulative Persuasion Rate', fontsize=14)

    ax1.tick_params(axis='x', labelsize=14)
    ax1.tick_params(axis='y', labelsize=14)
    ax2.tick_params(axis='y', labelsize=14)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()

    ax1.legend(h1 + h2, l1 + l2,
               bbox_to_anchor=(0, 1),   
               loc='upper left',
               borderaxespad=0.,          
               fontsize=14,
               frameon=False)


    plt.subplots_adjust(right=0.75)

    fig.tight_layout()
    plt.savefig("highlighted_agents_14_up_to_R22D235.png", format='png', dpi=300)
    plt.show()

if __name__ == "__main__":
    main()


