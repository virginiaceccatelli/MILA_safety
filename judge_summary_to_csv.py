import json
import pandas as pd

json_file = "judge_results/summary.json"
csv_file = "results.csv"
excel_file = "results.xlsx"

with open(json_file, "r") as f:
    data = json.load(f)

rows = []

for model, model_data in data.items():
    for lang, lang_data in model_data.items():

        base_info = {
            "model": model,
            "language": lang,
            "n_total": lang_data.get("n_total"),
            "n_judged": lang_data.get("n_judged"),
            "n_no_response": lang_data.get("n_no_response"),
            "refused_rate": lang_data.get("refused_rate"),
            "jailbroken_rate": lang_data.get("jailbroken_rate"),
            "deflected_rate": lang_data.get("deflected_rate"),
            "refused": lang_data.get("refused"),
            "jailbroken": lang_data.get("jailbroken"),
            "deflected": lang_data.get("deflected"),
        }

        prompt_types = lang_data.get("by_prompt_type", {})

        for prompt_type, stats in prompt_types.items():
            row = base_info.copy()
            row.update({
                "prompt_type": prompt_type,
                "refused_prompt": stats.get("Refused"),
                "jailbroken_prompt": stats.get("Jailbroken"),
                "deflected_prompt": stats.get("Deflected"),
                "prompt_total": stats.get("total"),
            })
            rows.append(row)

df = pd.DataFrame(rows)

df.to_csv(csv_file, index=False)
df.to_excel(excel_file, index=False)

print(f"Saved CSV to {csv_file}")
print(f"Saved Excel to {excel_file}")