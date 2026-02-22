import json
import os
import sys
import pandas as pd

def main():
    path = "data/cities/models_v7_sweep/sweep_results.json"
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return

    with open(path, "r") as f:
        data = json.load(f)

    # Sort by mean_r2 descending
    data = sorted(data, key=lambda x: x.get('mean_r2', -999), reverse=True)

    print("=== Top 5 Models ===")
    top5 = data[:5]
    for i, m in enumerate(top5):
        print(f"{i+1}. {m['name']:30s} | R2: {m['mean_r2']:.4f} | Params: {m['n_params']:,} | Dropout: {m['dropout']}")

    print("\n=== Per-City Comparison (Top 5) ===")
    
    # Let's collect city R2s for top 5
    cities = []
    for m in top5:
        if 'per_city' in m:
            for city in m['per_city']:
                if city not in cities:
                    cities.append(city)
    
    cities.sort()
    
    city_data = []
    for city in cities:
        row = {'City': city}
        for i, m in enumerate(top5):
            r2 = None
            if 'per_city' in m and city in m['per_city']:
                r2 = m['per_city'][city].get('r2', None)
            row[f"Model_{i+1}"] = r2
        city_data.append(row)
        
    df = pd.DataFrame(city_data)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(df.to_string(index=False))
    
    print("\n=== Strict Superiority Check ===")
    m1_scores = df["Model_1"].values
    
    # Comparing Model 1 to Model 2, 3, 4, 5
    for i in range(2, 6):
        col = f"Model_{i}"
        if col in df.columns:
            m_scores = df[col].values
            
            better_count = (m1_scores > m_scores).sum()
            worse_count = (m1_scores < m_scores).sum()
            tie_count = (m1_scores == m_scores).sum()
            
            print(f"Model 1 vs Model {i}:")
            print(f"  Model 1 better on {better_count} cities")
            print(f"  Model 1 worse on {worse_count} cities")
            print(f"  Ties on {tie_count} cities")
            if worse_count > 0:
                print(f"  -> Model 1 is NOT strictly better than Model {i}.")
                # which cities?
                worse_cities = df[df["Model_1"] < df[col]]["City"].tolist()
                print(f"     It is worse on: {', '.join(worse_cities)}")
            else:
                print(f"  -> Model 1 IS strictly better (or equal) than Model {i}.")
            print()

if __name__ == "__main__":
    main()
