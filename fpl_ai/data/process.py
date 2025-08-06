def build_weekly():
    static = json.loads((RAW / "bootstrap_static.json").read_text())
    elems  = pd.DataFrame(static["elements"])
    # simple features
    df = elems[["id", "first_name", "second_name",
                "now_cost", "total_points", "minutes",
                "influence", "creativity", "threat"]]
    df["ppg"] = df["total_points"] / 38
    df["target"] = df["ppg"].shift(-1)  # LABEL: naïve next-GW points

    # Keep only numeric columns plus "id"
    numeric = df.select_dtypes(include=[float, int]).columns.tolist()
    df = df[numeric + ["id"]]

    return df
