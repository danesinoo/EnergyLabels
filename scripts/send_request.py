import requests
import json


def extract_heating_data(text: str, server_url: str = "http://localhost:8000") -> dict:
    try:
        response = requests.post(
            f"{server_url}/extract",
            json={"text": text},
            headers={"Content-Type": "application/json"},
            timeout=10,  # Adjust timeout as needed
        )
        response.raise_for_status()  # Raise error for HTTP failures
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return {"error": str(e)}
    except json.JSONDecodeError:
        print("Failed to parse server response")
        return {"error": "Invalid JSON from server"}


# Usage
if __name__ == "__main__":
    import pandas as pd

    df = pd.read_csv("data/gt.csv")
    i = 84
    line = df.loc[i, "S_text"] + df.loc[i, "L_text"]
    print(df.iloc[i])
    result = extract_heating_data(line)
    print(json.dumps(result, indent=2))
