import json
import sys

def compute_avg_latency(json_path, start_index=5):
    with open(json_path, "r") as f:
        data = json.load(f)

    samples = data.get("samples", [])

    # Filter samples by index
    filtered_latencies = [
        s["latency_sec"]
        for s in samples
        if s.get("index", -1) >= start_index and "latency_sec" in s
    ]

    if not filtered_latencies:
        raise ValueError("No valid samples found after index threshold.")

    avg_latency = sum(filtered_latencies) / len(filtered_latencies)

    return avg_latency, len(filtered_latencies)


if __name__ == "__main__":

    json_path = "/vol/bitbucket/ns1324/Qwen3-VL/scripts/results/13_01_2025/VCoT/qwen3vl_4b_tooltoken_pre_icl_1000scale_xy_run2.json"

    avg_latency, num_samples = compute_avg_latency(json_path)

    print(f"Average latency (index ≥ 5): {avg_latency:.4f} sec")
    print(f"Number of samples used: {num_samples}")
