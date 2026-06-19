from datetime import datetime
def promote_model(models):
    best_model = max(models, key = lambda m: (m["accuracy"], -m["latency"], datetime.fromisoformat(m["timestamp"])))
    return best_model["name"]