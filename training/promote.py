"""Make the challenger the champion -- only if it is better, and only when someone runs this.

train.py registers every new winner as `challenger` and stops there. The API serves
`champion`. This script is the review step between the two: it prints both versions side by
side on the held-out MAE and moves `champion` only if the challenger beats it.

Both MAEs are comparable because every run is measured on the same split of the same file,
with the same seed. A champion missing (first deployment) means the challenger is promoted.
"""

import os

from dotenv import load_dotenv
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

load_dotenv()

CHALLENGER = "challenger"
METRIC = "test_mae"   # euros of average error on the held-out cars, lower is better


def describe(client, name, alias):
    """The version an alias points to, its run name and its held-out MAE -- or None."""
    try:
        version = client.get_model_version_by_alias(name, alias)
    except MlflowException:
        return None
    run = client.get_run(version.run_id)
    return {"version": version.version, "run": run.info.run_name,
            "mae": run.data.metrics[METRIC]}


def main():
    name = os.environ["MLFLOW_REGISTERED_MODEL_NAME"]
    champion_alias = os.environ["MLFLOW_MODEL_ALIAS"]   # the alias the API serves

    client = MlflowClient(tracking_uri=os.environ["MLFLOW_TRACKING_URI"])
    challenger = describe(client, name, CHALLENGER)
    champion = describe(client, name, champion_alias)

    if challenger is None:
        print(f"[INFO] No '{CHALLENGER}' alias on '{name}': run training/train.py first.")
        return

    for label, model in [(CHALLENGER, challenger), (champion_alias, champion)]:
        if model is None:
            print(f"[INFO] {label:>10}: none")
        else:
            print(f"[INFO] {label:>10}: version {model['version']:>3}  "
                  f"{METRIC} {model['mae']:.2f} EUR  ({model['run']})")

    if champion is not None and champion["version"] == challenger["version"]:
        print("[INFO] The challenger is already the champion. Nothing to do.")
    elif champion is not None and challenger["mae"] >= champion["mae"]:
        print("[INFO] The challenger is not better. The champion stays.")
    else:
        client.set_registered_model_alias(name, champion_alias, challenger["version"])
        print(f"[INFO] '{champion_alias}' now points to version {challenger['version']}. "
              "Restart the API Space to serve it.")


if __name__ == "__main__":
    main()
