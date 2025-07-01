# src/deploy_distilbert.py
import argparse, datetime
from google.cloud import aiplatform

def main(args):
    aiplatform.init(project=args.project, location=args.region)

    model = aiplatform.Model(args.model_id)

    eps = aiplatform.Endpoint.list(
        filter=f'display_name="{args.endpoint_name}"',
        location=args.region
    )
    endpoint = eps[0] if eps else aiplatform.Endpoint.create(
        display_name=args.endpoint_name, sync=True
    )

    model.deploy(
        endpoint          = endpoint,
        machine_type      = "n1-standard-4",
        min_replica_count = args.min_replicas,
        max_replica_count = 1,
        sync=True,
    )
    print("✅ Endpoint LIVE:", endpoint.resource_name)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--project", required=True)
    p.add_argument("--region",  default="us-central1")
    p.add_argument("--model-id", required=True,
                   help="Vertex model resource name, e.g. projects/.../models/123")
    p.add_argument("--endpoint-name", default="steam-sentiment-endpoint")
    p.add_argument("--min-replicas",  type=int, default=0)
    main(p.parse_args())