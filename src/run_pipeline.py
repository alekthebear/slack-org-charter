import argparse

from extract.messages import get_all_messages
from features.channel_conventions import get_channel_naming_conventions
from features.channel_features import get_channel_features
from features.user_features import get_user_features
from features.web_search import get_web_search_employees_info
from inference.normalize_user_managers import get_normalized_user_managers
from inference.normalize_user_roles import get_normalized_user_roles, get_user_roles
from inference.user_manager import get_user_managers
from orgchart.generate import generate_org_chart


def run_pipeline(force_refresh: bool = False, org_chart_output: str = None):
    print("=== Extracting Raw Data ===")
    get_all_messages(force_refresh=force_refresh)

    print("=== Features Extraction===")
    print("*** Aggregating Channel Features ***")
    get_channel_features(force_refresh=force_refresh)
    print("*** Aggregating User Features ***")
    get_user_features(force_refresh=force_refresh)
    print("*** Querying Channel Naming Conventions ***")
    get_channel_naming_conventions(force_refresh=force_refresh)
    print("*** Querying Web Search Employees Info ***")
    get_web_search_employees_info(force_refresh=force_refresh)

    print("=== Inference ===")
    print("*** Inferring User Roles ***")
    get_user_roles(force_refresh=force_refresh)
    get_normalized_user_roles(force_refresh=force_refresh)
    print("*** Inferring User Managers ***")
    get_user_managers(force_refresh=force_refresh)
    get_normalized_user_managers(force_refresh=force_refresh)

    print("=== Generate Org Chart ===")
    org_chart = generate_org_chart()
    org_chart.to_md_file(org_chart_output)
    print(f"Org chart saved to: {org_chart_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the pipeline")
    parser.add_argument(
        "--force-refresh",
        default=False,
        action="store_true",
        help="Force refresh the pipeline",
    )
    parser.add_argument(
        "-o",
        "--org-chart-output",
        type=str,
        required=True,
        help="Path to the output org chart file.",
    )
    args = parser.parse_args()
    run_pipeline(
        force_refresh=args.force_refresh,
        org_chart_output=args.org_chart_output,
    )
