import argparse

from inference.user_manager import UserManager, get_user_managers
from inference.user_role import UserRole, get_user_roles
from orgchart.model import OrgChart, OrgChartEntry


def build_org_chart(
    user_roles: list[UserRole], user_managers: list[UserManager]
) -> OrgChart:
    # Create lookup dictionaries
    manager_map = {
        um.name.title(): um.manager.title() if um.manager else None
        for um in user_managers
    }
    project_map = {ur.name.title(): ur.project for ur in user_roles}

    # Get all names
    all_names = set(manager_map.keys())

    # Calculate direct reports for each person
    direct_reports_map = {}
    for name in all_names:
        direct_reports_map[name] = []

    for name, manager in manager_map.items():
        if manager and manager in direct_reports_map:
            direct_reports_map[manager].append(name)

    # Calculate teammates for each person (people with the same manager)
    teammates_map = {}
    for name in all_names:
        manager = manager_map.get(name)
        if manager:
            # Find all people with the same manager, excluding self
            teammates = [
                other_name
                for other_name, other_manager in manager_map.items()
                if other_manager == manager and other_name != name
            ]
            teammates_map[name] = teammates if teammates else None
        else:
            teammates_map[name] = None

    # Build the org chart entries
    entries = []
    for name in sorted(all_names):
        entry = OrgChartEntry(
            name=name,
            manager=manager_map.get(name),
            direct_reports=direct_reports_map[name]
            if direct_reports_map[name]
            else None,
            teammates=teammates_map[name],
            working_on=project_map.get(name, ""),
        )
        entries.append(entry)

    return OrgChart(entries=entries)


def generate_org_chart() -> OrgChart:
    user_roles = get_user_roles()
    user_managers = get_user_managers()
    org_chart = build_org_chart(user_roles, user_managers)
    return org_chart


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate an organizational chart from user roles and manager data."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="org_chart.md",
        help="Output file path for the org chart (default: org_chart.md).",
    )
    parser.add_argument(
        "-f",
        "--format",
        type=str,
        default="md",
        choices=["md", "json"],
        help="Output format: 'md' for markdown or 'json' (default: md).",
    )
    args = parser.parse_args()
    org_chart = generate_org_chart()
    if args.format == "md":
        org_chart.to_md_file(args.output)
    elif args.format == "json":
        org_chart.to_json_file(args.output)
    else:
        raise ValueError(
            f"Invalid output format: {args.format}. Must be 'md' or 'json'."
        )
    print(f"Org chart saved to: {args.output}")
