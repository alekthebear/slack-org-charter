from pathlib import Path
import argparse

from anytree import Node, RenderTree

from orgchart.model import OrgChart


def visualize_org_chart(org_chart: OrgChart) -> None:
    """
    Visualize an organizational chart in ASCII tree format.

    Args:
        org_chart: The OrgChart model to visualize.

    The function displays the org chart as an ASCII tree,
    showing the hierarchical structure with each employee's name, title, and project.
    """
    # Create a mapping from name to entry for easy lookup
    name_to_entry = {entry.name: entry for entry in org_chart.entries}

    # Create a mapping from name to Node for building the tree
    name_to_node = {}

    # Find the root(s) - employees with no manager
    roots = [entry for entry in org_chart.entries if entry.manager is None]

    if not roots:
        print("No root employee found (no one without a manager).")
        return

    def create_node(entry) -> Node:
        """Create a node with formatted label."""
        name = entry.name
        title = getattr(entry, "title", "")
        project = entry.working_on

        # Format the label
        if title and project:
            label = f"{name} ({title} - {project})"
        elif title:
            label = f"{name} ({title})"
        elif project:
            label = f"{name} - {project}"
        else:
            label = name

        return Node(label)

    def build_tree(entry, parent_node: Node = None) -> Node:
        """Recursively build the tree structure."""
        name = entry.name

        # Avoid cycles
        if name in name_to_node:
            return name_to_node[name]

        # Create the node
        node = create_node(entry)
        if parent_node:
            node.parent = parent_node

        name_to_node[name] = node

        # Add direct reports (sorted alphabetically)
        if entry.direct_reports:
            sorted_reports = sorted(entry.direct_reports)
            for report_name in sorted_reports:
                if report_name in name_to_entry:
                    build_tree(name_to_entry[report_name], node)

        return node

    # Build tree for each root
    for root_entry in roots:
        root_node = build_tree(root_entry)

        # Print the tree
        print("\n" + "=" * 80)
        print(f"Organization Chart (Root: {root_entry.name})")
        print("=" * 80 + "\n")

        for pre, _, node in RenderTree(root_node):
            print(f"{pre}{node.name}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize org chart in ASCII format")
    parser.add_argument(
        "org_chart_path",
        type=Path,
        help="Path to the org chart md file",
    )
    args = parser.parse_args()

    org_chart = OrgChart.from_md_file(args.org_chart_path)
    visualize_org_chart(org_chart)
