import json
import re
from pathlib import Path
from typing import Self

from pydantic import BaseModel


class OrgChartEntry(BaseModel):
    name: str
    manager: str | None
    direct_reports: list[str] | None
    teammates: list[str] | None
    working_on: str


class OrgChart(BaseModel):
    entries: list[OrgChartEntry]

    @classmethod
    def from_json_file(cls, path: str | Path) -> Self:
        with open(path, "r") as f:
            data = json.load(f)

        # Handle both old format (list) and new format (dict with entries key)
        if isinstance(data, list):
            return cls(entries=data)
        else:
            return cls(**data)

    @classmethod
    def from_md_file(cls, path: str | Path) -> Self:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        return cls.from_md_string(content)

    @classmethod
    def from_json_string(cls, content: str) -> Self:
        data = json.loads(content)
        if isinstance(data, list):
            return cls(entries=data)
        else:
            return cls(**data)

    @classmethod
    def from_md_string(cls, content: str) -> Self:
        # Remove the title line
        content = content.replace("# Org Structure", "").strip()

        # Split by the separator (---) to get individual employee sections
        sections = content.split("---")
        entries = []
        for section in sections:
            section = section.strip()
            entry = _parse_employee_section(section)
            if entry:
                entries.append(entry)

        return cls(entries=entries)

    def to_json_file(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.model_dump(), f, indent=2, ensure_ascii=False)

    def to_md_file(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_md_string())

    def to_json_string(self) -> str:
        return json.dumps(self.model_dump(), indent=2)

    def to_md_string(self) -> str:
        lines = ["# Org Structure", ""]

        for entry in self.entries:
            # Add person name as header
            lines.append(f"## {entry.name}")

            # Add manager
            manager_value = entry.manager if entry.manager else "null"
            lines.append(f"- **Manager:** {manager_value}")

            # Add direct reports
            if entry.direct_reports:
                direct_reports_str = ", ".join(entry.direct_reports)
            else:
                direct_reports_str = "null"
            lines.append(f"- **Direct Reports:** {direct_reports_str}")

            # Add teammates
            if entry.teammates:
                teammates_str = ", ".join(entry.teammates)
            else:
                teammates_str = "null"
            lines.append(f"- **Teammates:** {teammates_str}")

            # Add working on
            lines.append(f"- **Working on:** {entry.working_on}")

            # Add separator
            lines.append("")
            lines.append("---")
            lines.append("")

        return "\n".join(lines)


def _parse_employee_section(section: str) -> OrgChartEntry | None:
    """Parse a single employee section from the markdown."""
    lines = section.strip().split("\n")

    # First line is the name (header with ##)
    name_match = re.match(r"^##\s+(.+)$", lines[0])
    if not name_match:
        return None

    name = name_match.group(1).strip()

    # Initialize fields
    manager = None
    direct_reports = None
    teammates = None
    working_on = ""

    # Parse the bullet points
    for line in lines[1:]:
        line = line.strip()
        if not line.startswith("- **"):
            continue

        # Extract field name and value
        match = re.match(r"^-\s+\*\*([^:]+):\*\*\s+(.*)$", line)
        if not match:
            continue

        field_name = match.group(1).strip()
        value = match.group(2).strip()

        if field_name == "Manager":
            manager = None if value == "null" else value
        elif field_name == "Direct Reports":
            if value == "null" or value == "":
                direct_reports = None
            else:
                direct_reports = [name.strip() for name in value.split(",")]
        elif field_name == "Teammates":
            if value == "null" or value == "":
                teammates = None
            else:
                teammates = [name.strip() for name in value.split(",")]
        elif field_name == "Working on":
            working_on = value

    return OrgChartEntry(
        name=name,
        manager=manager,
        direct_reports=direct_reports,
        teammates=teammates,
        working_on=working_on,
    )
