from orgchart.model import OrgChart, OrgChartEntry


class TestOrgChartModel:
    def test_from_md_string_single_entry(self):
        """Test parsing a single employee from markdown."""
        md_content = """# Org Structure

## Alice Smith
- **Manager:** null
- **Direct Reports:** Bob Johnson
- **Teammates:** null
- **Working on:** Leadership

---
"""
        org_chart = OrgChart.from_md_string(md_content)

        assert len(org_chart.entries) == 1
        entry = org_chart.entries[0]
        assert entry.name == "Alice Smith"
        assert entry.manager is None
        assert entry.direct_reports == ["Bob Johnson"]
        assert entry.teammates is None
        assert entry.working_on == "Leadership"

    def test_from_md_string_multiple_entries(self):
        """Test parsing multiple employees from markdown."""
        md_content = """# Org Structure

## Alice Smith
- **Manager:** null
- **Direct Reports:** Bob Johnson, Charlie Brown
- **Teammates:** null
- **Working on:** Leadership

---

## Bob Johnson
- **Manager:** Alice Smith
- **Direct Reports:** null
- **Teammates:** Charlie Brown
- **Working on:** Engineering

---

## Charlie Brown
- **Manager:** Alice Smith
- **Direct Reports:** null
- **Teammates:** Bob Johnson
- **Working on:** Product

---
"""
        org_chart = OrgChart.from_md_string(md_content)

        assert len(org_chart.entries) == 3

        alice = org_chart.entries[0]
        assert alice.name == "Alice Smith"
        assert alice.manager is None
        assert set(alice.direct_reports) == {"Bob Johnson", "Charlie Brown"}
        assert alice.teammates is None

        bob = org_chart.entries[1]
        assert bob.name == "Bob Johnson"
        assert bob.manager == "Alice Smith"
        assert bob.direct_reports is None
        assert bob.teammates == ["Charlie Brown"]

        charlie = org_chart.entries[2]
        assert charlie.name == "Charlie Brown"
        assert charlie.manager == "Alice Smith"
        assert charlie.direct_reports is None
        assert charlie.teammates == ["Bob Johnson"]

    def test_from_md_string_empty_fields(self):
        """Test parsing with null/empty fields."""
        md_content = """# Org Structure

## John Doe
- **Manager:** null
- **Direct Reports:** null
- **Teammates:** null
- **Working on:** 

---
"""
        org_chart = OrgChart.from_md_string(md_content)

        assert len(org_chart.entries) == 1
        entry = org_chart.entries[0]
        assert entry.name == "John Doe"
        assert entry.manager is None
        assert entry.direct_reports is None
        assert entry.teammates is None
        assert entry.working_on == ""

    def test_to_md_string_single_entry(self):
        """Test serializing a single entry to markdown."""
        org_chart = OrgChart(
            entries=[
                OrgChartEntry(
                    name="Alice Smith",
                    manager=None,
                    direct_reports=["Bob Johnson"],
                    teammates=None,
                    working_on="Leadership",
                )
            ]
        )

        md_output = org_chart.to_md_string()

        assert "# Org Structure" in md_output
        assert "## Alice Smith" in md_output
        assert "- **Manager:** null" in md_output
        assert "- **Direct Reports:** Bob Johnson" in md_output
        assert "- **Teammates:** null" in md_output
        assert "- **Working on:** Leadership" in md_output
        assert "---" in md_output

    def test_to_md_string_multiple_entries(self):
        """Test serializing multiple entries to markdown."""
        org_chart = OrgChart(
            entries=[
                OrgChartEntry(
                    name="Alice Smith",
                    manager=None,
                    direct_reports=["Bob Johnson", "Charlie Brown"],
                    teammates=None,
                    working_on="Leadership",
                ),
                OrgChartEntry(
                    name="Bob Johnson",
                    manager="Alice Smith",
                    direct_reports=None,
                    teammates=["Charlie Brown"],
                    working_on="Engineering",
                ),
            ]
        )

        md_output = org_chart.to_md_string()

        # Check structure
        assert md_output.count("##") == 2
        assert md_output.count("---") == 2

        # Check Alice's section
        assert "## Alice Smith" in md_output
        assert "- **Direct Reports:** Bob Johnson, Charlie Brown" in md_output

        # Check Bob's section
        assert "## Bob Johnson" in md_output
        assert "- **Manager:** Alice Smith" in md_output
        assert "- **Teammates:** Charlie Brown" in md_output

    def test_to_md_string_null_fields(self):
        """Test serializing with null fields produces 'null' string."""
        org_chart = OrgChart(
            entries=[
                OrgChartEntry(
                    name="John Doe",
                    manager=None,
                    direct_reports=None,
                    teammates=None,
                    working_on="",
                )
            ]
        )

        md_output = org_chart.to_md_string()

        # All null fields should be represented as "null"
        assert "- **Manager:** null" in md_output
        assert "- **Direct Reports:** null" in md_output
        assert "- **Teammates:** null" in md_output
        assert "- **Working on:** " in md_output

    def test_roundtrip_md_string(self):
        """Test that parsing and serializing produces consistent output."""
        original_md = """# Org Structure

## Alice Smith
- **Manager:** null
- **Direct Reports:** Bob Johnson, Charlie Brown
- **Teammates:** null
- **Working on:** Leadership

---

## Bob Johnson
- **Manager:** Alice Smith
- **Direct Reports:** null
- **Teammates:** Charlie Brown
- **Working on:** Engineering

---
"""
        # Parse the markdown
        org_chart = OrgChart.from_md_string(original_md)

        # Serialize it back
        serialized_md = org_chart.to_md_string()

        # Parse again
        org_chart_2 = OrgChart.from_md_string(serialized_md)

        # Both should have the same entries
        assert len(org_chart.entries) == len(org_chart_2.entries)

        for entry1, entry2 in zip(org_chart.entries, org_chart_2.entries):
            assert entry1.name == entry2.name
            assert entry1.manager == entry2.manager
            assert entry1.direct_reports == entry2.direct_reports
            assert entry1.teammates == entry2.teammates
            assert entry1.working_on == entry2.working_on

    def test_from_md_string_multiple_direct_reports(self):
        """Test parsing comma-separated direct reports."""
        md_content = """# Org Structure

## Manager
- **Manager:** null
- **Direct Reports:** Alice, Bob, Charlie, Diana
- **Teammates:** null
- **Working on:** Management

---
"""
        org_chart = OrgChart.from_md_string(md_content)

        entry = org_chart.entries[0]
        assert entry.direct_reports == ["Alice", "Bob", "Charlie", "Diana"]

    def test_from_md_string_multiple_teammates(self):
        """Test parsing comma-separated teammates."""
        md_content = """# Org Structure

## Engineer
- **Manager:** CTO
- **Direct Reports:** null
- **Teammates:** Alice, Bob, Charlie
- **Working on:** Backend

---
"""
        org_chart = OrgChart.from_md_string(md_content)

        entry = org_chart.entries[0]
        assert entry.teammates == ["Alice", "Bob", "Charlie"]

    def test_from_md_string_whitespace_handling(self):
        """Test that whitespace in names is properly handled."""
        md_content = """# Org Structure

## Alice Smith
- **Manager:** null
- **Direct Reports:** Bob Johnson , Charlie Brown
- **Teammates:** null
- **Working on:** Leadership

---
"""
        org_chart = OrgChart.from_md_string(md_content)

        entry = org_chart.entries[0]
        # Names should be stripped of whitespace
        assert entry.direct_reports == ["Bob Johnson", "Charlie Brown"]
