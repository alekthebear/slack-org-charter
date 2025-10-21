from orgchart.generate import build_org_chart
from inference.user_manager import UserManager
from inference.user_role import UserRole


class TestGenerateOrgChart:
    def test_simple_org_chart(self):
        user_roles = [
            UserRole(
                name="CEO",
                title="Chief Executive Officer",
                project="Leadership",
                reason="",
            ),
            UserRole(
                name="Engineer", title="Software Engineer", project="Product", reason=""
            ),
        ]
        user_managers = [
            UserManager(name="CEO", manager=None, reason=""),
            UserManager(name="Engineer", manager="CEO", reason=""),
        ]

        org_chart = build_org_chart(user_roles, user_managers)

        assert len(org_chart.entries) == 2

        # Check CEO entry (note: names are title-cased in build_org_chart)
        ceo_entry = next(e for e in org_chart.entries if e.name == "Ceo")
        assert ceo_entry.manager is None
        assert ceo_entry.direct_reports == ["Engineer"]
        assert ceo_entry.teammates is None
        assert ceo_entry.working_on == "Leadership"

        # Check Engineer entry
        eng_entry = next(e for e in org_chart.entries if e.name == "Engineer")
        assert eng_entry.manager == "Ceo"
        assert eng_entry.direct_reports is None
        assert eng_entry.teammates is None
        assert eng_entry.working_on == "Product"

    def test_teammates_calculation(self):
        """Test that teammates are correctly identified."""
        user_roles = [
            UserRole(name="Manager", title="Manager", project="Team", reason=""),
            UserRole(name="Engineer1", title="Engineer", project="Product", reason=""),
            UserRole(name="Engineer2", title="Engineer", project="Product", reason=""),
            UserRole(name="Engineer3", title="Engineer", project="Product", reason=""),
        ]
        user_managers = [
            UserManager(name="Manager", manager=None, reason=""),
            UserManager(name="Engineer1", manager="Manager", reason=""),
            UserManager(name="Engineer2", manager="Manager", reason=""),
            UserManager(name="Engineer3", manager="Manager", reason=""),
        ]

        org_chart = build_org_chart(user_roles, user_managers)

        # Check Engineer1's teammates
        eng1 = next(e for e in org_chart.entries if e.name == "Engineer1")
        assert eng1.teammates is not None
        assert set(eng1.teammates) == {"Engineer2", "Engineer3"}

        # Check Engineer2's teammates
        eng2 = next(e for e in org_chart.entries if e.name == "Engineer2")
        assert eng2.teammates is not None
        assert set(eng2.teammates) == {"Engineer1", "Engineer3"}

    def test_multi_level_hierarchy(self):
        """Test a multi-level organizational hierarchy."""
        user_roles = [
            UserRole(name="CEO", title="CEO", project="Leadership", reason=""),
            UserRole(
                name="VP", title="Vice President", project="Engineering", reason=""
            ),
            UserRole(name="Manager", title="Manager", project="Product", reason=""),
            UserRole(name="IC", title="Engineer", project="Product", reason=""),
            UserRole(name="IC2", title="Engineer", project="Product", reason=""),
        ]
        user_managers = [
            UserManager(name="CEO", manager=None, reason=""),
            UserManager(name="VP", manager="CEO", reason=""),
            UserManager(name="Manager", manager="VP", reason=""),
            UserManager(name="IC", manager="Manager", reason=""),
            UserManager(name="IC2", manager="Manager", reason=""),
        ]

        org_chart = build_org_chart(user_roles, user_managers)

        assert len(org_chart.entries) == 5

        # CEO has VP as direct report (note: names are title-cased)
        ceo = next(e for e in org_chart.entries if e.name == "Ceo")
        assert ceo.direct_reports == ["Vp"]

        # VP has Manager as direct report
        vp = next(e for e in org_chart.entries if e.name == "Vp")
        assert vp.direct_reports == ["Manager"]
        assert vp.manager == "Ceo"

        # Manager has IC as direct report
        manager = next(e for e in org_chart.entries if e.name == "Manager")
        assert set(manager.direct_reports) == {"Ic", "Ic2"}
        assert manager.manager == "Vp"

        # IC has no direct reports
        ic = next(e for e in org_chart.entries if e.name == "Ic")
        assert ic.direct_reports is None
        assert ic.teammates == ["Ic2"]
        assert ic.manager == "Manager"
        assert ic.working_on == "Product"
